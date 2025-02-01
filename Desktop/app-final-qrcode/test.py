import warnings
import os
import re
import io
import json
import logging
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    Image as RLImage,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import DeepSeek-style client from openai package
try:
    from openai import OpenAI
except ImportError:
    st.error("Please install the openai library: pip3 install openai")
    raise

# ----------------------------------------------------------------------------
# 1) BASIC SETUP
# ----------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI (DeepSeek) API Key (from Streamlit secrets)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("API key for OpenAI/DeepSeek not found in secrets.")
    OPENAI_API_KEY = None

# Initialize the DeepSeek (OpenAI-like) client if we have the key
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

# API Endpoints
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
API_TEACHER_WEAK_CONCEPTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts"
API_BASELINE_REPORT = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Baseline_Report_Single_Student"
API_ALL_CONCEPTS_URL = "https://webapi.edubull.com/api/eProfessor/eProf_Org_ConceptList_Single_Student"  # New API Endpoint for All Concepts

# Initialize session state variables
if "auth_data" not in st.session_state:
    st.session_state.auth_data = None
if "selected_concept_id" not in st.session_state:
    st.session_state.selected_concept_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_teacher" not in st.session_state:
    st.session_state.is_teacher = False
if "topic_id" not in st.session_state:
    st.session_state.topic_id = None
if "teacher_weak_concepts" not in st.session_state:
    st.session_state.teacher_weak_concepts = []
if "selected_batch_id" not in st.session_state:
    st.session_state.selected_batch_id = None
if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = ""
if "learning_path_generated" not in st.session_state:
    st.session_state.learning_path_generated = False
    st.session_state.learning_path = None
if "generated_description" not in st.session_state:
    st.session_state.generated_description = ""
if "is_english_mode" not in st.session_state:
    st.session_state.is_english_mode = False
if "student_learning_paths" not in st.session_state:
    st.session_state.student_learning_paths = {}
if "student_weak_concepts" not in st.session_state:
    st.session_state.student_weak_concepts = []
if "available_concepts" not in st.session_state:
    st.session_state.available_concepts = {}
if "baseline_data" not in st.session_state:
    st.session_state.baseline_data = None
if "subject_id" not in st.session_state:
    st.session_state.subject_id = None  # Default if unknown
if "user_id" not in st.session_state:
    st.session_state.user_id = None  # Initialize UserID
if "all_concepts" not in st.session_state:
    st.session_state.all_concepts = []  # Initialize All Concepts
if "remedial_info" not in st.session_state:
    st.session_state.remedial_info = None
if 'show_gap_message' not in st.session_state:
    st.session_state.show_gap_message = False

# Define the show_gap_message function globally
def show_gap_message():
    st.session_state.show_gap_message = True

# Streamlit page config
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Hide default Streamlit UI
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ----------------------------------------------------------------------------

# ------------------- 2A) LATEX TO IMAGE -------------------
def latex_to_image(latex_code, dpi=300):
    """
    Converts LaTeX code to PNG and returns it as a BytesIO object.
    """
    try:
        plt.figure(figsize=(0.01, 0.01))
        plt.text(0.5, 0.5, f"${latex_code}$", fontsize=12, ha='center', va='center')
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1, transparent=True)
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error converting LaTeX to image: {e}")
        return None

# ------------------- 2B) FETCHING RESOURCES -------------------
def get_matching_resources(concept_text, concept_list, topic_id):
    def clean_text(text):
        return text.lower().strip().replace(" ", "")

    matching_concept = next(
        (c for c in concept_list if clean_text(c['ConceptText']) == clean_text(concept_text)),
        None
    )
    if matching_concept:
        content_payload = {
            'TopicID': topic_id,
            'ConceptID': int(matching_concept['ConceptID'])
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        try:
            response = requests.post(API_CONTENT_URL, json=content_payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching resources: {e}")
            return None
    return None

def get_resources_for_concept(concept_text, concept_list, topic_id):
    return get_matching_resources(concept_text, concept_list, topic_id)

def format_resources_message(resources):
    """
    Format resources data into a chat-friendly message.
    """
    if not resources:
        return "No remedial resources available for this concept."

    message = ""

    if resources.get("Video_List"):
        message += "**🎥 Video Lectures:**\n"
        for video in resources["Video_List"]:
            video_url = f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}"
            title = video.get('LectureTitle', 'Video Lecture')
            message += f"- [{title}]({video_url})\n"
        message += "\n"

    if resources.get("Notes_List"):
        message += "**📄 Study Notes:**\n"
        for note in resources["Notes_List"]:
            note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
            title = note.get('NotesTitle', 'Study Notes')
            message += f"- [{title}]({note_url})\n"
        message += "\n"

    if resources.get("Exercise_List"):
        message += "**📝 Practice Exercises:**\n"
        for exercise in resources["Exercise_List"]:
            exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
            title = exercise.get('ExerciseTitle', 'Practice Exercise')
            message += f"- [{title}]({exercise_url})\n"

    return message

# ------------------- 2C) PDF GENERATION -------------------
def generate_exam_questions_pdf(questions, concept_text, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        alignment=TA_LEFT,
        spaceAfter=8
    )
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    story.append(Paragraph("Exam Questions", title_style))
    user_name_display = user_name if user_name else "Teacher"
    concept_text_display = concept_text if concept_text else "Selected Concept"
    story.append(Paragraph(f"For {user_name_display} - {concept_text_display}", subtitle_style))
    story.append(Spacer(1, 12))

    sections = re.split(r'\n\n', questions.strip())
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        story.append(Paragraph(lines[0], section_title_style))
        story.append(Spacer(1, 8))

        question_items = []
        for line in lines[1:]:
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
                last_index = 0
                for match in latex_matches:
                    if match.group(1):
                        latex = match.group(1).strip()
                        display_math = True
                    else:
                        latex = match.group(2).strip()
                        display_math = False

                    if latex:
                        pre_text = line[last_index:match.start()]
                        if pre_text:
                            question_items.append(ListItem(Paragraph(pre_text, question_style)))

                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            question_items.append(ListItem(img))
                        last_index = match.end()

                post_text = line[last_index:]
                if post_text:
                    question_items.append(ListItem(Paragraph(post_text, question_style)))
            else:
                question_items.append(ListItem(Paragraph(line, question_style)))

        story.append(ListFlowable(question_items, bulletType='1'))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_learning_path(concept_text):
    """
    Generates a learning path using DeepSeek Chat. 
    Replace the prompt/model as needed for your scenario.
    """
    if not client:
        st.error("DeepSeek client is not initialized. Check your API key.")
        return None

    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    prompt = (
        f"You are a highly experienced educational AI assistant specializing in the NCERT curriculum. "
        f"A student in {branch_name} is struggling with the weak concept: '{concept_text}'. "
        f"Please create a structured, step-by-step learning path tailored to {branch_name} students, "
        f"ensuring clarity, engagement, and curriculum alignment.\n\n"
        f"Sections:\n1. **Introduction**\n2. **Step-by-Step Learning**\n3. **Engagement**\n"
        f"4. **Real-World Applications**\n5. **Practice Problems**\n\n"
        f"All math expressions must be in LaTeX."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the DeepSeek model name
            messages=[{"role": "system", "content": prompt}],
            stream=False,
            max_tokens=1500
        )
        gpt_response = response.choices[0].message.content.strip()
        return gpt_response
    except Exception as e:
        st.error(f"Error generating learning path: {e}")
        return None

def generate_learning_path_pdf(learning_path, concept_text, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    content_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    story.append(Paragraph("Personalized Learning Path", title_style))
    user_name_display = user_name if user_name else "Student"
    concept_text_display = concept_text if concept_text else "Selected Concept"
    story.append(Paragraph(f"For {user_name_display} - {concept_text_display}", subtitle_style))
    story.append(Spacer(1, 12))

    sections = re.split(r'\n\n', learning_path.strip())
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        story.append(Paragraph(lines[0], styles['Heading3']))
        story.append(Spacer(1, 6))

        for line in lines[1:]:
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
                last_index = 0
                for match in latex_matches:
                    if match.group(1):
                        latex = match.group(1).strip()
                        display_math = True
                    else:
                        latex = match.group(2).strip()
                        display_math = False

                    if latex:
                        pre_text = line[last_index:match.start()]
                        if pre_text:
                            story.append(Paragraph(pre_text, content_style))

                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            story.append(img)
                        last_index = match.end()

                post_text = line[last_index:]
                if post_text:
                    story.append(Paragraph(post_text, content_style))
            else:
                story.append(Paragraph(line, content_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ------------------- 2D) LEARNING PATH GENERATION -------------------
def display_learning_path_with_resources(concept_text, learning_path, concept_list, topic_id):
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    with st.expander(f"📚 Learning Path for {concept_text} (Grade: {branch_name})", expanded=False):
        st.markdown(learning_path, unsafe_allow_html=True)

        resources = get_matching_resources(concept_text, concept_list, topic_id)
        if resources:
            st.markdown("### 📌 Additional Learning Resources")
            if resources.get("Video_List"):
                st.markdown("#### 🎥 Video Lectures")
                for video in resources["Video_List"]:
                    video_url = f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}"
                    st.markdown(f"- [{video.get('LectureTitle', 'Video Lecture')}]({video_url})")
            if resources.get("Notes_List"):
                st.markdown("#### 📄 Study Notes")
                for note in resources["Notes_List"]:
                    note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                    st.markdown(f"- [{note.get('NotesTitle', 'Study Notes')}]({note_url})")
            if resources.get("Exercise_List"):
                st.markdown("#### 📝 Practice Exercises")
                for exercise in resources["Exercise_List"]:
                    exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                    st.markdown(f"- [{exercise.get('ExerciseTitle', 'Practice Exercise')}]({exercise_url})")

        pdf_bytes = generate_learning_path_pdf(
            learning_path,
            concept_text,
            st.session_state.auth_data['UserInfo'][0]['FullName']
        )
        st.download_button(
            label="📥 Download Learning Path as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.auth_data['UserInfo'][0]['FullName']}_Learning_Path_{concept_text}.pdf",
            mime="application/pdf"
        )

# ------------------- 2E) FETCH ALL CONCEPTS -------------------
def fetch_all_concepts(org_code, subject_id, user_id):
    payload = {
        "OrgCode": org_code,
        "SubjectID": subject_id,
        "UserID": user_id
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        response = requests.post(API_ALL_CONCEPTS_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching all concepts: {e}")
        return None

# ------------------- 2F) PDF GENERATION FOR ALL CONCEPTS -------------------
def generate_all_concepts_pdf(concepts, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Heading4'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=6
    )

    story.append(Paragraph("All Concepts", title_style))
    user_name_display = user_name if user_name else "User"
    story.append(Paragraph(f"User: {user_name_display}", subtitle_style))
    story.append(Spacer(1, 12))

    headers = ["Concept ID", "Concept Text", "Topic ID", "Status"]
    table_data = [headers]

    for concept in concepts:
        row = [
            str(concept.get("ConceptID", "")),
            concept.get("ConceptText", ""),
            str(concept.get("TopicID", "")),
            concept.get("ConceptStatus", "")
        ]
        table_data.append(row)

    table = Table(table_data, repeatRows=1, colWidths=[1.2*inch, 3*inch, 1.2*inch, 1*inch])
    table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), '#4CAF50'),
        ('TEXTCOLOR', (0,0), (-1,0), '#FFFFFF'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), '#F9F9F9'),
        ('GRID', (0,0), (-1,-1), 1, '#DDDDDD'),
    ])
    table.setStyle(table_style)

    story.append(table)
    story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ------------------- 2G) FETCH REMEDIAL RESOURCES -------------------
def fetch_remedial_resources(topic_id, concept_id):
    remedial_api_url = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
    payload = {
        "TopicID": topic_id,
        "ConceptID": concept_id
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        response = requests.post(remedial_api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching remedial resources: {e}")
        return None

# ------------------- 2H) FORMAT REMEDIAL RESOURCES -------------------
def format_remedial_resources(resources):
    if not resources:
        return "No remedial resources available for this concept."

    message = ""

    if resources.get("Video_List"):
        message += "**🎥 Video Lectures:**\n"
        for video in resources["Video_List"]:
            video_url = f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}"
            title = video.get('LectureTitle', 'Video Lecture')
            message += f"- [{title}]({video_url})\n"
        message += "\n"

    if resources.get("Notes_List"):
        message += "**📄 Study Notes:**\n"
        for note in resources["Notes_List"]:
            note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
            title = note.get('NotesTitle', 'Study Notes')
            message += f"- [{title}]({note_url})\n"
        message += "\n"

    if resources.get("Exercise_List"):
        message += "**📝 Practice Exercises:**\n"
        for exercise in resources["Exercise_List"]:
            exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
            title = exercise.get('ExerciseTitle', 'Practice Exercise')
            message += f"- [{title}]({exercise_url})\n"

    return message

# ----------------------------------------------------------------------------
# 3) BASELINE TESTING REPORT (MODIFIED)
# ----------------------------------------------------------------------------
def fetch_baseline_data(org_code, subject_id, user_id):
    payload = {
        "UserID": user_id,
        "SubjectID": subject_id,
        "OrgCode": org_code
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        with st.spinner("EeeBee is waking up..."):
            response = requests.post(API_BASELINE_REPORT, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error fetching baseline data: {e}")
        return None

def baseline_testing_report():
    if not st.session_state.baseline_data:
        user_info = st.session_state.auth_data.get('UserInfo', [{}])[0]
        user_id = user_info.get('UserID')
        org_code = user_info.get('OrgCode', '012')
        
        subject_id = st.session_state.subject_id
        if not subject_id:
            st.error("Subject ID not available")
            return

        st.session_state.baseline_data = fetch_baseline_data(
            org_code=org_code,
            subject_id=subject_id,
            user_id=user_id
        )
    
    baseline_data = st.session_state.baseline_data
    if not baseline_data:
        st.warning("No baseline data available.")
        return

    u_list = baseline_data.get("u_list", [])
    s_skills = baseline_data.get("s_skills", [])
    concept_wise_data = baseline_data.get("concept_wise_data", [])
    taxonomy_list = baseline_data.get("taxonomy_list", [])

    if u_list:
        user_summary = u_list[0]
        st.markdown("### Overall Performance Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Name", user_summary.get("FullName"))
        col2.metric("Subject", user_summary.get("SubjectName"))
        col3.metric("Batch", user_summary.get("BatchName"))
        col4.metric("Attempt Date", user_summary.get("AttendDate"))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Marks (%)", f"{user_summary.get('MarksPercent', 0)}%")
        col2.metric("Total Concepts", user_summary.get("TotalQuestion"))
        col3.metric("Cleared Concepts", user_summary.get("CorrectQuestion"))
        col4.metric("Weak Concepts", user_summary.get("WeakConceptCount"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Difficult Ques. (%)", f"{user_summary.get('DiffQuesPercent', 0)}%")
        col2.metric("Easy Ques. (%)", f"{user_summary.get('EasyQuesPercent', 0)}%")
        duration_hh = user_summary.get("DurationHH", 0)
        duration_mm = user_summary.get("DurationMM", 0)
        col3.metric("Time Taken", f"{duration_hh}h {duration_mm}m")

    st.markdown("---")
    st.markdown("### Skill-wise Performance")
    if s_skills:
        df_skills = pd.DataFrame(s_skills)
        skill_chart = alt.Chart(df_skills).mark_bar().encode(
            x=alt.X('RightAnswerPercent:Q', title='Correct %', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('SubjectSkillName:N', sort='-x'),
            tooltip=['SubjectSkillName:N', 'TotalQuestion:Q', 'RightAnswerCount:Q', 'RightAnswerPercent:Q']
        ).properties(
            width=700,
            height=400,
            title="Skill-wise Correct Percentage"
        )
        st.altair_chart(skill_chart, use_container_width=True)
    else:
        st.info("No skill-wise data available.")

    st.markdown("---")
    st.markdown("### Concept-wise Performance")
    if concept_wise_data:
        df_concepts = pd.DataFrame(concept_wise_data).copy()
        df_concepts["S.No."] = range(1, len(df_concepts) + 1)
        df_concepts["Concept Status"] = df_concepts["RightAnswerPercent"].apply(
            lambda x: "✅" if x == 100.0 else "❌"
        )
        df_concepts.rename(columns={"ConceptText": "Concept Name", "BranchName": "Class"}, inplace=True)
        df_display = df_concepts[["S.No.", "Concept Name","Concept Status", "Class"]]
        st.dataframe(df_display, hide_index=True)
    else:
        st.info("No concept-wise data available.")

    st.markdown("---")
    st.markdown("### Bloom’s Taxonomy Performance")
    if taxonomy_list:
        df_taxonomy = pd.DataFrame(taxonomy_list)
        tax_chart = alt.Chart(df_taxonomy).mark_bar().encode(
            x=alt.X('PercentObt:Q', title='Percent Correct', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('TaxonomyText:N', sort='-x', title="Bloom's Level"),
            color=alt.Color('TaxonomyText:N', legend=alt.Legend(title="Bloom's Level")),
            tooltip=['TaxonomyText:N', 'TotalQuestion:Q', 'CorrectAnswer:Q', 'PercentObt:Q']
        ).properties(
            width=700,
            height=300,
            title="Performance by Bloom's Taxonomy Level"
        )
        st.altair_chart(tax_chart, use_container_width=True)
    else:
        st.info("No taxonomy data available.")

# ------------------- 2I) ALL CONCEPTS TAB -------------------
def display_all_concepts_tab():
    st.markdown("### 📌 EeeBee is generating remedials according to your current gaps.")
    all_concepts = st.session_state.all_concepts
    if not all_concepts:
        st.warning("No concepts found.")
        return
    
    col_widths = [3, 1, 1.5, 1.5]
    headers = ["Concept Text", "Status", "Remedial", "Previous Learning GAP"]
    header_columns = st.columns(col_widths)
    for idx, header in enumerate(headers):
        header_columns[idx].markdown(f"**{header}**")
    
    concepts_to_fetch = [
        {
            "concept_id": concept['ConceptID'],
            "concept_text": concept['ConceptText'],
            "topic_id": concept['TopicID'],
            "status": concept['ConceptStatus']
        }
        for concept in all_concepts
    ]
    
    @st.cache_data(show_spinner=False)
    def cached_fetch_remedial_resources(topic_id, concept_id):
        return fetch_remedial_resources(topic_id, concept_id)
    
    def fetch_resources(concept):
        if concept['status'] in ["Weak", "Not-Attended"]:
            resources = cached_fetch_remedial_resources(concept['topic_id'], concept['concept_id'])
            formatted_resources = format_remedial_resources(resources)
        else:
            formatted_resources = "-"
        return (concept, formatted_resources)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_concept = {executor.submit(fetch_resources, c): c for c in concepts_to_fetch}
        for future in as_completed(future_to_concept):
            concept, remedial = future.result()
            concept_text = concept['concept_text']
            status = concept['status']
            status_color = 'red' if status == 'Weak' else 'green' if status == 'Cleared' else 'orange'
            status_icon = '🔴' if status == 'Weak' else '🟢' if status == 'Cleared' else '🟠'
            status_html = f"<span style='color:{status_color};'>{status_icon} {status}</span>"
            
            row_columns = st.columns(col_widths)
            row_columns[0].markdown(concept_text)
            row_columns[1].markdown(status_html, unsafe_allow_html=True)
            with row_columns[2]:
                if remedial != "-":
                    with st.expander("🧠 Remedial Resources"):
                        st.markdown(remedial)
                else:
                    st.markdown("-")
            with row_columns[3]:
                if status in ["Weak", "Not-Attended"]:
                    st.button("Previous GAP", key=f"gap_{concept['concept_id']}", on_click=show_gap_message)
                else:
                    st.markdown("-")

# ----------------------------------------------------------------------------
# 4) TEACHER DASHBOARD
# ----------------------------------------------------------------------------
def display_additional_graphs(weak_concepts):
    df = pd.DataFrame(weak_concepts)
    total_attended = df["AttendedStudentCount"].sum()
    total_cleared = df["ClearedStudentCount"].sum()
    total_not_cleared = total_attended - total_cleared

    data_overall = pd.DataFrame({
        'Category': ['Cleared', 'Not Cleared'],
        'Count': [total_cleared, total_not_cleared]
    })
    donut_chart = alt.Chart(data_overall).mark_arc(innerRadius=50).encode(
        theta='Count:Q',
        color=alt.Color('Category:N', legend=alt.Legend(title="Category")),
        tooltip=['Category:N', 'Count:Q']
    ).properties(
        title='Overall Cleared vs Not Cleared Students'
    )
    st.altair_chart(donut_chart, use_container_width=True)

    df_long = df.melt(
        id_vars='ConceptText',
        value_vars=['AttendedStudentCount', 'ClearedStudentCount'],
        var_name='Category',
        value_name='Count'
    )
    df_long['Category'] = df_long['Category'].replace({
        'AttendedStudentCount': 'Attended',
        'ClearedStudentCount': 'Cleared'
    })

    horizontal_bar = alt.Chart(df_long).mark_bar().encode(
        x=alt.X('Count:Q'),
        y=alt.Y('ConceptText:N', sort='-x', title='Concepts'),
        color=alt.Color('Category:N', legend=alt.Legend(title="Category")),
        tooltip=['ConceptText:N', 'Category:N', 'Count:Q']
    ).properties(
        title='Attended vs Cleared per Concept (Horizontal View)',
        width=600
    )
    st.altair_chart(horizontal_bar, use_container_width=True)

def teacher_dashboard():
    batches = st.session_state.auth_data.get("BatchList", [])
    if not batches:
        st.warning("No batches found for the teacher.")
        return

    batch_options = {b['BatchName']: b for b in batches}
    selected_batch_name = st.selectbox("Select a Batch:", list(batch_options.keys()))
    selected_batch = batch_options.get(selected_batch_name)
    selected_batch_id = selected_batch["BatchID"]
    total_students = selected_batch.get("StudentCount", 0)

    if selected_batch_id and st.session_state.selected_batch_id != selected_batch_id:
        st.session_state.selected_batch_id = selected_batch_id
        user_info = st.session_state.auth_data.get('UserInfo', [{}])[0]
        org_code = user_info.get('OrgCode', '012')
        params = {
            "BatchID": selected_batch_id,
            "TopicID": st.session_state.topic_id,
            "OrgCode": org_code
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        with st.spinner("EeeBee is fetching weak concepts..."):
            try:
                response = requests.post(API_TEACHER_WEAK_CONCEPTS, json=params, headers=headers)
                response.raise_for_status()
                weak_concepts = response.json()
                st.session_state.teacher_weak_concepts = weak_concepts
            except Exception as e:
                st.error(f"Error fetching weak concepts: {e}")
                st.session_state.teacher_weak_concepts = []

    if st.session_state.teacher_weak_concepts:
        df = []
        for wc in st.session_state.teacher_weak_concepts:
            df.append({
                "Concept": wc["ConceptText"],
                "Attended": wc["AttendedStudentCount"],
                "Cleared": wc["ClearedStudentCount"]
            })
        df = pd.DataFrame(df)

        df_long = df.melt('Concept', var_name='Category', value_name='Count')
        chart = alt.Chart(df_long).mark_bar().encode(
            x='Concept:N',
            y='Count:Q',
            color=alt.Color('Category:N', legend=alt.Legend(title="Category")),
            tooltip=['Concept:N', 'Category:N', 'Count:Q']
        ).properties(
            title='Weak Concepts Overview',
            width=600
        )

        rule = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_rule(
            color='red', strokeDash=[4, 4]
        ).encode(y='y:Q')
        text = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_text(
            align='left', dx=5, dy=-5, color='red'
        ).encode(y='y:Q', text=alt.value(f'Total Students: {total_students}'))

        final_chart = (chart + rule + text).interactive()
        st.altair_chart(final_chart, use_container_width=True)

        display_additional_graphs(st.session_state.teacher_weak_concepts)

        bloom_level = st.radio(
            "Select Bloom's Taxonomy Level for the Questions",
            [
                "L1 (Remember)",
                "L2 (Understand)",
                "L3 (Apply)",
                "L4 (Analyze)",
                "L5 (Evaluate)"
            ],
            index=3
        )
        bloom_short = bloom_level.split()[0]

        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.radio("Select a Concept to Generate Exam Questions:", list(concept_list.keys()))

        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions"):
                if not client:
                    st.error("DeepSeek client is not initialized. Check your API key.")
                    return

                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                prompt = (
                    f"You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {st.session_state.auth_data.get('TopicName', 'Unknown Topic')}.\n\n"
                    f"Teacher Mode Instructions:\n"
                    f"- The user is a teacher instructing {branch_name} students under the NCERT curriculum.\n"
                    f"- Provide detailed suggestions on how to explain concepts and design assessments for the {branch_name} level.\n"
                    f"- Offer insights into common student difficulties and ways to address them.\n"
                    f"- Encourage a teaching methodology where students learn progressively, asking guiding questions rather than providing direct answers.\n"
                    f"- Do not provide final solutions, only the questions.\n\n"
                    f"Now, generate a set of 20 exam questions for the concept '{chosen_concept_text}' at Bloom's Taxonomy **{bloom_short}**.\n"
                    f"Label each question clearly with **({bloom_short})** and use LaTeX for any math.\n"
                )

                with st.spinner("Generating exam questions... Please wait."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=4000,
                            stream=False
                        )
                        questions = response.choices[0].message.content.strip()
                        st.session_state.exam_questions = questions
                        st.success("Exam questions generated successfully!")
                        
                        st.markdown("### 📝 Generated Exam Questions")
                        st.markdown(questions.replace("\n", "<br>"), unsafe_allow_html=True)
                        
                        pdf_bytes = generate_exam_questions_pdf(
                            questions,
                            chosen_concept_text,
                            st.session_state.auth_data['UserInfo'][0]['FullName']
                        )
                        st.download_button(
                            label="📥 Download Exam Questions as PDF",
                            data=pdf_bytes,
                            file_name=f"{st.session_state.auth_data['UserInfo'][0]['FullName']}_Exam_Questions_{chosen_concept_text}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

# ------------------- 2J) CHAT FUNCTIONS WITH AUTO SCROLLING -------------------
# Stream GPT responses chunk by chunk.
def get_gpt_response_stream(user_input):
    if not client:
        yield "DeepSeek client not initialized."
        return

    system_prompt = get_system_prompt()
    conversation_history = [{"role": "system", "content": system_prompt}]
    conversation_history += [
        {"role": role, "content": content}
        for role, content in st.session_state.chat_history
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            max_tokens=2000,
            stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            text_chunk = getattr(delta, "content", "")
            if text_chunk is None:
                text_chunk = ""
            yield text_chunk
    except Exception as e:
        yield f"Error: {e}"

# Modified handler so that the user message appears immediately and streaming occurs within the chat container.
def handle_user_input(user_input, streaming_placeholder):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        assistant_reply = ""
        for chunk in get_gpt_response_stream(user_input):
            if chunk is None:
                chunk = ""
            assistant_reply += chunk
            streaming_placeholder.markdown(
                f"<div style='text-align:left; color:#000; background-color:#e0e7ff; padding:8px; border-radius:8px; margin-bottom:5px;'><b>EeeBee:</b> {assistant_reply}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.05)
        st.session_state.chat_history.append(("assistant", assistant_reply))
        st.experimental_rerun()

# The chat display now uses a container for messages with streaming updates appearing inside it.
def display_chat(user_name: str):
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                st.markdown(
                    f"<div style='text-align:left; color:#000; background-color:#e0e7ff; padding:8px; border-radius:8px; margin-bottom:5px;'><b>EeeBee:</b> {message}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:left; color:#fff; background-color:#2563eb; padding:8px; border-radius:8px; margin-bottom:5px;'><b>{user_name}:</b> {message}</div>",
                    unsafe_allow_html=True
                )
        # This placeholder will be updated with streaming reply text.
        streaming_placeholder = st.empty()
    user_input = st.chat_input("Enter your question about the topic")
    if user_input:
        handle_user_input(user_input, streaming_placeholder)

def get_system_prompt():
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    if st.session_state.is_teacher:
        return f"""
You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {topic_name}.

Teacher Mode Instructions:
- The user is a teacher instructing {branch_name} students under the NCERT curriculum.
- Provide detailed suggestions on how to explain concepts and design assessments for the {branch_name} level.
- Offer insights into common student difficulties and ways to address them.
- Encourage a teaching methodology where students learn progressively.
- Keep all mathematical expressions within LaTeX delimiters.
- Focus on helping teachers design effective teaching strategies and assessments.
"""
    else:
        weak_concepts = [concept['ConceptText'] for concept in st.session_state.student_weak_concepts]
        weak_concepts_text = ", ".join(weak_concepts) if weak_concepts else "none"

        return f"""
You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {topic_name}.

Student Mode Instructions:
- The student is in {branch_name}, following the NCERT curriculum.
- The student's weak concepts include: {weak_concepts_text}.
- Always provide the list of weak concepts as: [{weak_concepts_text}].
- Encourage the student to solve problems step-by-step and think critically.
- Avoid giving answers. Instead, ask guiding questions and offer hints.
- Your job is to help the student reach the answers on their own.
- If asked for exam or practice questions, present them progressively, aligned with {branch_name} NCERT guidelines.
- All mathematical expressions must be enclosed in LaTeX delimiters ($...$ or $$...$$).
"""

# ----------------------------------------------------------------------------
# 5) AUTHENTICATION SYSTEM WITH MODE-SPECIFIC HANDLING
# ----------------------------------------------------------------------------
def verify_auth_response(auth_data, is_english_mode):
    if not auth_data:
        return False, None, "No authentication data received"
        
    if auth_data.get("statusCode") != 1:
        return False, None, "Authentication failed - invalid status code"
    
    if is_english_mode:
        return True, None, None
    
    subject_id = auth_data.get("SubjectID")
    if subject_id is None:
        subject_id = auth_data.get("UserInfo", [{}])[0].get("SubjectID")
        if subject_id is None:
            return False, None, "Subject ID not found in authentication response"
    
    return True, subject_id, None

def enhanced_login(org_code, login_id, password, topic_id, is_english_mode, user_type_value=3):
    api_url = API_AUTH_URL_ENGLISH if is_english_mode else API_AUTH_URL_MATH_SCIENCE
    
    auth_payload = {
        'OrgCode': org_code,
        'TopicID': int(topic_id),
        'LoginID': login_id,
        'Password': password,
    }
    
    if not is_english_mode:
        auth_payload['UserType'] = user_type_value
        
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(api_url, json=auth_payload, headers=headers)
        response.raise_for_status()
        auth_data = response.json()
        logging.info(f"Authentication Response: {auth_data}")
        
        is_valid, subject_id, error_msg = verify_auth_response(auth_data, is_english_mode)
        if not is_valid:
            return False, error_msg
            
        st.session_state.auth_data = auth_data
        st.session_state.is_authenticated = True
        st.session_state.topic_id = int(topic_id)
        st.session_state.is_english_mode = is_english_mode
        
        st.session_state.is_teacher = (user_type_value == 2)
        
        if not is_english_mode:
            st.session_state.subject_id = subject_id
            user_info = auth_data.get("UserInfo", [{}])[0]
            st.session_state.user_id = user_info.get("UserID")
            
            if user_type_value == 3:  # Student
                st.session_state.student_weak_concepts = auth_data.get("WeakConceptList", [])
        else:
            user_info = auth_data.get("UserInfo", [{}])[0]
            st.session_state.user_id = user_info.get("UserID")
        
        return True, None
            
    except requests.exceptions.RequestException as e:
        logging.error(f"API Request failed: {e}")
        return False, f"API Request failed: {str(e)}"
    except ValueError as e:
        logging.error(f"Invalid JSON response: {e}")
        return False, f"Invalid JSON response: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False, f"Unexpected error: {str(e)}"

def login_screen():
    try:
        image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_url, width=160)
        st.markdown(
            """<style>
               @media only screen and (max-width: 600px) {
                   .title { font-size: 2.5em; margin-top: 20px; text-align: center; }
               }
               @media only screen and (min-width: 601px) {
                   .title { font-size: 4em; font-weight: bold; margin-top: 90px; margin-left: -125px; text-align: left; }
               }
               </style>
            """, unsafe_allow_html=True
        )
        with col2:
            st.markdown('<div class="title">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

    st.markdown('<h3 style="font-size: 1.5em;">🦾 Welcome! Please enter your credentials to chat with your AI Buddy!</h3>', unsafe_allow_html=True)

    user_type_choice = st.radio("Select User Type", ["Student", "Teacher"])
    user_type_value = 2 if user_type_choice == "Teacher" else 3

    org_code = st.text_input("🏫 School Code", key="org_code")
    login_id = st.text_input("👤 Login ID", key="login_id")
    password = st.text_input("🔒 Password", type="password", key="password")

    query_params = st.experimental_get_query_params()
    E_params = query_params.get("E", [None])
    T_params = query_params.get("T", [None])

    E_value = E_params[0]
    T_value = T_params[0]

    if E_value is not None and T_value is not None:
        st.warning("Provide either ?E=xx for English OR ?T=xx for Non-English, not both.")
    elif E_value is not None and T_value is None:
        st.session_state.is_english_mode = True
        topic_id = E_value
    elif E_value is None and T_value is not None:
        st.session_state.is_english_mode = False
        topic_id = T_value
    else:
        st.warning("Please provide ?E=... or ?T=... in the URL.")
        return

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if topic_id is None:
            st.warning("Please ensure correct E or T parameter is provided.")
            return

        if not org_code or not login_id or not password:
            st.error("Please fill in all the fields.")
            return

        success, error_message = enhanced_login(
            org_code=org_code,
            login_id=login_id,
            password=password,
            topic_id=topic_id,
            is_english_mode=st.session_state.is_english_mode,
            user_type_value=user_type_value
        )

        if success:
            st.success("✅ Authentication successful!")
            st.rerun()
        else:
            st.error(f"🚫 Authentication failed: {error_message}")

# ----------------------------------------------------------------------------
# 6) MAIN SCREEN
# ----------------------------------------------------------------------------
def load_data_parallel():
    """
    Load baseline and concepts data in parallel using ThreadPoolExecutor
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(
            fetch_baseline_data,
            org_code=st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012'),
            subject_id=st.session_state.subject_id,
            user_id=st.session_state.user_id
        )
        
        concepts_future = executor.submit(
            fetch_all_concepts,
            org_code=st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012'),
            subject_id=st.session_state.subject_id,
            user_id=st.session_state.user_id
        )
        
        st.session_state.baseline_data = None
        st.session_state.all_concepts = []
        
        try:
            st.session_state.baseline_data = baseline_future.result()
        except Exception as e:
            st.error(f"Error fetching baseline data: {e}")
        
        try:
            st.session_state.all_concepts = concepts_future.result() or []
        except Exception as e:
            st.error(f"Error fetching all concepts: {e}")

def display_tabs_parallel():
    tab_containers = st.tabs(["💬 Chat", "🧠 Learning Path", "🔎 Gap Analyzer™", "📝 Baseline Testing"])
    
    chat_placeholder = tab_containers[0].empty()
    learning_path_placeholder = tab_containers[1].empty()
    all_concepts_placeholder = tab_containers[2].empty()
    baseline_testing_placeholder = tab_containers[3].empty()
    
    if not st.session_state.baseline_data or not st.session_state.all_concepts:
        with st.spinner("Loading data..."):
            load_data_parallel()
    
    with tab_containers[0]:
        chat_placeholder.subheader("Chat with your EeeBee AI buddy")
        add_initial_greeting()
        display_chat(st.session_state.auth_data['UserInfo'][0]['FullName'])
    
    with tab_containers[1]:
        learning_path_placeholder.subheader("Your Personalized Learning Path")
        display_learning_path_tab()
    
    with tab_containers[2]:
        all_concepts_placeholder.subheader("Gap Analyzer")
        display_all_concepts_tab()
    
    with tab_containers[3]:
        baseline_testing_placeholder.subheader("Baseline Testing Report")
        baseline_testing_report()

def display_learning_path_tab():
    weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
    concept_list = st.session_state.auth_data.get('ConceptList', [])

    if not weak_concepts:
        st.warning("No weak concepts found.")
    else:
        for idx, concept in enumerate(weak_concepts):
            concept_text = concept.get("ConceptText", f"Concept {idx+1}")
            concept_id = concept.get("ConceptID", f"id_{idx+1}")

            st.markdown(f"#### **Weak Concept {idx+1}:** {concept_text}")

            button_key = f"generate_lp_{concept_id}"
            if st.button("🧠 Generate Learning Path", key=button_key):
                if concept_id not in st.session_state.student_learning_paths:
                    with st.spinner(f"Generating learning path for {concept_text}..."):
                        learning_path = generate_learning_path(concept_text)
                        if learning_path:
                            st.session_state.student_learning_paths[concept_id] = {
                                "concept_text": concept_text,
                                "learning_path": learning_path
                            }
                            st.success(f"Learning path generated for {concept_text}!")
                        else:
                            st.error(f"Failed to generate learning path for {concept_text}.")
                else:
                    st.info(f"Learning path for {concept_text} is already generated.")

            if concept_id in st.session_state.student_learning_paths:
                lp_data = st.session_state.student_learning_paths[concept_id]
                display_learning_path_with_resources(
                    lp_data["concept_text"],
                    lp_data["learning_path"],
                    concept_list,
                    st.session_state.topic_id
                )

def add_initial_greeting():
    if len(st.session_state.chat_history) == 0 and st.session_state.auth_data:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data.get('TopicName', "Topic")

        concept_list = st.session_state.auth_data.get('ConceptList', [])
        weak_concepts = st.session_state.auth_data.get('WeakConceptList', [])

        concept_options = "\n\n**📚 Available Concepts:**\n"
        for concept in concept_list:
            concept_options += f"- {concept['ConceptText']}\n"

        weak_concepts_text = ""
        if weak_concepts:
            weak_concepts_text = "\n\n**🎯 Your Current Learning Gaps:**\n"
            for concept in weak_concepts:
                weak_concepts_text += f"- {concept['ConceptText']}\n"

        st.session_state.available_concepts = {
            concept['ConceptText']: concept['ConceptID'] for concept in concept_list
        }

        greeting_message = (
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy. "
            f"I'm here to help you with {topic_name}.\n\n"
            f"You can:\n"
            f"1. Ask me questions about any concept\n"
            f"2. Request learning resources (videos, notes, exercises)\n"
            f"3. Get help understanding specific topics\n"
            f"{concept_options}"
            f"{weak_concepts_text}\n\n"
            f"What would you like to discuss?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

def main_screen():
    user_info = st.session_state.auth_data['UserInfo'][0]
    user_name = user_info['FullName']
    topic_name = st.session_state.auth_data.get('TopicName')

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.is_teacher:
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
        with tabs[0]:
            st.subheader("Chat with your EeeBee AI buddy", anchor=None)
            add_initial_greeting()
            display_chat(user_name)
        with tabs[1]:
            st.subheader("Teacher Dashboard")
            teacher_dashboard()
    else:
        if st.session_state.is_english_mode:
            tab = st.tabs(["💬 Chat"])[0]
            with tab:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                display_chat(user_name)
        else:
            display_tabs_parallel()

def main():
    if st.session_state.is_authenticated:
        main_screen()
        st.stop()
    else:
        placeholder = st.empty()
        with placeholder.container():
            login_screen()

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Example snippet of a direct call to the DeepSeek ChatCompletion outside the app:
#
# if client:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": "Hello"},
#         ],
#         stream=False
#     )
#     print(response.choices[0].message.content)
# else:
#     print("DeepSeek client not available (missing API key).")
# -----------------------------------------------------------------------------
