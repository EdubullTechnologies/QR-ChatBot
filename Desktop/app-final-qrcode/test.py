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
import plotly.express as px

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"  # This makes the sidebar expanded by default but collapsible
)

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
    # Create the client with your base_url
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
API_STUDENT_INFO = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students"
API_STUDENT_CONCEPTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Concepts_OF_Students"

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
if "selected_student" not in st.session_state:
    st.session_state.selected_student = None
if "student_info" not in st.session_state:
    st.session_state.student_info = None

# Define the show_gap_message function globally
def show_gap_message():
    st.session_state.show_gap_message = True

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

    # Title
    story.append(Paragraph("Exam Questions", title_style))
    user_name_display = user_name if user_name else "Teacher"
    concept_text_display = concept_text if concept_text else "Selected Concept"
    story.append(Paragraph(f"For {user_name_display} - {concept_text_display}", subtitle_style))
    story.append(Spacer(1, 12))

    # Split questions by blank lines => sections
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

                        # convert latex to image
                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            question_items.append(ListItem(img))
                        last_index = match.end()

                # leftover text
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
        # NOTE: Use dot-notation to access the message content
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

        # Download Button
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

    # Title
    story.append(Paragraph("All Concepts", title_style))
    user_name_display = user_name if user_name else "User"
    story.append(Paragraph(f"User: {user_name_display}", subtitle_style))
    story.append(Spacer(1, 12))

    # Table Headers
    headers = ["Concept ID", "Concept Text", "Topic ID", "Status"]
    table_data = [headers]

    # Table Rows
    for concept in concepts:
        row = [
            str(concept.get("ConceptID", "")),
            concept.get("ConceptText", ""),
            str(concept.get("TopicID", "")),
            concept.get("ConceptStatus", "")
        ]
        table_data.append(row)

    # Create Table
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
        
        # Get subject_id from session state
        subject_id = st.session_state.subject_id
        if not subject_id:
            st.error("Subject ID not available")
            return

        # Fetch Baseline Data Early
        st.session_state.baseline_data = fetch_baseline_data(
            org_code=org_code,
            subject_id=subject_id,
            user_id=user_id
        )
    
    baseline_data = st.session_state.baseline_data
    if not baseline_data:
        st.warning("No baseline data available.")
        return

    # Unpack needed sections
    u_list = baseline_data.get("u_list", [])
    s_skills = baseline_data.get("s_skills", [])
    concept_wise_data = baseline_data.get("concept_wise_data", [])
    taxonomy_list = baseline_data.get("taxonomy_list", [])

    # ----------------------------------------------------------------
    # A) Student Summary
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # B) Skill-wise Performance (NO TABLE, HORIZONTAL BAR)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Skill-wise Performance")
    if s_skills:
        df_skills = pd.DataFrame(s_skills)

        skill_chart = alt.Chart(df_skills).mark_bar().encode(
            x=alt.X('RightAnswerPercent:Q', title='Correct %', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('SubjectSkillName:N', sort='-x'),
            tooltip=['SubjectSkillName:N', 'TotalQuestion:Q', 
                     'RightAnswerCount:Q', 'RightAnswerPercent:Q']
        ).properties(
            width=700,
            height=400,
            title="Skill-wise Correct Percentage"
        )
        st.altair_chart(skill_chart, use_container_width=True)
    else:
        st.info("No skill-wise data available.")

    # ----------------------------------------------------------------
    # C) Concept-wise Data
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Concept-wise Performance")
    if concept_wise_data:
        df_concepts = pd.DataFrame(concept_wise_data).copy()
        df_concepts["S.No."] = range(1, len(df_concepts) + 1)
        df_concepts["Concept Status"] = df_concepts["RightAnswerPercent"].apply(
            lambda x: "✅" if x == 100.0 else "❌"
        )
        df_concepts.rename(columns={"ConceptText": "Concept Name", 
                                    "BranchName": "Class"}, inplace=True)
        df_display = df_concepts[["S.No.", "Concept Name","Concept Status", "Class"]]
        st.dataframe(df_display, hide_index=True)
    else:
        st.info("No concept-wise data available.")

    # ----------------------------------------------------------------
    # D) Bloom's Taxonomy Performance
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Bloom's Taxonomy Performance")
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
    """Display all concepts with their status and remedial resources"""
    if not st.session_state.all_concepts:
        st.info("Loading concepts data...")
        return
    
    # Create a DataFrame for better display
    concept_data = []
    for concept in st.session_state.all_concepts:
        # Calculate status based on marks percent and attempts
        attended = concept.get('AttendedQuestion', 0)
        avg_marks = concept.get('AvgMarksPercent', 0)
        
        if attended == 0:
            status = "Not Attempted"
            status_color = "gray"
        elif avg_marks >= 70:
            status = "Strong"
            status_color = "green"
        else:
            status = "Weak"
            status_color = "red"
        
        concept_data.append({
            "Concept ID": concept.get('ConceptID', ''),
            "Concept": concept.get('ConceptText', 'Unknown'),
            "Questions Attempted": f"{concept.get('CorrectQuestion', 0)}/{concept.get('AttendedQuestion', 0)}",
            "Score": f"{concept.get('AvgMarksPercent', 0)}%",
            "Time Spent": format_time(concept.get('TotalTimeTaken_SS', 0)),
            "Status": status,
            "Status Color": status_color
        })
    
    # Create a styled table
    st.markdown("### 🔍 Concept Status Analysis")
    st.markdown("This analysis shows your performance across all concepts in this topic.")
    
    # Create a custom HTML table with better styling
    html_table = """
    <style>
    .concept-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    .concept-table th {
        background-color: #f0f2f6;
        padding: 10px;
        text-align: left;
        border-bottom: 2px solid #ddd;
    }
    .concept-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .status-strong {
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 4px;
    }
    .status-weak {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 4px;
    }
    .status-not-attempted {
        background-color: #e2e3e5;
        color: #383d41;
        font-style: italic;
        padding: 5px 10px;
        border-radius: 4px;
    }
    </style>
    <table class="concept-table">
        <thead>
            <tr>
                <th>Concept</th>
                <th>Questions</th>
                <th>Score</th>
                <th>Time Spent</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for concept in concept_data:
        status_class = ""
        if concept["Status"] == "Strong":
            status_class = "status-strong"
        elif concept["Status"] == "Weak":
            status_class = "status-weak"
        else:
            status_class = "status-not-attempted"
        
        html_table += f"""
        <tr>
            <td>{concept["Concept"]}</td>
            <td>{concept["Questions Attempted"]}</td>
            <td>{concept["Score"]}</td>
            <td>{concept["Time Spent"]}</td>
            <td><span class="{status_class}">{concept["Status"]}</span></td>
        </tr>
        """
    
    html_table += """
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Add a section for remedial resources
    st.markdown("### 📚 Remedial Resources")
    st.markdown("Select a concept to view remedial resources:")
    
    # Create a selectbox with all concepts
    concept_names = [c["Concept"] for c in concept_data]
    selected_concept = st.selectbox(
        "Choose a concept:",
        options=concept_names,
        key="remedial_concept_selector"
    )
    
    if selected_concept:
        # Find the concept in the original data
        selected_concept_obj = next(
            (c for c in st.session_state.all_concepts if c.get('ConceptText') == selected_concept),
            None
        )
        
        if selected_concept_obj:
            # Display concept status
            attended = selected_concept_obj.get('AttendedQuestion', 0)
            avg_marks = selected_concept_obj.get('AvgMarksPercent', 0)
            
            if attended == 0:
                status = "Not Attempted"
                status_color = "gray"
            elif avg_marks >= 70:
                status = "Strong"
                status_color = "green"
            else:
                status = "Weak"
                status_color = "red"
            
            st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            
            # Only show remedial resources for weak concepts
            if status == "Weak":
                # Get remedial resources
                with st.spinner("Fetching remedial resources..."):
                    resources = get_resources_for_concept(
                        selected_concept,
                        st.session_state.auth_data.get('ConceptList', []),
                        st.session_state.topic_id
                    )
                
                if resources:
                    # Videos
                    if resources.get("Video_List"):
                        st.markdown("#### 🎥 Video Lectures")
                        for video in resources["Video_List"]:
                            video_url = f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}"
                            title = video.get('LectureTitle', 'Video Lecture')
                            st.markdown(f"- [{title}]({video_url})")
                    
                    # Notes
                    if resources.get("Notes_List"):
                        st.markdown("#### 📄 Study Notes")
                        for note in resources["Notes_List"]:
                            note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                            title = note.get('NotesTitle', 'Study Notes')
                            st.markdown(f"- [{title}]({note_url})")
                    
                    # Exercises
                    if resources.get("Exercise_List"):
                        st.markdown("#### 📝 Practice Exercises")
                        for exercise in resources["Exercise_List"]:
                            exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                            title = exercise.get('ExerciseTitle', 'Practice Exercise')
                            st.markdown(f"- [{title}]({exercise_url})")
                    
                    if not (resources.get("Video_List") or resources.get("Notes_List") or resources.get("Exercise_List")):
                        st.info("No remedial resources available for this concept.")
                else:
                    st.info("No remedial resources available for this concept.")
            elif status == "Not Attempted":
                st.info("Please attempt questions on this concept to see remedial resources.")
            else:  # Strong
                st.success("Great job! You have a strong understanding of this concept.")

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
    selected_batch_name = st.selectbox("Select a Batch:", list(batch_options.keys()), key="batch_selector")
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

        # Main bar chart
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

    if selected_batch_id:
        # Fetch student info when batch is selected
        user_info = st.session_state.auth_data.get('UserInfo', [{}])[0]
        org_code = user_info.get('OrgCode', '012')
        
        student_info = fetch_student_info(
            batch_id=selected_batch_id,
            topic_id=st.session_state.topic_id,
            org_code=org_code
        )
        
        if student_info:
            st.session_state.student_info = student_info
            
            # Display concept-wise analytics
            st.subheader("📊 Concept-wise Analysis")
            concepts_df = pd.DataFrame(student_info["Concepts"])
            if not concepts_df.empty:
                fig = px.bar(concepts_df, 
                    x="ConceptText",
                    y=["AttendedStudentCount", "ClearedStudentCount"],
                    title="Concept Performance Overview",
                    barmode="group"
                )
                st.plotly_chart(fig)
            
            # Student selection
            st.subheader("👥 Student Selection")
            students = student_info["Students"]
            student_options = {f"{s['FullName']} (ID: {s['UserID']})": s for s in students}
            
            selected_student_name = st.selectbox(
                "Select a student to view detailed analysis:",
                options=list(student_options.keys())
            )
            
            if selected_student_name:
                selected_student = student_options[selected_student_name]
                st.session_state.selected_student = selected_student
                
                # Display student analytics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Concepts", selected_student["TotalConceptCount"])
                col2.metric("Weak Concepts", selected_student["WeakConceptCount"])
                col3.metric("Cleared Concepts", selected_student["ClearedConceptCount"])
                
                # Calculate and display progress
                progress = (selected_student["ClearedConceptCount"] / 
                          selected_student["TotalConceptCount"]) * 100 if selected_student["TotalConceptCount"] > 0 else 0
                
                st.progress(progress/100)
                st.markdown(f"**Overall Progress:** {progress:.1f}%")

        # Bloom's Level
        st.subheader("📝 Question Generation")
        bloom_level = st.radio(
            "Select Bloom's Taxonomy Level for the Questions",
            [
                "L1 (Remember)",
                "L2 (Understand)",
                "L3 (Apply)",
                "L4 (Analyze)",
                "L5 (Evaluate)"
            ],
            index=3,
            key="bloom_taxonomy_selector"
        )
        bloom_short = bloom_level.split()[0]  # e.g. "L4"

        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.radio("Select a Concept to Generate Exam Questions:", list(concept_list.keys()), key="concept_selector_teacher")

        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions", key="generate_exam_btn"):
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
                    f"- Maintain a professional, informative tone, and ensure all advice aligns with the NCERT curriculum.\n"
                    f"- Keep all mathematical expressions within LaTeX delimiters ($...$ or $$...$$).\n"
                    f"- Emphasize to the teacher the importance of fostering critical thinking.\n"
                    f"- If the teacher requests sample questions, provide them in a progressive manner, ensuring they prompt the student to reason through each step.\n\n"
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
                        # Use dot-notation to access the content
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
                
                

# ------------------- 2J) CHAT FUNCTIONS -------------------
def add_initial_greeting():
    if len(st.session_state.chat_history) == 0 and st.session_state.auth_data:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data.get('TopicName', "Topic")

        if st.session_state.is_teacher:
            batches = st.session_state.auth_data.get("BatchList", [])
            batch_list = "\n".join([
                f"- {b['BatchName']} ({b.get('StudentCount', 0)} students)"
                for b in batches
            ])
            
            greeting_message = (
                f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy for {topic_name}.\n\n"
                f"Your classes:\n{batch_list}\n\n"
                f"Quick Guide:\n"
                f"1. Simply type a class name (e.g., 'Class-8 DB') to see class analysis and student list\n"
                f"2. Then type any student's name to analyze their individual performance\n"
                f"3. Or type 'show classes' anytime to see your class list again\n\n"
                f"I can help you with:\n"
                f"- Creating personalized lesson plans based on class performance\n"
                f"- Suggesting targeted teaching strategies for specific concepts\n"
                f"- Identifying which concepts need more attention in your class\n"
                f"- Analyzing individual student progress and learning gaps\n\n"
                f"Which class would you like to analyze first?"
            )
            st.session_state.chat_history.append(("assistant", greeting_message))
        else:
            # Existing student mode code remains unchanged
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

def handle_user_input(user_input):
    """Process user input and generate a response"""
    if not user_input:
        return
    
    # Check if this is a teacher command
    if st.session_state.is_teacher:
        command_response = handle_teacher_commands(user_input)
        if command_response:
            # Display the command response directly
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(command_response)
            
            # Add to chat history
            st.session_state.chat_history.append(("assistant", command_response))
            return
    
    # If not a command, get GPT response
    get_gpt_response(user_input)

def format_time(seconds):
    """Format seconds into a readable time string"""
    if seconds < 60:
        return f"{seconds} sec"
    elif seconds < 3600:
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes} min {sec} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hr {minutes} min"

def format_concept_details(concept):
    """Format concept details with performance metrics"""
    if concept['AttendedQuestion'] == 0:
        return (
            f"- {concept['ConceptText']}\n"
            f"  📝 Not attempted any questions yet"
        )
    
    return (
        f"- {concept['ConceptText']}\n"
        f"  📝 Questions: {concept['CorrectQuestion']}/{concept['AttendedQuestion']} correct "
        f"({concept['AvgMarksPercent']}%)\n"
        f"  ⏱️ Average time per question: {format_time(concept['AvgTimeTaken_SS'])}\n"
        f"  ⌛ Total time spent: {format_time(concept['TotalTimeTaken_SS'])}"
    )

def fetch_student_info(batch_id, topic_id, org_code):
    """Fetch student information for a specific batch - exact implementation from test.py"""
    params = {
        "BatchID": batch_id,
        "TopicID": topic_id,
        "OrgCode": org_code
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        logging.info(f"Fetching student info with params: {params}")
        response = requests.post(API_STUDENT_INFO, json=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Student info API response status: {data.get('Status', 'No status')}")
        return data
    except Exception as e:
        logging.error(f"Error fetching student info: {e}")
        st.error(f"Error fetching student info: {e}")
        return None

def fetch_student_concepts(user_id, topic_id, org_code):
    """Fetch detailed concept information for a specific student - exact implementation from test.py"""
    params = {
        "UserID": user_id,
        "TopicID": topic_id,
        "OrgCode": org_code
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        logging.info(f"Fetching student concepts with params: {params}")
        response = requests.post(API_STUDENT_CONCEPTS, json=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        logging.info(f"Student concepts API response status: {data.get('Status', 'No status')}")
        return data
    except Exception as e:
        logging.error(f"Error fetching student concepts: {e}")
        st.error(f"Error fetching student concepts: {e}")
        return None

def handle_teacher_commands(user_input):
    """Handle teacher-specific chat commands - exact implementation from test.py"""
    input_lower = user_input.lower().strip()
    
    # Show classes
    if any(cmd in input_lower for cmd in ["show classes", "show batches", "list classes", "list batches"]):
        batches = st.session_state.auth_data.get("BatchList", [])
        batch_list = "\n".join([
            f"- {b['BatchName']} ({b.get('StudentCount', 0)} students)"
            for b in batches
        ])
        return f"Your classes:\n{batch_list}\n\nJust type the class name you want to analyze (e.g., '10A')"
    
    # Select class (checking if input matches any batch name)
    batches = st.session_state.auth_data.get("BatchList", [])
    selected_batch = next((b for b in batches if b['BatchName'].lower() == input_lower), None)
    if selected_batch:
        # Fetch student info
        student_info = fetch_student_info(
            selected_batch["BatchID"],
            st.session_state.topic_id,
            st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012')
        )
        
        if student_info and student_info.get("Students"):
            st.session_state.current_batch_students = student_info["Students"]
            st.session_state.current_batch_concepts = student_info.get("Concepts", [])
            
            # Calculate class statistics
            total_students = len(student_info["Students"])
            concepts = student_info.get("Concepts", [])
            
            # Prepare concept statistics
            concept_stats = []
            for concept in concepts:
                cleared_percent = (concept['ClearedStudentCount'] / concept['AttendedStudentCount'] * 100) if concept['AttendedStudentCount'] > 0 else 0
                concept_stats.append(
                    f"- {concept['ConceptText']}: {concept['ClearedStudentCount']}/{concept['AttendedStudentCount']} "
                    f"students cleared ({cleared_percent:.1f}%)"
                )
            
            concept_overview = "\n".join(concept_stats)
            
            # Automatically show students after selecting a class
            students = student_info["Students"]
            # Group students by progress
            all_cleared = []
            partial_progress = []
            no_progress = []
            
            for student in students:
                if student['ClearedConceptCount'] == student['TotalConceptCount']:
                    all_cleared.append(student['FullName'])
                elif student['ClearedConceptCount'] > 0:
                    partial_progress.append(
                        f"- {student['FullName']} ({student['ClearedConceptCount']}/{student['TotalConceptCount']} concepts cleared)"
                    )
                else:
                    no_progress.append(student['FullName'])
            
            student_list = "Students in this class:\n\n"
            
            if all_cleared:
                student_list += "✅ Completed all concepts:\n- " + "\n- ".join(all_cleared) + "\n\n"
            if partial_progress:
                student_list += "🔄 In progress:\n" + "\n".join(partial_progress) + "\n\n"
            if no_progress:
                student_list += "⚠️ No concepts cleared:\n- " + "\n- ".join(no_progress) + "\n\n"
                
            student_list += "Just type a student's name to analyze their progress"
            
            # Combine class overview and student list
            return (
                f"Looking at class {selected_batch['BatchName']}:\n\n"
                f"Class Overview:\n"
                f"- Total Students: {total_students}\n"
                f"- Concepts Coverage:\n{concept_overview}\n\n"
                f"{student_list}"
            )
        else:
            return "I couldn't get the student information for this class. Please try again."
    
    # Show students in current class (still keep this command for convenience)
    if "show students" in input_lower or "list students" in input_lower:
        if hasattr(st.session_state, 'current_batch_students'):
            students = st.session_state.current_batch_students
            # Group students by progress
            all_cleared = []
            partial_progress = []
            no_progress = []
            
            for student in students:
                if student['ClearedConceptCount'] == student['TotalConceptCount']:
                    all_cleared.append(student['FullName'])
                elif student['ClearedConceptCount'] > 0:
                    partial_progress.append(
                        f"- {student['FullName']} ({student['ClearedConceptCount']}/{student['TotalConceptCount']} concepts cleared)"
                    )
                else:
                    no_progress.append(student['FullName'])
            
            response = "Students in this class:\n\n"
            
            if all_cleared:
                response += "✅ Completed all concepts:\n- " + "\n- ".join(all_cleared) + "\n\n"
            if partial_progress:
                response += "🔄 In progress:\n" + "\n".join(partial_progress) + "\n\n"
            if no_progress:
                response += "⚠️ No concepts cleared:\n- " + "\n- ".join(no_progress) + "\n\n"
                
            response += "Just type a student's name to analyze their progress"
            return response
        else:
            return "Please select a class first by typing the class name (e.g., '10A')"
    
    # Select student (checking if input matches any student name)
    if hasattr(st.session_state, 'current_batch_students'):
        selected_student = next(
            (s for s in st.session_state.current_batch_students 
             if s['FullName'].lower() == input_lower), None)
        
        if selected_student:
            st.session_state.selected_student = selected_student
            
            # Fetch detailed concept information
            student_concepts = fetch_student_concepts(
                user_id=selected_student['UserID'],
                topic_id=st.session_state.topic_id,
                org_code=st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012')
            )
            
            # Calculate progress percentage
            progress = (selected_student['ClearedConceptCount'] / selected_student['TotalConceptCount'] * 100) if selected_student['TotalConceptCount'] > 0 else 0
            
            # Format concept details with performance metrics
            weak_concepts_details = []
            cleared_concepts_details = []
            
            if student_concepts:
                weak_concepts_details = [
                    format_concept_details(concept)
                    for concept in student_concepts.get('WeakConcepts_List', [])
                ]
                cleared_concepts_details = [
                    format_concept_details(concept)
                    for concept in student_concepts.get('ClearedConcepts_List', [])
                ]
            
            # Calculate overall statistics
            total_questions = 0
            total_correct = 0
            total_time = 0
            attempted_concepts = 0
            
            for concept in student_concepts.get('WeakConcepts_List', []) + student_concepts.get('ClearedConcepts_List', []):
                if concept['AttendedQuestion'] > 0:
                    attempted_concepts += 1
                    total_questions += concept['AttendedQuestion']
                    total_correct += concept['CorrectQuestion']
                    total_time += concept['TotalTimeTaken_SS']
            
            # Build response message
            response = (
                f"Looking at {selected_student['FullName']}'s progress:\n\n"
                f"📊 Overall Performance:\n"
                f"- Progress: {progress:.1f}%\n"
            )
            
            if attempted_concepts > 0:
                avg_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
                response += (
                    f"- Overall Accuracy: {avg_accuracy:.1f}%\n"
                    f"- Total Questions Attempted: {total_questions}\n"
                    f"- Total Time Spent: {format_time(total_time)}\n"
                )
            
            response += "\n🔍 Concepts Needing Attention:\n"
            if weak_concepts_details:
                response += "\n".join(weak_concepts_details) + "\n"
            else:
                response += "✅ No weak concepts identified\n"
            
            response += "\n✨ Mastered Concepts:\n"
            if cleared_concepts_details:
                response += "\n".join(cleared_concepts_details) + "\n"
            else:
                response += "⚠️ No concepts cleared yet\n"
            
            response += (
                f"\nYou can ask me about:\n"
                f"- Specific teaching strategies for concepts they're struggling with\n"
                f"- How to improve their accuracy and speed\n"
                f"- Ways to help them progress in specific concepts\n"
                f"- Detailed analysis of their performance in any concept"
            )
            
            return response
    
    return None

def get_system_prompt():
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    if st.session_state.is_teacher:
        # Teacher mode prompt remains unchanged
        batches = st.session_state.auth_data.get("BatchList", [])
        batch_list = "\n".join([f"- {b['BatchName']} (ID: {b['BatchID']})" for b in batches])
        
        return f"""
You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {topic_name}.

Teacher Mode Instructions:
- The user is a teacher instructing {branch_name} students under the NCERT curriculum.
- Available batches:\n{batch_list}
- When the user types a class name (e.g., "Class-8 DB"), show class analysis and student list.
- When the user types a student name, show detailed analysis for that student.
- Keep all mathematical expressions within LaTeX delimiters.
- Focus on helping teachers analyze student performance and design effective strategies.

Commands to recognize:
- Simply typing a class name (e.g., "Class-8 DB") - Show class analysis and student list
- Simply typing a student name (e.g., "John Smith") - Show detailed student analysis
- "show classes" or "list classes" - Display available classes
- "generate lesson plan" - Create a customized lesson plan based on class performance
- "suggest strategies" - Provide instructional strategies to improve student outcomes
"""
    else:
        # Enhanced student mode prompt with improved test generation
        weak_concepts = [concept['ConceptText'] for concept in st.session_state.student_weak_concepts]
        weak_concepts_text = ", ".join(weak_concepts) if weak_concepts else "none"

        return f"""
You are a highly knowledgeable educational assistant named EeeBee, developed by iEdubull and specialized in {topic_name}.

CRITICAL INSTRUCTION: You must NEVER directly answer a student's question or solve a problem for them. Instead, use the Socratic method to guide them toward discovering the answer themselves.

Student Mode Instructions:
- The student is in {branch_name} and follows the NCERT curriculum.
- The student's weak concepts are: {weak_concepts_text}
- Focus exclusively on {topic_name} in your discussions.

Socratic Teaching Method (MANDATORY):
1. When a student asks a direct question or wants a solution:
   - NEVER provide the direct answer or solution
   - Instead, respond with 2-3 guiding questions that help them think through the problem
   - Ask them what they already know about the topic
   - Suggest they try a specific approach and explain their reasoning
   - Break down complex problems into smaller, manageable steps

2. When a student attempts to answer:
   - Acknowledge their effort positively
   - If incorrect, don't simply state they're wrong
   - Guide them to discover their mistake through targeted questions
   - If correct, ask them to explain their reasoning to reinforce learning

3. For conceptual questions:
   - Ask them to relate the concept to real-world examples
   - Guide them to make connections with previously learned material
   - Encourage them to formulate their own examples

4. For problem-solving:
   - Ask them to identify the given information and what they're trying to find
   - Guide them to select appropriate formulas or methods
   - Have them estimate a reasonable answer before calculating
   - Encourage them to check their work and verify the solution

Test Generation and Learning Gap Analysis:
- When a student requests a test, create a comprehensive 10-question MCQ test covering key concepts in {topic_name}
- Present all 10 questions at once, clearly numbered from 1-10
- Each question should have 4 options (A, B, C, D) with only one correct answer
- Include a mix of:
  - Current grade-level concepts from NCERT {branch_name} curriculum
  - Prerequisite concepts from previous grades that are foundational to current topics
- After the student submits all answers, provide:
  1. A score summary (X/10 correct)
  2. A detailed analysis for each question showing:
     - The correct answer
     - The student's answer
     - A brief explanation of the concept tested
  3. A comprehensive learning gap analysis that:
     - Identifies current grade-level gaps based on NCERT curriculum
     - Pinpoints specific previous grade-level gaps, explicitly stating:
       * Which concept is weak
       * Which previous class/grade it belongs to (e.g., "This is a Class 7 concept on...")
       * How this gap impacts current learning
     - Recommends targeted remedial activities for each identified gap

Formatting:
- All mathematical expressions must be enclosed in LaTeX delimiters ($...$ or $$...$$)
- Use bullet points and numbered lists for clarity
- Bold important concepts or key points

Remember: Your goal is to develop the student's critical thinking and problem-solving skills, not to provide answers. Success is measured by how well you guide them to discover solutions independently.
"""

def display_chat(user_name):
    """Display the chat interface with message history"""
    # Create a container for the chat history
    chat_container = st.container()
    
    # Display all messages from history
    with chat_container:
        for role, content in st.session_state.chat_history:
            with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                st.markdown(content)
    
    # Get user input
    if prompt := st.chat_input(f"Ask me anything about {st.session_state.auth_data.get('TopicName', 'your topic')}..."):
        # Add user message to chat history
        st.session_state.chat_history.append(("user", prompt))
        
        # Force a rerun to display the user message before processing
        st.rerun()

def process_pending_messages():
    """Process any pending user messages that need responses"""
    # Check if the last message is from the user and needs a response
    if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
        user_input = st.session_state.chat_history[-1][1]
        
        # Check if this is a teacher command
        if st.session_state.is_teacher:
            command_response = handle_teacher_commands(user_input)
            if command_response:
                # Add command response to chat history
                st.session_state.chat_history.append(("assistant", command_response))
                return
        
        # If not a command, get GPT response
        get_gpt_response(user_input)

def get_gpt_response(user_input):
    if not client:
        st.error("DeepSeek client is not initialized. Check your API key.")
        return
    
    system_prompt = get_system_prompt()
    conversation_history_formatted = [{"role": "system", "content": system_prompt}]
    
    # Format the conversation history for the API
    for role, content in st.session_state.chat_history:
        conversation_history_formatted.append({"role": role, "content": content})
    
    try:
        # Create a chat message container for the assistant
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Create a streaming response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history_formatted,
                max_tokens=2000,
                stream=True
            )
            
            # Process the streaming response
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Update the placeholder with the current response
                    message_placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            message_placeholder.markdown(full_response)
        
        # Add the complete response to chat history
        st.session_state.chat_history.append(("assistant", full_response))
        
    except Exception as e:
        error_message = f"I'm sorry, I encountered an error: {str(e)}"
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(error_message)
        st.session_state.chat_history.append(("assistant", error_message))

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

    # Read query parameters from URL
    query_params = st.experimental_get_query_params()
    E_params = query_params.get("E", [None])
    T_params = query_params.get("T", [None])
    E_value = E_params[0]
    T_value = T_params[0]

    # Check for conflicting parameters or missing ones
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

    # Conditionally display the user type selection:
    if st.session_state.is_english_mode:
        # For English mode, force Student login and hide the radio button.
        st.markdown("**User Type:** Student")
        user_type_value = 3  # 3 for Student
    else:
        user_type_choice = st.radio("Select User Type", ["Student", "Teacher"], key="user_type_selector")
        user_type_value = 2 if user_type_choice == "Teacher" else 3

    org_code = st.text_input("🏫 School Code", key="org_code")
    login_id = st.text_input("👤 Login ID", key="login_id")
    password = st.text_input("🔒 Password", type="password", key="password")

    if st.button("🚀 Login and Start Chatting!", key="login_button") and not st.session_state.get("is_authenticated", False):
        if topic_id is None:
            st.warning("Please ensure a correct E or T parameter is provided.")
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
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    tab_selection = st.sidebar.radio(
        "Choose a section:",
        ["💬 Chat", "🧠 Learning Path", "🔎 Gap Analyzer™", "📝 Baseline Testing"]
    )
    
    # Add logout button to sidebar
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.clear()
        st.rerun()
    
    # Main content area
    if tab_selection == "💬 Chat":
        st.subheader("Chat with your EeeBee AI buddy")
        add_initial_greeting()
        display_chat(st.session_state.auth_data['UserInfo'][0]['FullName'])
    
    elif tab_selection == "🧠 Learning Path":
        st.subheader("Your Personalized Learning Path")
        display_learning_path_tab()
    
    elif tab_selection == "🔎 Gap Analyzer™":
        st.subheader("Gap Analyzer")
        display_all_concepts_tab()
    
    elif tab_selection == "📝 Baseline Testing":
        st.subheader("Baseline Testing Report")
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

            if concept_id in st.session_state.student_learning_paths:
                lp_data = st.session_state.student_learning_paths[concept_id]
                display_learning_path_with_resources(
                    lp_data["concept_text"],
                    lp_data["learning_path"],
                    concept_list,
                    st.session_state.topic_id
                )

def main_screen():
    user_info = st.session_state.auth_data['UserInfo'][0]
    user_name = user_info['FullName']
    topic_name = st.session_state.auth_data.get('TopicName')

    # Display header
    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.is_teacher:
        # Create a sidebar for navigation in teacher mode
        st.sidebar.title("Navigation")
        tab_selection = st.sidebar.radio(
            "Choose a section:",
            ["💬 Chat", "📊 Teacher Dashboard"]
        )
        
        # Add logout button to sidebar
        if st.sidebar.button("Logout", key="logout_button_teacher"):
            st.session_state.clear()
            st.rerun()
        
        if tab_selection == "💬 Chat":
            st.subheader("Chat with your EeeBee AI buddy", anchor=None)
            add_initial_greeting()
            display_chat(user_name)
        else:
            st.subheader("Teacher Dashboard")
            teacher_dashboard()
    else:
        if st.session_state.is_english_mode:
            st.subheader("Chat with your EeeBee AI buddy", anchor=None)
            add_initial_greeting()
            display_chat(user_name)
        else:
            # Load data if not already loaded
            if not st.session_state.baseline_data or not st.session_state.all_concepts:
                with st.spinner("EeeBee is waking up..."):
                    load_data_parallel()
            
            # Display tabs in sidebar
            display_tabs_parallel()

def main():
    if st.session_state.is_authenticated:
        main_screen()
        # Process any pending messages after rendering the UI
        process_pending_messages()
        st.stop()
    else:
        placeholder = st.empty()
        with placeholder.container():
            login_screen()

if __name__ == "__main__":
    main()
