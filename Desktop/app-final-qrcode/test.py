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
import time  # Added for retry logic

# ----------------------------------------------------------------------------------
# If you are using a custom LLM or a DeepSeek-like client, import it; 
# otherwise, for official OpenAI, do something like: import openai
# and set openai.api_key = ...
# ----------------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    st.warning("Using a placeholder client (please install or import your actual LLM client).")
    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            pass
        class chat:
            class completions:
                @staticmethod
                def create(model, messages, stream=False, max_tokens=800, **kwargs):
                    # This is a mocked response
                    return {"choices": [{"message": {"content": "Mocked LLM response."}}]}

# ----------------------------------------------------------------------------------
# 1) BASIC SETUP
# ----------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

logging.basicConfig(level=logging.INFO)

# Load (or set) your LLM API key and initialize client 
# (If using official openai, do openai.api_key = "YOUR_API_KEY")
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    OPENAI_API_KEY = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")
else:
    # Fall back to a dummy or your custom setup
    client = OpenAI(api_key="dummy_key")

# --- API Endpoints (Adjust as needed) ---
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
API_TEACHER_WEAK_CONCEPTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts"
API_BASELINE_REPORT = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Baseline_Report_Single_Student"
API_ALL_CONCEPTS_URL = "https://webapi.edubull.com/api/eProfessor/eProf_Org_ConceptList_Single_Student"
# >>> NEW ENDPOINT <<<
API_TEACHER_TOPICWISE_WEAK_STUDENTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students"

# --- Initialize session states ---
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
    st.session_state.subject_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "all_concepts" not in st.session_state:
    st.session_state.all_concepts = []
if "remedial_info" not in st.session_state:
    st.session_state.remedial_info = None
if 'show_gap_message' not in st.session_state:
    st.session_state.show_gap_message = False
# NEW: Storing teacher topicwise data (concept+students)
if "teacher_topicwise_data" not in st.session_state:
    st.session_state.teacher_topicwise_data = {}
# For storing a selected student in teacher mode
if "selected_student_for_chat" not in st.session_state:
    st.session_state.selected_student_for_chat = None

# A helper to show gap message
def show_gap_message():
    st.session_state.show_gap_message = True

# --- Streamlit page config ---
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
# ----------------------------------------------------------------------------------
# 2A) LATEX TO IMAGE
def latex_to_image(latex_code, dpi=300):
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

# 2B) FETCHING REMEDIAL RESOURCES
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
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

# 2C) PDF GENERATION FOR EXAM QUESTIONS
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

# 2D) LEARNING PATH GENERATION
def generate_learning_path(concept_text):
    if not client:
        st.error("DeepSeek client is not initialized. Check your API key.")
        return None

    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    prompt = f"""You are a highly experienced educational AI assistant specializing in the NCERT curriculum.
A student in {branch_name} is struggling with the weak concept: '{concept_text}'.
Create a structured, step-by-step learning path that ensures maximum understanding and engagement.
Use proper LaTeX for math. Provide an intro, step-by-step learning, engagement activities,
real-world applications, and practice problems. Target NCERT level for {branch_name} students."""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": prompt}],
            stream=False,
            max_tokens=1500
        )
        gpt_response = response["choices"][0]["message"]["content"].strip()
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
        # Assume the first line can be a heading
        story.append(Paragraph(lines[0], styles['Heading3']))
        story.append(Spacer(1, 6))

        for line in lines[1:]:
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
                last_index = 0
                text_len = len(line)
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

# 2E) FETCH ALL CONCEPTS
def fetch_all_concepts(org_code, subject_id, user_id):
    payload = {
        "OrgCode": org_code,
        "SubjectID": subject_id,
        "UserID": user_id
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    try:
        response = requests.post(API_ALL_CONCEPTS_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching all concepts: {e}")
        return None

# 2F) PDF GENERATION FOR ALL CONCEPTS
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

# 2G) FETCH REMEDIAL RESOURCES
def fetch_remedial_resources(topic_id, concept_id):
    remedial_api_url = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
    payload = {
        "TopicID": topic_id,
        "ConceptID": concept_id
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    try:
        response = requests.post(remedial_api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching remedial resources: {e}")
        return None

# 2H) FORMAT REMEDIAL RESOURCES
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

# ----------------------------------------------------------------------------------
# 3) BASELINE TESTING REPORT
# ----------------------------------------------------------------------------------
def fetch_baseline_data(org_code, subject_id, user_id):
    payload = {
        "UserID": user_id,
        "SubjectID": subject_id,
        "OrgCode": org_code
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    try:
        with st.spinner("EeeBee is fetching baseline data..."):
            response = requests.post(API_BASELINE_REPORT, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error fetching baseline data: {e}")
        return None

def baseline_testing_report():
    if not st.session_state.baseline_data:
        # fetch if not already fetched
        user_info = st.session_state.auth_data.get('UserInfo', [{}])[0]
        user_id = user_info.get('UserID')
        org_code = user_info.get('OrgCode', '012')
        subject_id = st.session_state.subject_id
        if not subject_id:
            st.error("Subject ID not available for baseline.")
            return
        st.session_state.baseline_data = fetch_baseline_data(org_code, subject_id, user_id)

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
        df_concepts = pd.DataFrame(concept_wise_data)
        df_concepts["S.No."] = range(1, len(df_concepts)+1)
        df_concepts["Concept Status"] = df_concepts["RightAnswerPercent"].apply(
            lambda x: "✅" if x == 100.0 else "❌"
        )
        df_concepts.rename(columns={"ConceptText": "Concept Name", "BranchName": "Class"}, inplace=True)
        df_display = df_concepts[["S.No.", "Concept Name", "Concept Status", "Class"]]
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

# 2I) ALL CONCEPTS TAB
def display_all_concepts_tab():
    st.markdown("### 📌 EeeBee is generating remedials according to your current gaps.")

    all_concepts = st.session_state.all_concepts
    if not all_concepts:
        st.warning("No concepts found.")
        return

    col_widths = [3, 1, 1.5, 1.5]
    headers = ["Concept Text", "Status", "Remedial", "Previous Learning GAP"]
    header_cols = st.columns(col_widths)
    for i, header in enumerate(headers):
        header_cols[i].markdown(f"**{header}**")

    @st.cache_data(show_spinner=False)
    def cached_fetch_remedial_resources(topic_id, concept_id):
        return fetch_remedial_resources(topic_id, concept_id)

    def fetch_resources(concept):
        if concept['ConceptStatus'] in ["Weak", "Not-Attended"]:
            resources = cached_fetch_remedial_resources(concept['TopicID'], concept['ConceptID'])
            return format_remedial_resources(resources)
        else:
            return "-"

    concepts_to_display = []
    for c in all_concepts:
        concepts_to_display.append({
            "ConceptID": c['ConceptID'],
            "ConceptText": c['ConceptText'],
            "TopicID": c['TopicID'],
            "ConceptStatus": c['ConceptStatus']
        })

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(fetch_resources, c): c for c in concepts_to_display}
        for future in as_completed(future_map):
            concept = future_map[future]
            remedial_str = future.result()

            row = st.columns(col_widths)
            concept_text = concept["ConceptText"]
            status = concept["ConceptStatus"]
            status_color = 'red' if status == 'Weak' else ('green' if status == 'Cleared' else 'orange')
            status_icon = '🔴' if status == 'Weak' else ('🟢' if status == 'Cleared' else '🟠')
            status_html = f"<span style='color:{status_color};'>{status_icon} {status}</span>"

            row[0].markdown(concept_text)
            row[1].markdown(status_html, unsafe_allow_html=True)

            with row[2]:
                if remedial_str != "-":
                    with st.expander("🧠 Remedial Resources"):
                        st.markdown(remedial_str)
                else:
                    st.markdown("-")

            with row[3]:
                if status in ["Weak", "Not-Attended"]:
                    st.button("Previous GAP", key=f"gap_{concept['ConceptID']}", on_click=show_gap_message)
                else:
                    st.markdown("-")

# ----------------------------------------------------------------------------------
# 4) TEACHER DASHBOARD WITH NEW (Concept+Student) API
# ----------------------------------------------------------------------------------
def fetch_teacher_topicwise_data(batch_id, topic_id, org_code):
    """
    Calls the new endpoint:
    /api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students
    """
    payload = {
        "BatchID": batch_id,
        "TopicID": topic_id,
        "OrgCode": org_code
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    try:
        response = requests.post(API_TEACHER_TOPICWISE_WEAK_STUDENTS, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching teacher topicwise data: {e}")
        return None

def display_concept_stats():
    """
    Shows the teacher's 'Weak Concepts' overview from the `Concepts` part of the new API.
    """
    data = st.session_state.teacher_topicwise_data
    if not data:
        st.info("No teacher data loaded. Please select a batch to fetch data.")
        return

    concepts = data.get("Concepts", [])
    if not concepts:
        st.warning("No concept data available.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(concepts)
    # Basic bar chart
    if not df.empty:
        st.markdown("### Weak Concepts Overview")
        df_long = df.melt(id_vars="ConceptText", 
                          value_vars=["AttendedStudentCount", "ClearedStudentCount"],
                          var_name="Category", 
                          value_name="Count")
        df_long["Category"] = df_long["Category"].replace({
            "AttendedStudentCount": "Attended",
            "ClearedStudentCount": "Cleared"
        })
        chart = alt.Chart(df_long).mark_bar().encode(
            x="ConceptText:N",
            y="Count:Q",
            color=alt.Color("Category:N", legend=alt.Legend(title="Category")),
            tooltip=["ConceptText:N", "Category:N", "Count:Q"]
        ).properties(
            width=600,
            title="Attended vs Cleared per Concept"
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No concept data to plot.")

def display_student_management():
    """
    Shows the teacher's list of students, plus a selectbox to view more details,
    and pass that student to EeeBee in the chat.
    """
    data = st.session_state.teacher_topicwise_data
    if not data:
        st.info("No teacher data loaded. Please select a batch to fetch data.")
        return

    students = data.get("Students", [])
    if not students:
        st.warning("No student data available.")
        return

    st.markdown("### Student Roster")
    df_students = pd.DataFrame(students)
    st.dataframe(
        df_students[["UserID", "FullName", "TotalConceptCount", "WeakConceptCount", "ClearedConceptCount"]],
        use_container_width=True
    )

    # Select a student to drill down
    student_map = {str(s["FullName"]): s for s in students}
    selected_student_name = st.selectbox("Select a student to view details:", list(student_map.keys()))
    if selected_student_name:
        selected_student = student_map[selected_student_name]
        # Quick summary
        st.write("---")
        st.markdown(f"**Student Name:** {selected_student['FullName']}")
        st.markdown(f"- **Total Concepts:** {selected_student['TotalConceptCount']}")
        st.markdown(f"- **Weak Concepts:** {selected_student['WeakConceptCount']}")
        st.markdown(f"- **Cleared Concepts:** {selected_student['ClearedConceptCount']}")
    
        # Show a quick bar chart
        chart_data = pd.DataFrame({
            "Category": ["Weak", "Cleared", "Remaining"],
            "Count": [
                selected_student["WeakConceptCount"],
                selected_student["ClearedConceptCount"],
                selected_student["TotalConceptCount"] - (
                    selected_student["WeakConceptCount"] + selected_student["ClearedConceptCount"]
                )
            ]
        })
        st.altair_chart(
            alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Category:N", sort=None),
                y="Count:Q",
                color="Category:N",
                tooltip=["Category:N", "Count:Q"]
            ).properties(
                title="Concept Distribution"
            ),
            use_container_width=True
        )

        # Button to discuss with EeeBee
        if st.button("Ask EeeBee about this student"):
            st.session_state["selected_student_for_chat"] = selected_student
            st.success(f"You can now discuss {selected_student['FullName']} with EeeBee in the chat tab.")

def teacher_dashboard():
    # 1) Let teacher select a batch
    batches = st.session_state.auth_data.get("BatchList", [])
    if not batches:
        st.warning("No batches found for the teacher.")
        return

    batch_options = {b['BatchName']: b for b in batches}
    selected_batch_name = st.selectbox("Select a Batch:", list(batch_options.keys()))
    selected_batch = batch_options.get(selected_batch_name)
    selected_batch_id = selected_batch["BatchID"]
    st.session_state.selected_batch_id = selected_batch_id

    # 2) Fetch data from new endpoint
    user_info = st.session_state.auth_data.get('UserInfo', [{}])[0]
    org_code = user_info.get('OrgCode', '012')
    if st.button("Fetch Topicwise Data"):
        data = fetch_teacher_topicwise_data(selected_batch_id, st.session_state.topic_id, org_code)
        if data:
            st.session_state.teacher_topicwise_data = data
            st.success("Data fetched successfully!")

    # 3) Show two tabs: concept-level, student management
    tabs = st.tabs(["Concepts Overview", "Student Management"])
    with tabs[0]:
        display_concept_stats()
    with tabs[1]:
        display_student_management()

# ----------------------------------------------------------------------------------
# 5) CHAT FUNCTIONS
# ----------------------------------------------------------------------------------
def add_initial_greeting():
    """
    Display a greeting if chat_history is empty.
    """
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
            for wc in weak_concepts:
                weak_concepts_text += f"- {wc['ConceptText']}\n"

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
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()

def get_system_prompt():
    """
    Return the system (role=system) prompt that orients EeeBee. 
    In teacher mode, we can incorporate a selected student's context if available.
    """
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    # If teacher, a special system prompt
    if st.session_state.is_teacher:
        prompt = f"""You are EeeBee, an AI assisting teachers instructing {branch_name} 
under the NCERT curriculum. Provide grade-appropriate guidance, best practices, 
and can reference data about students if provided. 
Topic: {topic_name}.

Teaching approach:
- Foster progressive learning
- Encourage critical thinking
- Provide scaffolded learning
- Offer suggestions for remedial steps

Mathematical notation:
- $...$ for inline 
- $$...$$ for display
        """
        if st.session_state.get("selected_student_for_chat"):
            student = st.session_state["selected_student_for_chat"]
            prompt += f"""
Currently discussing student: {student["FullName"]}.
Student's total concepts: {student["TotalConceptCount"]}
Weak concepts: {student["WeakConceptCount"]}
Cleared concepts: {student["ClearedConceptCount"]}.
Suggest strategies or remedials specifically for them.
"""
        return prompt
    else:
        # Student mode prompt
        weak_concepts_list = [wc["ConceptText"] for wc in st.session_state.student_weak_concepts]
        weak_concepts_text = ", ".join(weak_concepts_list) if weak_concepts_list else "none"
        return f"""You are EeeBee, an AI buddy for a student in {branch_name}, topic: {topic_name}.

Student Mode:
- Current learning gaps: [{weak_concepts_text}]
- Provide step-by-step guidance
- Use math notation carefully
- Encourage conceptual understanding
"""

def get_gpt_response(user_input):
    if not client:
        st.error("LLM client is not initialized.")
        return

    system_prompt = get_system_prompt()
    conversation_history = [{"role": "system", "content": system_prompt}]
    for role, content in st.session_state.chat_history:
        conversation_history.append({"role": role, "content": content})

    try:
        with st.spinner("EeeBee is thinking..."):
            # If user specifically references a concept, we can fetch resources
            concept_list = st.session_state.auth_data.get('ConceptList', [])
            mentioned_concept = None
            for c in concept_list:
                if c['ConceptText'].lower() in user_input.lower():
                    mentioned_concept = c['ConceptText']
                    break

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=conversation_history,
                max_tokens=2000,
                stream=False
            )
            gpt_response = response["choices"][0]["message"]["content"].strip()
            st.session_state.chat_history.append(("assistant", gpt_response))

            # If the user specifically requests resources
            if mentioned_concept and any(x in user_input.lower() for x in ["resource", "video", "note", "exercise", "material"]):
                resources = get_resources_for_concept(
                    mentioned_concept,
                    concept_list,
                    st.session_state.topic_id
                )
                if resources:
                    resource_msg = format_resources_message(resources)
                    st.session_state.chat_history.append(("assistant", resource_msg))

    except Exception as e:
        st.error(f"Error in GPT response: {e}")

def display_chat(user_name: str):
    chat_container = st.container()
    with chat_container:
        chat_html = """
        <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd;
        padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
        """
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                chat_html += f"""
                <div style='text-align: left; color: #000; background-color: #e0e7ff;
                padding: 8px; border-radius: 8px; margin-bottom: 5px;'>
                <b>EeeBee:</b> {message}</div>
                """
            else:
                chat_html += f"""
                <div style='text-align: left; color: #fff; background-color: #2563eb;
                padding: 8px; border-radius: 8px; margin-bottom: 5px;'>
                <b>{user_name}:</b> {message}</div>
                """
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    user_input = st.chat_input("Enter your question about the topic:")
    if user_input:
        handle_user_input(user_input)

# ----------------------------------------------------------------------------------
# 6) AUTHENTICATION & MODE-SPECIFIC HANDLING
# ----------------------------------------------------------------------------------
def verify_auth_response(auth_data, is_english_mode):
    if not auth_data:
        return False, None, "No authentication data"
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

def enhanced_login(org_code, login_id, password, topic_id, is_english_mode, user_type_value=3, retries=3):
    api_url = API_AUTH_URL_ENGLISH if is_english_mode else API_AUTH_URL_MATH_SCIENCE
    auth_payload = {
        'OrgCode': org_code,
        'TopicID': int(topic_id),
        'LoginID': login_id,
        'Password': password,
    }
    if not is_english_mode:
        auth_payload['UserType'] = int(user_type_value)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(api_url, json=auth_payload, headers=headers)
            response.raise_for_status()
            auth_data = response.json()
            logging.info(f"Auth Response: {auth_data}")

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
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 403:
                st.header(f"🚫 Authentication failed: Forbidden (403). Attempt {attempt} of {retries}.")
                if attempt < retries:
                    wait_time = 2 ** attempt
                    st.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.header("🚫 Authentication failed: Access Forbidden. Please contact support.")
                    return False, "Access Forbidden (403)."
            else:
                st.error(f"HTTP error occurred: {http_err}")
                return False, str(http_err)
        except requests.exceptions.RequestException as e:
            st.error(f"Request exception: {e}")
            return False, str(e)
        except ValueError as e:
            st.error(f"Invalid JSON response: {e}")
            return False, str(e)
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return False, str(e)

def login_screen():
    try:
        image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_url, width=160)
        with col2:
            st.markdown('<div style="font-size: 2.5em; font-weight: bold; margin-top: 20px;">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)
    except:
        st.markdown("## EeeBee AI Buddy Login")

    st.markdown("#### 🦾 Welcome! Please enter your credentials.")

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
        st.warning("Provide either ?E=xxx (English) or ?T=xxx (Non-English), not both.")
        return
    elif E_value is not None:
        st.session_state.is_english_mode = True
        topic_id = E_value
    elif T_value is not None:
        st.session_state.is_english_mode = False
        topic_id = T_value
    else:
        st.warning("Please provide ?E=... or ?T=... in the URL.")
        return

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if not org_code or not login_id or not password:
            st.error("Please fill in all fields.")
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
            # Changed from st.error to st.header for prominent display
            st.header(f"🚫 Authentication failed: {error_message}")

# ----------------------------------------------------------------------------------
# 7) MAIN APP SCREEN
# ----------------------------------------------------------------------------------
def load_data_parallel():
    """
    Load baseline and all concepts data in parallel for the student case.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(
            fetch_baseline_data,
            st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012'),
            st.session_state.subject_id,
            st.session_state.user_id
        )
        concepts_future = executor.submit(
            fetch_all_concepts,
            st.session_state.auth_data['UserInfo'][0].get('OrgCode', '012'),
            st.session_state.subject_id,
            st.session_state.user_id
        )

        # Wait for results
        try:
            st.session_state.baseline_data = baseline_future.result() or None
        except Exception as e:
            st.error(f"Error fetching baseline data: {e}")

        try:
            st.session_state.all_concepts = concepts_future.result() or []
        except Exception as e:
            st.error(f"Error fetching all concepts: {e}")

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
                        lp = generate_learning_path(concept_text)
                        if lp:
                            st.session_state.student_learning_paths[concept_id] = {
                                "concept_text": concept_text,
                                "learning_path": lp
                            }
                            st.success(f"Learning path generated for {concept_text}!")
                        else:
                            st.error(f"Failed to generate path for {concept_text}.")
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

def display_tabs_parallel():
    tabs = st.tabs(["💬 Chat", "🧠 Learning Path", "🔎 Gap Analyzer™", "📝 Baseline Testing"])
    with tabs[0]:
        st.subheader("Chat with EeeBee")
        add_initial_greeting()
        display_chat(st.session_state.auth_data['UserInfo'][0]['FullName'])

    with tabs[1]:
        st.subheader("Personalized Learning Path")
        display_learning_path_tab()

    with tabs[2]:
        st.subheader("Gap Analyzer")
        display_all_concepts_tab()

    with tabs[3]:
        st.subheader("Baseline Testing Report")
        baseline_testing_report()

def main_screen():
    user_info = st.session_state.auth_data['UserInfo'][0]
    user_name = user_info['FullName']
    topic_name = st.session_state.auth_data.get('TopicName', 'Your Topic')

    col1, col2 = st.columns([9,1])
    with col2:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> 
        EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True
    )

    if st.session_state.is_teacher:
        # Teacher mode: Chat + Teacher Dashboard
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
        with tabs[0]:
            st.subheader("Chat with EeeBee (Teacher Mode)")
            add_initial_greeting()
            display_chat(user_name)
        with tabs[1]:
            st.subheader("Teacher Dashboard")
            teacher_dashboard()
    else:
        # Student mode
        if st.session_state.is_english_mode:
            # For English mode, maybe just show chat
            tab = st.tabs(["💬 Chat"])[0]
            with tab:
                st.subheader("Chat with EeeBee (English Mode)")
                add_initial_greeting()
                display_chat(user_name)
        else:
            # For non-English subjects, we do everything
            if (not st.session_state.baseline_data) or (not st.session_state.all_concepts):
                with st.spinner("Loading additional data..."):
                    load_data_parallel()
            display_tabs_parallel()

def main():
    if st.session_state.is_authenticated:
        main_screen()
    else:
        login_screen()

if __name__ == "__main__":
    main()
