import warnings
import os
import re
import io
import json
import streamlit as st
import openai
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
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from streamlit_cookies_manager import EncryptedCookieManager

# ----------------------------------------------------------------------------
# 1) BASIC SETUP
# ----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Load OpenAI API Key and Cookie Password from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in secrets. Please set OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

try:
    COOKIE_SECRET_KEY = st.secrets["cookies_manager"]["password"]
except KeyError:
    st.error("Cookie encryption password not found in secrets. Please set cookies_manager.password in Streamlit secrets.")
    st.stop()

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# API Endpoints
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
API_TEACHER_WEAK_CONCEPTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts"
API_BASELINE_REPORT = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Baseline_Report_Single_Student"

# Initialize session state variables
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "auth_data" not in st.session_state:
    st.session_state.auth_data = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_teacher" not in st.session_state:
    st.session_state.is_teacher = False
if "topic_id" not in st.session_state:
    st.session_state.topic_id = None
if "student_weak_concepts" not in st.session_state:
    st.session_state.student_weak_concepts = []
if "baseline_data" not in st.session_state:
    st.session_state.baseline_data = None
if "selected_batch_id" not in st.session_state:
    st.session_state.selected_batch_id = None
if "student_learning_paths" not in st.session_state:
    st.session_state.student_learning_paths = {}
if "available_concepts" not in st.session_state:
    st.session_state.available_concepts = {}

# Streamlit page config
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Hide default Streamlit UI components
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2) COOKIE MANAGER INITIALIZATION
# ----------------------------------------------------------------------------
cookies = EncryptedCookieManager(
    prefix="eeebee_ai_buddy",
    password=COOKIE_SECRET_KEY
)

if not cookies.ready():
    # Prevent the app from running until cookies are loaded
    st.stop()

# ----------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ----------------------------------------------------------------------------

def latex_to_image(latex_code, dpi=300):
    """Converts LaTeX code to a PNG image and returns a BytesIO object."""
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

def get_resources_for_concept(concept_text, concept_list, topic_id):
    """Fetches resources for a given concept from the content API."""
    def clean_text(text):
        return text.lower().strip().replace(" ", "")

    matching_concept = next(
        (c for c in concept_list if clean_text(c['ConceptText']) == clean_text(concept_text)),
        None
    )
    if not matching_concept:
        return None

    payload = {
        "TopicID": topic_id,
        "ConceptID": int(matching_concept['ConceptID'])
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        response = requests.post(API_CONTENT_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching resources: {e}")
        return None

def format_resources_message(resources):
    """Formats the resources data into a user-friendly message."""
    message = "Here are the available resources for this concept:\n\n"

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

def generate_exam_questions_pdf(questions, concept_text, user_name):
    """Generates a PDF file containing the exam questions."""
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

                        # Convert latex to image
                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            question_items.append(ListItem(img))
                        last_index = match.end()

                # Leftover text
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
    """Generates a personalized learning path for a given concept."""
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
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500
        ).choices[0].message['content'].strip()
        return gpt_response
    except Exception as e:
        st.error(f"Error generating learning path: {e}")
        return None

def generate_learning_path_pdf(learning_path, concept_text, user_name):
    """Generates a PDF file containing the personalized learning path."""
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

    # Split learning path into sections
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

                # Leftover text
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

# ----------------------------------------------------------------------------
# 4) LOGIN SCREEN
# ----------------------------------------------------------------------------

def login_screen():
    """Displays the login screen for user authentication."""
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

    api_url = None
    topic_id = None

    if E_value and T_value:
        st.warning("Provide either ?E=xx for English OR ?T=xx for Non-English, not both.")
    elif E_value and not T_value:
        # English mode
        st.session_state.is_english_mode = True
        api_url = API_AUTH_URL_ENGLISH
        topic_id = E_value
    elif not E_value and T_value:
        # Non-English mode
        st.session_state.is_english_mode = False
        api_url = API_AUTH_URL_MATH_SCIENCE
        topic_id = T_value
    else:
        st.warning("Please provide ?E=... or ?T=... in the URL.")

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if not api_url or not topic_id:
            st.warning("Please ensure correct E or T parameter is provided.")
            return

        auth_payload = {
            'OrgCode': org_code,
            'TopicID': int(topic_id),
            'LoginID': login_id,
            'Password': password,
        }
        if not st.session_state.is_english_mode:
            auth_payload['UserType'] = user_type_value

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        try:
            with st.spinner("🔄 Authenticating..."):
                auth_response = requests.post(api_url, json=auth_payload, headers=headers)
                auth_response.raise_for_status()
                auth_data = auth_response.json()
                if auth_data.get("statusCode") == 1:
                    # Authentication success
                    user_info = auth_data.get("UserInfo", [{}])[0]
                    user_data = {
                        "UserID": user_info.get("UserID"),
                        "FullName": user_info.get("FullName"),
                        "Email": user_info.get("Email"),
                        "Mobile": user_info.get("Mobile"),
                        "ActiveStatus": user_info.get("ActiveStatus"),
                        "ProfileImgURL": user_info.get("ProfileImgURL"),
                        "SubjectID": auth_data.get("SubjectID", 21),
                        "TopicName": auth_data.get("TopicName", "Unknown Topic"),
                        "SubjectName": auth_data.get("SubjectName", "Unknown Subject"),
                        "BranchName": auth_data.get("BranchName", "Unknown Branch"),
                        "ConceptList": auth_data.get("ConceptList", []),
                        "WeakConceptList": auth_data.get("WeakConceptList", []),
                        "BatchList": auth_data.get("BatchList", [])
                    }

                    # Serialize user data to JSON and store in cookies
                    user_data_json = json.dumps(user_data)
                    cookies["user_data"] = user_data_json
                    # Optionally, set cookie expiry to 7 days
                    cookies["user_data"]["max_age"] = 7 * 24 * 60 * 60  # 7 days in seconds

                    # Store user data in session state
                    st.session_state.auth_data = user_data
                    st.session_state.is_authenticated = True
                    st.session_state.topic_id = int(topic_id)
                    st.session_state.is_teacher = (user_type_value == 2)
                    if not st.session_state.is_teacher:
                        st.session_state.student_weak_concepts = auth_data.get("WeakConceptList", [])

                    # Rerun the app to load the main screen
                    st.rerun()
                else:
                    st.error("🚫 Authentication failed. Check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the authentication API: {e}")

# ----------------------------------------------------------------------------
# 5) MAIN APPLICATION SCREEN
# ----------------------------------------------------------------------------

def main_screen():
    """Displays the main application screen after successful login."""
    user_info = st.session_state.auth_data
    user_name = user_info.get('FullName', 'User')
    topic_name = user_info.get('TopicName', 'Subject')

    # Top bar with logout button
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            # Clear session state
            st.session_state.clear()
            # Remove the user_data cookie
            if "user_data" in cookies:
                del cookies["user_data"]
            st.success("You have been logged out.")
            # Rerun the app to show login screen
            st.rerun()

    # Header with user greeting
    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    # Depending on user type, display appropriate tabs
    if st.session_state.is_teacher:
        # Teacher view: Chat and Dashboard
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
        with tabs[0]:
            st.subheader("Chat with your EeeBee AI buddy", anchor=None)
            add_initial_greeting()
            display_chat()
        with tabs[1]:
            st.subheader("Teacher Dashboard")
            teacher_dashboard()
    else:
        # Student view: Chat, Learning Path, Baseline Testing
        if st.session_state.is_english_mode:
            # English mode: only Chat
            tab = st.tabs(["💬 Chat"])[0]
            with tab:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                display_chat()
        else:
            # Non-English mode: Chat, Learning Path, Baseline Testing
            tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Learning Path", "📝 Baseline Testing"])
            with tab1:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                display_chat()
            with tab2:
                st.subheader("Your Personalized Learning Path")
                display_learning_path_tab()
            with tab3:
                st.subheader("Baseline Testing Report")
                baseline_testing_report()

# ----------------------------------------------------------------------------
# 6) ADDITIONAL FUNCTIONS
# ----------------------------------------------------------------------------

def add_initial_greeting():
    """Adds an initial greeting message to the chat history."""
    if len(st.session_state.chat_history) == 0 and st.session_state.auth_data:
        user_name = st.session_state.auth_data['FullName']
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

        greeting_message = (
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy. "
            f"I'm here to help you with {topic_name}.\n\n"
            f"You can:\n"
            f"1. Ask me questions about any concept\n"
            f"2. Request learning resources (videos, notes, exercises)\n"
            f"3. Get a personalized learning path\n"
            f"{concept_options}"
            f"{weak_concepts_text}\n\n"
            f"What would you like to discuss?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

def handle_user_input(user_input):
    """Handles user input by sending it to GPT and updating the chat history."""
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()

def get_system_prompt():
    """Returns the system prompt based on user type."""
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    if st.session_state.is_teacher:
        system_prompt = f"""
You are a highly knowledgeable educational assistant named EeeBee, specialized in {topic_name}.

Teacher Mode Instructions:
- The user is a teacher instructing {branch_name} students under the NCERT curriculum.
- Provide suggestions for lesson planning, concept explanation, and exam question design.
- Encourage step-by-step reasoning and critical thinking.
- Use LaTeX for math.
"""
    else:
        weak_concepts = st.session_state.auth_data.get('WeakConceptList', [])
        weak_concepts_str = ", ".join([wc['ConceptText'] for wc in weak_concepts]) if weak_concepts else "none"

        system_prompt = f"""
You are a highly knowledgeable educational assistant named EeeBee, specialized in {topic_name}.

Student Mode Instructions:
- The student is in {branch_name}, following NCERT.
- The student's weak concepts: {weak_concepts_str}.
- Provide step-by-step explanations and encourage problem-solving.
- Use LaTeX for math expressions.
"""
    return system_prompt

def get_gpt_response(user_input):
    """Generates a response from GPT based on user input and updates chat history."""
    system_prompt = get_system_prompt()
    conversation_history_formatted = [{"role": "system", "content": system_prompt}]
    conversation_history_formatted += [
        {"role": role, "content": content}
        for role, content in st.session_state.chat_history
    ]

    try:
        with st.spinner("EeeBee is thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation_history_formatted,
                max_tokens=2000
            )
            gpt_text = response.choices[0].message['content'].strip()
            st.session_state.chat_history.append(("assistant", gpt_text))

            # Check if a concept was mentioned and resources are requested
            concept_list = st.session_state.auth_data.get('ConceptList', [])
            mentioned_concept = None
            for concept in concept_list:
                if concept['ConceptText'].lower() in user_input.lower():
                    mentioned_concept = concept['ConceptText']
                    break

            if mentioned_concept and any(x in user_input.lower() for x in ["resource", "video", "note", "exercise", "material"]):
                resources = get_resources_for_concept(mentioned_concept, concept_list, st.session_state.topic_id)
                if resources:
                    resource_message = format_resources_message(resources)
                    st.session_state.chat_history.append(("assistant", resource_message))

    except Exception as e:
        st.error(f"Error in GPT response: {e}")

def display_chat():
    """Displays the chat interface with chat history and input box."""
    user_name = st.session_state.auth_data.get('FullName', 'User')
    chat_container = st.container()

    # Display chat history
    with chat_container:
        chat_history_html = """
        <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd;
        padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
        """
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                chat_history_html += (
                    "<div style='text-align: left; color: #000; background-color: #e0e7ff;"
                    "padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                    f"<b>EeeBee:</b> {message}</div>"
                )
            else:
                chat_history_html += (
                    "<div style='text-align: left; color: #fff; background-color: #2563eb;"
                    "padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                    f"<b>{user_name}:</b> {message}</div>"
                )
        chat_history_html += "</div>"
        st.markdown(chat_history_html, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Enter your question about the topic")
    if user_input:
        handle_user_input(user_input)

def display_learning_path_with_resources(concept_text, learning_path, concept_list, topic_id):
    """Displays the learning path along with additional resources."""
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    with st.expander(f"📚 Learning Path for {concept_text} (Grade: {branch_name})", expanded=False):
        st.markdown(learning_path, unsafe_allow_html=True)

        resources = get_resources_for_concept(concept_text, concept_list, topic_id)
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

        # Download button for PDF
        pdf_bytes = generate_learning_path_pdf(
            learning_path,
            concept_text,
            st.session_state.auth_data.get('FullName', 'User')
        )
        st.download_button(
            label="📥 Download Learning Path as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.auth_data.get('FullName', 'User')}_Learning_Path_{concept_text}.pdf",
            mime="application/pdf"
        )

def display_learning_path_tab():
    """Handles the learning path generation for students."""
    weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
    concept_list = st.session_state.auth_data.get('ConceptList', [])

    if not weak_concepts:
        st.warning("No weak concepts found.")
        return

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

def baseline_testing_report():
    """Displays the baseline testing report."""
    if not st.session_state.baseline_data:
        user_info = st.session_state.auth_data
        user_id = user_info.get('UserID')
        org_code = st.session_state.auth_data.get('OrgCode', '012')
        subject_id = st.session_state.get("subject_id", 21)

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
            with st.spinner("Fetching Baseline Report..."):
                response = requests.post(API_BASELINE_REPORT, json=payload, headers=headers)
                response.raise_for_status()
                st.session_state.baseline_data = response.json()
        except Exception as e:
            st.error(f"Error fetching baseline data: {e}")
            return

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
        col2.metric("Total Concepts.", user_summary.get("TotalQuestion"))
        col3.metric("Cleared Concepts.", user_summary.get("CorrectQuestion"))
        col4.metric("Weak Concepts", user_summary.get("WeakConceptCount"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Difficult Ques. (%)", f"{user_summary.get('DiffQuesPercent', 0)}%")
        col2.metric("Easy Ques. (%)", f"{user_summary.get('EasyQuesPercent', 0)}%")
        duration_hh = user_summary.get("DurationHH", 0)
        duration_mm = user_summary.get("DurationMM", 0)
        col3.metric("Time Taken", f"{duration_hh}h {duration_mm}m")

    # ----------------------------------------------------------------
    # B) Skill-wise Performance (HORIZONTAL BAR)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Skill-wise Performance")
    if s_skills:
        df_skills = pd.DataFrame(s_skills)

        # Horizontal Bar Chart: X-axis => RightAnswerPercent, Y-axis => SubjectSkillName
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
    # C) Concept-wise Data: S.No, Concept Status, Concept Name, Class
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Concept-wise Performance")
    if concept_wise_data:
        df_concepts = pd.DataFrame(concept_wise_data).copy()

        # Create S.No
        df_concepts["S.No."] = range(1, len(df_concepts) + 1)

        # Concept Status => Cleared if RightAnswerPercent == 100, else Not Cleared
        df_concepts["Concept Status"] = df_concepts["RightAnswerPercent"].apply(
            lambda x: "✅" if x == 100.0 else "❌"
        )

        # Rename columns
        df_concepts.rename(
            columns={
                "ConceptText": "Concept Name", 
                "BranchName": "Class"
            }, 
            inplace=True
        )

        # Final columns: S.No., Concept Status, Concept Name, Class
        df_display = df_concepts[["S.No.", "Concept Name","Concept Status", "Class"]]

        # Use hide_index=True for Streamlit dataframe to hide the index
        st.dataframe(df_display, hide_index=True)
    else:
        st.info("No concept-wise data available.")

    # ----------------------------------------------------------------
    # D) Bloom’s Taxonomy Performance (MULTI-COLOR BAR)
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Bloom’s Taxonomy Performance")
    if taxonomy_list:
        df_taxonomy = pd.DataFrame(taxonomy_list)

        # Bar Chart with Different Colors for Each Bloom's Level
        tax_chart = alt.Chart(df_taxonomy).mark_bar().encode(
            x=alt.X('PercentObt:Q', title='Percent Correct', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('TaxonomyText:N', sort='-x', title="Bloom's Level"),
            color=alt.Color('TaxonomyText:N', legend=alt.Legend(title="Bloom's Level")),
            tooltip=['TaxonomyText:N', 'TotalQuestion:Q', 
                     'CorrectAnswer:Q', 'PercentObt:Q']
        ).properties(
            width=700,
            height=300,
            title="Performance by Bloom's Taxonomy Level"
        )
        st.altair_chart(tax_chart, use_container_width=True)
    else:
        st.info("No taxonomy data available.")

def teacher_dashboard():
    """Displays the teacher dashboard with relevant data and functionalities."""
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
        user_info = st.session_state.auth_data
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
        with st.spinner("🔄 Fetching weak concepts..."):
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

        # Bar chart
        df_long = df.melt('Concept', var_name='Category', value_name='Count')
        chart = alt.Chart(df_long).mark_bar().encode(
            x='Concept:N',
            y='Count:Q',
            color='Category:N',
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

        # Additional graphs
        display_additional_graphs(st.session_state.teacher_weak_concepts)

        # Generate Exam Questions
        bloom_level = st.radio(
            "Select Bloom's Taxonomy Level for the Questions",
            ["L1 (Remember)", "L2 (Understand)", "L3 (Apply)", "L4 (Analyze)", "L5 (Evaluate)"],
            index=3
        )

        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.radio("Select a Concept to Generate Exam Questions:", list(concept_list.keys()))

        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions"):
                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                bloom_short = bloom_level.split()[0]

                prompt = (
                    f"You are an educational AI assistant helping a teacher. The teacher wants to create "
                    f"exam questions for the concept '{chosen_concept_text}'.\n"
                    f"The teacher is teaching students in {branch_name}, following the NCERT curriculum.\n"
                    f"Generate a set of 20 challenging and thought-provoking exam questions.\n"
                    f"No answers, only questions. Provide them in LaTeX format as needed.\n"
                    f"Focus on Bloom's Level {bloom_short}.\n"
                )

                with st.spinner("Generating exam questions... Please wait."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=4000
                        )
                        questions = response.choices[0].message['content'].strip()
                        st.session_state.exam_questions = questions

                        # Display the questions
                        st.markdown("### Generated Exam Questions")
                        st.text_area("Exam Questions", questions, height=300)

                        # Provide a download button
                        pdf_bytes = generate_exam_questions_pdf(
                            questions,
                            chosen_concept_text,
                            st.session_state.auth_data.get('FullName', 'Teacher')
                        )
                        st.download_button(
                            label="📥 Download Exam Questions as PDF",
                            data=pdf_bytes,
                            file_name=f"{st.session_state.auth_data.get('FullName', 'Teacher')}_Exam_Questions_{chosen_concept_text}.pdf",
                            mime="application/pdf"
                        )

                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

def display_additional_graphs(weak_concepts):
    """Displays additional graphs for the teacher dashboard."""
    df = pd.DataFrame(weak_concepts)
    total_attended = df["AttendedStudentCount"].sum()
    total_cleared = df["ClearedStudentCount"].sum()
    total_not_cleared = total_attended - total_cleared

    # Donut chart
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

    # Horizontal bar chart
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

def display_learning_path_tab():
    """Handles the learning path generation for students."""
    weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
    concept_list = st.session_state.auth_data.get('ConceptList', [])

    if not weak_concepts:
        st.warning("No weak concepts found.")
        return

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

# ----------------------------------------------------------------------------
# 7) MAIN APP FUNCTION
# ----------------------------------------------------------------------------

def main():
    """Entry point of the Streamlit app."""
    if st.session_state.is_authenticated:
        # User is authenticated, display the main screen
        main_screen()
    else:
        # Not authenticated, check if a cookie exists
        user_data_json = cookies.get("user_data")
        if user_data_json:
            try:
                # Deserialize user data from cookie
                user_data = json.loads(user_data_json)
                # Populate session state
                st.session_state.auth_data = user_data
                st.session_state.is_authenticated = True
                st.session_state.topic_id = user_data.get("TopicID", None)
                # Determine if user is teacher based on presence of BatchList
                st.session_state.is_teacher = bool(user_data.get("BatchList"))
                # If student, get weak concepts
                if not st.session_state.is_teacher:
                    st.session_state.student_weak_concepts = user_data.get("WeakConceptList", [])
                # Rerun the app to display the main screen
                st.rerun()
            except json.JSONDecodeError:
                st.warning("Session cookie invalid. Please log in again.")
                del cookies["user_data"]
                st.session_state.clear()
                login_screen()
        else:
            # No cookie and not authenticated, show login screen
            login_screen()

if __name__ == "__main__":
    main()
