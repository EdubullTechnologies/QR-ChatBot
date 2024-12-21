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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
import pandas as pd
import altair as alt

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("API key for OpenAI not found in secrets.")

openai.api_key = OPENAI_API_KEY

# API URLs
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"
API_TEACHER_WEAK_CONCEPTS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts"

# Initialize session states if not present
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
    st.session_state.is_english_mode = False  # default initialization

# Page config
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

def generate_exam_questions_pdf(questions, concept_text, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    # Define custom styles
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

    # Add title and subtitle
    story.append(Paragraph("Exam Questions", title_style))
    user_name_display = user_name if user_name else "Teacher"
    concept_text_display = concept_text if concept_text else "Selected Concept"
    story.append(Paragraph(f"For {user_name_display} - {concept_text_display}", subtitle_style))
    story.append(Spacer(1, 12))

    # Parse questions into sections
    sections = re.split(r'\n\n', questions.strip())
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        # First line as a section title
        story.append(Paragraph(lines[0], section_title_style))
        story.append(Spacer(1, 8))

        # Add questions as a numbered list
        question_items = []
        for line in lines[1:]:
            question_items.append(ListItem(Paragraph(line, question_style)))
        story.append(ListFlowable(question_items, bulletType='1'))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_learning_path_pdf(learning_path, user_name, topic_name):
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
    topic_name_display = topic_name if topic_name else "Topic"
    story.append(Paragraph(f"For {user_name_display} - {topic_name_display}", subtitle_style))
    story.append(Spacer(1, 12))

    MATH_REGEX = r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])"
    for concept, path in learning_path.items():
        story.append(Paragraph(f"Weak Concept: {concept}", styles['Heading3']))
        story.append(Spacer(1, 6))
        parts = re.split(MATH_REGEX, path)
        for part in parts:
            part = part.strip()
            if part:
                story.append(Paragraph(part, content_style))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def generate_learning_path(weak_concepts):
    learning_path = {}
    for concept in weak_concepts:
        concept_text = concept.get("ConceptText", "Unknown Concept")
        prompt = (
            f"The student is struggling with the weak concept: '{concept_text}'. "
            f"Create a detailed and structured learning path with the following sections:\n\n"
            f"1. Introduction\n2. Step-by-Step Learning\n3. Engagement\n4. Real-World Applications\n5. Practice Problems\n"
            f"Ensure the response is well-organized and includes actionable steps."
        )

        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500
            ).choices[0].message['content'].strip()
            learning_path[concept_text] = gpt_response
        except Exception as e:
            learning_path[concept_text] = f"Error generating learning path: {e}"
    return learning_path

def display_learning_path(learning_path):
    MATH_REGEX = r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])"
    with st.expander("📚 Generated Learning Path", expanded=True):
        for concept, path in learning_path.items():
            st.markdown(f"### Weak Concept: {concept}")
            parts = re.split(MATH_REGEX, path)
            for part in parts:
                part = part.strip()
                if part:
                    st.markdown(part)

def display_additional_graphs(weak_concepts):
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
        x='Count:Q',
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
        response = requests.post(API_TEACHER_WEAK_CONCEPTS, json=params, headers=headers)
        response.raise_for_status()
        weak_concepts = response.json()
        st.session_state.teacher_weak_concepts = weak_concepts

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
            color='Category:N',
            tooltip=['Concept:N', 'Category:N', 'Count:Q']
        ).properties(
            title='Weak Concepts Overview',
            width=600
        )

        rule = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_rule(color='red', strokeDash=[4,4]).encode(
            y='y:Q'
        )
        text = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_text(
            align='left', dx=5, dy=-5, color='red'
        ).encode(
            y='y:Q',
            text=alt.value(f'Total Students: {total_students}')
        )

        final_chart = (chart + rule + text).interactive()
        st.altair_chart(final_chart, use_container_width=True)

        display_additional_graphs(st.session_state.teacher_weak_concepts)

        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.selectbox("Select a Concept to Generate Exam Questions:", list(concept_list.keys()))

        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions"):
                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                prompt = (
                    f"You are an educational AI assistant helping a teacher. The teacher wants to create exam questions for the concept '{chosen_concept_text}'.\n"
                    f"The teacher is teaching students in {branch_name}, following the NCERT curriculum.\n"
                    f"Generate a set of 20 challenging and thought-provoking exam questions related to this concept.\n"
                    f"Generated questions should be aligned with NEP 2020 and NCF guidelines.\n"
                    f"- Vary in difficulty.\n"
                    f"- Encourage critical thinking.\n"
                    f"- Be clearly formatted and numbered.\n\n"
                    f"Do not provide the answers, only the questions."
                )

                with st.spinner("Generating exam questions... Please wait."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=2000
                        )
                        questions = response.choices[0].message['content'].strip()
                        st.session_state.exam_questions = questions
                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

            if st.session_state.exam_questions:
                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                st.markdown(f"### Generated Exam Questions for {branch_name}")
                st.markdown(st.session_state.exam_questions)

                pdf_bytes = generate_exam_questions_pdf(
                    st.session_state.exam_questions,
                    st.session_state.selected_teacher_concept_text,
                    st.session_state.auth_data['UserInfo'][0]['FullName']
                )
                st.download_button(
                    label="Download Exam Questions as PDF",
                    data=pdf_bytes,
                    file_name=f"Exam_Questions_{st.session_state.selected_teacher_concept_text}.pdf",
                    mime="application/pdf"
                )

def login_screen():
    try:
        image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_url, width=160)
        st.markdown("""<style>
        @media only screen and (max-width: 600px) {
            .title { font-size: 2.5em; margin-top: 20px; text-align: center; }
        }
        @media only screen and (min-width: 601px) {
            .title { font-size: 4em; font-weight: bold; margin-top: 90px; margin-left: -125px; text-align: left; }
        }
        </style>""", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="title">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

    st.markdown('<h3 style="font-size: 1.5em;">🦾 Welcome! Please enter your credentials to chat with your AI Buddy!</h3>', unsafe_allow_html=True)

    user_type = st.radio("Select User Type", ["Student", "Teacher"])
    user_type_value = 2 if user_type == "Teacher" else 1

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

    # Determine mode based on E and T
    if E_value is not None and T_value is not None:
        st.warning("Please provide either E for English OR T for Non-English, not both.")
    elif E_value is not None and T_value is None:
        # English mode
        st.session_state.is_english_mode = True
        api_url = API_AUTH_URL_ENGLISH
        topic_id = E_value
    elif E_value is None and T_value is not None:
        # Non-English mode
        st.session_state.is_english_mode = False
        api_url = API_AUTH_URL_MATH_SCIENCE
        topic_id = T_value
    else:
        # Neither E nor T provided
        st.warning("Please provide E for English mode or T for Non-English mode.")

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if topic_id is None or api_url is None:
            st.warning("Please ensure correct E or T parameter is provided.")
            return

        auth_payload = {
            'OrgCode': org_code,
            'TopicID': int(topic_id),
            'LoginID': login_id,
            'Password': password,
            'UserType': user_type_value
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        try:
            auth_response = requests.post(api_url, json=auth_payload, headers=headers)
            auth_response.raise_for_status()
            auth_data = auth_response.json()
            if auth_data.get("statusCode") == 1:
                st.session_state.auth_data = auth_data
                st.session_state.is_authenticated = True
                st.session_state.topic_id = int(topic_id)
                st.session_state.is_teacher = (user_type_value == 2)
                st.rerun()
            else:
                st.error("🚫 Authentication failed. Please check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the authentication API: {e}")

def add_initial_greeting():
    if len(st.session_state.chat_history) == 0 and st.session_state.auth_data:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data['TopicName']
        greeting_message = (
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy. "
            f"How can I help you with {topic_name} today?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

def handle_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()

def get_gpt_response(user_input):
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    if st.session_state.is_teacher:
        system_prompt = f"""You are a highly knowledgeable educational assistant named EeeBee, specialized in {topic_name}.
Teacher Mode Instructions:
- The user is a teacher teaching students in {branch_name}, following the NCERT curriculum.
- Provide suggestions on how to explain concepts, create assessments, and improve student understanding at the {branch_name} level.
- Offer insights into student difficulties and how to address them.
- Maintain a professional, informative tone and provide curriculum-aligned advice."""
    else:
        system_prompt = f"""You are a highly knowledgeable educational assistant named EeeBee, specialized in {topic_name}.
Student Mode Instructions:
- The student is in {branch_name}, following the NCERT curriculum.
- only talk about {topic_name} and nothing else.
- Encourage the student to think critically and solve problems step-by-step.
- Avoid giving direct answers; ask guiding questions.
- Be supportive and build understanding and confidence.
- If asked for exam questions, provide progressive questions aligned with NCERT and suitable for {branch_name} students."""

    conversation_history_formatted = [{"role": "system", "content": system_prompt}]
    conversation_history_formatted += [{"role": role, "content": content} for role, content in st.session_state.chat_history]

    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=conversation_history_formatted,
            max_tokens=1000
        ).choices[0].message['content'].strip()
        st.session_state.chat_history.append(("assistant", gpt_response))
    except Exception as e:
        st.error(f"Error in GPT response generation: {e}")

def load_concept_content():
    selected_concept_id = st.session_state.selected_concept_id
    selected_concept_name = next(
        (concept['ConceptText'] for concept in st.session_state.auth_data['ConceptList'] if concept['ConceptID'] == selected_concept_id),
        "Unknown Concept"
    )
    content_payload = {
        'TopicID': st.session_state.topic_id,
        'ConceptID': int(selected_concept_id)
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    try:
        content_response = requests.post(API_CONTENT_URL, json=content_payload, headers=headers)
        content_response.raise_for_status()
        content_data = content_response.json()

        prompt = f"Provide a concise and educational description of the concept '{selected_concept_name}' to help students understand it better."
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=500
        ).choices[0].message['content'].strip()
        gpt_response = gpt_response.replace("This concept", selected_concept_name).replace("this concept", selected_concept_name)
        gpt_response += "\n\nYou can check the resources below for more information."
        st.session_state.generated_description = gpt_response

        display_resources(content_data)

    except requests.exceptions.RequestException as req_err:
        st.error(f"Error fetching content: {req_err}")
    except Exception as e:
        st.error(f"Error generating concept description: {e}")

def display_resources(content_data):
    with st.expander("Resources", expanded=True):
        concept_description = st.session_state.get("generated_description", "No description available.")
        st.markdown(f"### Concept Description\n{concept_description}\n")
        if content_data.get("Video_List"):
            for video in content_data["Video_List"]:
                video_url = video.get("LectureLink", f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}")
                st.write(f"- [Video]({video_url})")
        if content_data.get("Notes_List"):
            for note in content_data["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                st.write(f"- [Notes]({note_url})")
        if content_data.get("Exercise_List"):
            for exercise in content_data["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                st.write(f"- [Exercise]({exercise_url})")

def main_screen():
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']

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
        # Teacher Mode
        if st.session_state.is_english_mode:
            # Teacher in English mode: Chat + Teacher Dashboard
            tabs = st.tabs(["Chat", "Teacher Dashboard"])
            with tabs[0]:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>EeeBee:</b> {message}</div>"
                        else:
                            chat_history_html += f"<div style='text-align: left; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>{user_name}:</b> {message}</div>"
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)
                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)

            with tabs[1]:
                st.subheader("Teacher Dashboard")
                teacher_dashboard()
        else:
            # Teacher in Non-English mode: Chat + Teacher Dashboard
            tabs = st.tabs(["Chat", "Teacher Dashboard"])
            with tabs[0]:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>EeeBee:</b> {message}</div>"
                        else:
                            chat_history_html += f"<div style='text-align: left; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>{user_name}:</b> {message}</div>"
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)
                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)

            with tabs[1]:
                st.subheader("Teacher Dashboard")
                teacher_dashboard()
    else:
        # Student Mode
        if st.session_state.is_english_mode:
            # English Student: only Chat
            tab1 = st.tabs(["Chat"])[0]
            with tab1:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>EeeBee:</b> {message}</div>"
                        else:
                            chat_history_html += f"<div style='text-align: left; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>{user_name}:</b> {message}</div>"
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)
                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)
        else:
            # Non-English Student: Chat + Learning Path + Concepts
            tab1, tab2, tab3 = st.tabs(["Chat", "Learning Path", "Concepts"])
            with tab1:
                st.subheader("Chat with your EeeBee AI buddy", anchor=None)
                add_initial_greeting()
                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>EeeBee:</b> {message}</div>"
                        else:
                            chat_history_html += f"<div style='text-align: left; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>{user_name}:</b> {message}</div>"
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)
                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)

            with tab2:
               

                # Attempt to fetch weak concepts
                weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
                st.write("Debug: weak_concepts -> ", weak_concepts)

                if not st.session_state.learning_path_generated:
                    if st.button("🧠 Generate Learning Path"):
                        # Check if we have weak concepts
                        if weak_concepts:
                            with st.spinner("Generating learning path..."):
                                st.session_state.learning_path = generate_learning_path(weak_concepts)
                                st.session_state.learning_path_generated = True
                        else:
                            st.error("No weak concepts found! Please check auth_data or verify the API response.")
                
                if st.session_state.learning_path_generated and st.session_state.learning_path:
                    display_learning_path(st.session_state.learning_path)
                    if st.button("📄 Download Learning Path as PDF"):
                        try:
                            pdf_bytes = generate_learning_path_pdf(
                                st.session_state.learning_path,
                                user_name,
                                topic_name
                            )
                            st.download_button(
                                label="Click here to download PDF",
                                data=pdf_bytes,
                                file_name=f"{user_name}_Learning_Path_{topic_name}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Error creating PDF: {e}")

            with tab3:
                concept_list = st.session_state.auth_data.get('ConceptList', [])
                concept_options = {concept['ConceptText']: concept['ConceptID'] for concept in concept_list}
                for c_text, c_id in concept_options.items():
                    if st.button(c_text, key=f"concept_{c_id}"):
                        st.session_state.selected_concept_id = c_id

                if st.session_state.selected_concept_id:
                    load_concept_content()

if st.session_state.is_authenticated:
    main_screen()
    st.stop()
else:
    placeholder = st.empty()
    with placeholder.container():
        login_screen()
