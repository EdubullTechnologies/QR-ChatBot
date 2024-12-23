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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
import pandas as pd
import altair as alt
from pathlib import Path
# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Load OpenAI API Key
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
    initial_sidebar_state="expanded"
)

# Hide default Streamlit components
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load custom CSS
def load_css(file_name: str):
    """
    Load and inject CSS styles from an external file.
    """
    css_path = Path(__file__).parent / file_name
    if css_path.exists():
        with open(css_path) as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        st.error(f"CSS file '{file_name}' not found. Please ensure it is in the same directory as the main app script.")
# Call the function to load styles.css
load_css("styles.css")

# PDF Generation Functions
def generate_exam_questions_pdf(questions, concept_text, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    # Define custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    concept_title_style = ParagraphStyle(
        'ConceptTitle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )

    math_style = ParagraphStyle(
        'Math',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    # Add Title and Subtitle
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
        story.append(Paragraph(lines[0], concept_title_style))
        story.append(Spacer(1, 6))

        # Add questions as a numbered list
        question_items = []
        for line in lines[1:]:
            question_items.append(ListItem(Paragraph(line, normal_style)))
        story.append(ListFlowable(question_items, bulletType='1'))
        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)

    # Get the value of the BytesIO buffer and write it to the output
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes

def generate_learning_path_pdf(learning_path, user_name, topic_name):
    """
    Generate a PDF of the learning path with enhanced formatting.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    
    concept_title_style = ParagraphStyle(
        'ConceptTitle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=16,
        alignment=TA_LEFT,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    math_style = ParagraphStyle(
        'Math',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    # Add Title and Subtitle
    story.append(Paragraph("Personalized Learning Path", title_style))
    story.append(Paragraph(f"For {user_name} - {topic_name}", subtitle_style))
    story.append(Spacer(1, 12))
    
    # Regex for detecting LaTeX math expressions
    MATH_REGEX = r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])"
    
    for concept, path in learning_path.items():
        # Add Concept Title
        story.append(Paragraph(f"Weak Concept: {concept}", concept_title_style))
        story.append(Spacer(1, 6))
        
        # Split the learning path into parts (math and non-math)
        parts = re.split(MATH_REGEX, path)
        
        for part in parts:
            part = part.strip()
            if re.match(MATH_REGEX, part):
                # Handle LaTeX expressions
                try:
                    # Remove LaTeX delimiters for display
                    clean_part = re.sub(r"(\$\$|\\\(|\\\[)", "", part)
                    clean_part = re.sub(r"(\$\$|\\\)|\\\]|\\\[)", "", clean_part)
                    story.append(Paragraph(clean_part, math_style))
                except Exception as e:
                    story.append(Paragraph(f"Math Error: Unable to render expression. {e}", normal_style))
            elif part:
                # Handle regular text
                story.append(Paragraph(part, normal_style))
        
        # Add a spacer after each concept
        story.append(Spacer(1, 12))
        story.append(PageBreak())  # Start each concept on a new page
    
    # Build PDF
    doc.build(story)
    
    # Get the value of the BytesIO buffer and write it to the output
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

# Learning Path Generation Function
def generate_learning_path(weak_concepts):
    """
    Generate a learning path using WeakConceptList.
    """
    learning_path = {}
    for concept in weak_concepts:
        concept_text = concept.get("ConceptText", "Unknown Concept")
        prompt = (
            f"The student is struggling with the weak concept: '{concept_text}'. "
            f"Create a detailed and structured learning path with the following sections:\n\n"
            f"1. **Introduction to the Concept**: Explain the importance and applications of the concept.\n"
            f"2. **Step-by-Step Learning**: Provide a clear sequence of steps to master the concept.\n"
            f"3. **Engagement**: Suggest interactive activities or problem-solving exercises to reinforce learning.\n"
            f"4. **Real-World Applications**: Explain how this concept can be applied in practical situations.\n"
            f"5. **Practice Problems**: Recommend types of problems and exercises to practice.\n"
            f"Ensure the response is well-organized and includes actionable steps."
        )

        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",  # Corrected model name
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1500  # Increased tokens
            ).choices[0].message['content'].strip()
            learning_path[concept_text] = gpt_response
        except Exception as e:
            learning_path[concept_text] = f"Error generating learning path: {e}"
    return learning_path

# Learning Path Display Function
def display_learning_path(learning_path):
    """
    Display the generated learning path with enhanced formatting.
    """
    MATH_REGEX = r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])"  
    
    with st.expander("📚 Generated Learning Path", expanded=True):
        for concept, path in learning_path.items():
            # Concept Header with Emoji
            st.markdown(f"### 🧩 **Weak Concept:** {concept}")
            st.markdown("---")  # Horizontal divider
            
            # Split the learning path into parts (math and non-math)
            parts = re.split(MATH_REGEX, path)
            
            # Initialize an empty string to accumulate markdown content
            markdown_content = ""
            
            for part in parts:
                part = part.strip()
                if re.match(MATH_REGEX, part):
                    try:
                        # Clean LaTeX delimiters for Streamlit's st.latex
                        clean_part = re.sub(r"(\$\$|\\\(|\\\[)", "", part)
                        clean_part = re.sub(r"(\$\$|\\\)|\\\]|\\\[)", "", clean_part)
                        markdown_content += f"#### 📐 Mathematical Expression:\n"
                        markdown_content += f"`{part}`\n\n"
                        # Display LaTeX using st.latex
                        st.latex(clean_part)
                    except Exception as e:
                        markdown_content += f"**Math Error:** Unable to render `{part}`. Error: {e}\n\n"
                elif part:
                    # Append non-math content
                    markdown_content += f"{part}\n\n"
            
            # Display the accumulated markdown content
            if markdown_content:
                st.markdown(markdown_content)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Extra space after each concept

# Enhanced Resources Display Function
def display_resources(content_data):
    with st.expander("📚 Resources", expanded=True):
        # Display the generated concept description from ChatGPT
        concept_description = st.session_state.get("generated_description", "No description available.")
        st.markdown(f"### 📖 Concept Description\n{concept_description}\n")
        
        # Organize resources into categories with icons
        resources = {
            "🎥 Videos": content_data.get("Video_List", []),
            "📄 Notes": content_data.get("Notes_List", []),
            "📝 Exercises": content_data.get("Exercise_List", [])
        }

        for category, items in resources.items():
            if items:
                st.markdown(f"#### {category}")
                for item in items:
                    if category == "🎥 Videos":
                        video_url = item.get("LectureLink", f"https://www.edubull.com/courses/videos/{item.get('LectureID', '')}")
                        st.markdown(f"- [Video 📹]({video_url})")
                    elif category == "📄 Notes":
                        note_url = f"{item.get('FolderName', '')}{item.get('PDFFileName', '')}"
                        st.markdown(f"- [Notes 📄]({note_url})")
                    elif category == "📝 Exercises":
                        exercise_url = f"{item.get('FolderName', '')}{item.get('ExerciseFileName', '')}"
                        st.markdown(f"- [Exercise 📝]({exercise_url})")
                st.markdown("---")  # Divider after each category

# Teacher Dashboard Function
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
        with st.spinner("🔄 Fetching weak concepts..."):
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

        # Display DataFrame with enhanced styling
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Weak Concepts Overview")
        st.dataframe(df.style.set_properties(**{
            'background-color': '#ffffff',
            'color': '#333333',
            'border-color': '#4B0082'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4B0082'), ('color', 'white'), ('padding', '8px')]},
            {'selector': 'td', 'props': [('padding', '8px')]}
        ]).hide_index())
        st.markdown('</div>', unsafe_allow_html=True)

        # Interactive Charts
        df_long = df.melt('Concept', var_name='Category', value_name='Count')
        chart = alt.Chart(df_long).mark_bar().encode(
            x=alt.X('Concept:N', sort=None),
            y='Count:Q',
            color='Category:N',
            tooltip=['Concept:N', 'Category:N', 'Count:Q']
        ).properties(
            title='Weak Concepts Overview',
            width=700
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # Exam Questions Section
        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.selectbox("Select a Concept to Generate Exam Questions:", list(concept_list.keys()), key="teacher_exam_concept_select")
        
        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions", key="generate_exam_questions_button"):
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
                            model="gpt-4",  # Corrected model name
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=2000
                        )
                        questions = response.choices[0].message['content'].strip()
                        st.session_state.exam_questions = questions
                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

    # Login Screen Function
    def login_screen():
        try:
            image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image_url, width=160)
            with col2:
                st.markdown('<div class="app-header">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

        st.markdown('<div class="app-subtitle">🦾 Welcome! Please enter your credentials to chat with your AI Buddy!</div>', unsafe_allow_html=True)

        user_type = st.radio("Select User Type", ["Student", "Teacher"], key="user_type")
        user_type_value = 2 if user_type == "Teacher" else None  # Set to None for students

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

        if st.button("🚀 Login and Start Chatting!", key="login_button") and not st.session_state.is_authenticated:
            if topic_id is None or api_url is None:
                st.warning("Please ensure correct E or T parameter is provided.")
                return

            auth_payload = {
                'OrgCode': org_code,
                'TopicID': int(topic_id),
                'LoginID': login_id,
                'Password': password,
            }

            if user_type_value:
                auth_payload['UserType'] = user_type_value  # Only add if user is Teacher

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
                        st.session_state.auth_data = auth_data
                        st.session_state.is_authenticated = True
                        st.session_state.topic_id = int(topic_id)
                        st.session_state.is_teacher = (user_type_value == 2)

                        # Debugging: Display auth_data
                        st.markdown("🔍 **Auth Data:**")
                        st.json(auth_data)

                        st.experimental_rerun()
                    else:
                        st.error("🚫 Authentication failed. Please check your credentials.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the authentication API: {e}")

# Main Screen Function
def main_screen():
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']

    # Logout Button
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    # App Header and Subtitle
    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        <div class="app-header">Hello {user_name}</div>
        <div class="app-subtitle"><img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> EeeBee AI Buddy is here to help you with <span style="color:#4B0082;">{topic_name}</span></div>
        """,
        unsafe_allow_html=True,
    )

    # Determine User Mode
    if st.session_state.is_teacher:
        # Teacher Mode Tabs
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
    else:
        # Student Mode Tabs
        if st.session_state.is_english_mode:
            tabs = st.tabs(["💬 Chat", "🧠 Learning Path"])
        else:
            tabs = st.tabs(["💬 Chat", "🧠 Learning Path", "📚 Concepts"])

    # Chat Tab (Unchanged)
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

    # Learning Path Tab
    if not st.session_state.is_teacher:
        with tabs[1]:
            # Learning Path Tab Content
            st.markdown('<div class="learning-path-header">🧠 Learning Path</div>', unsafe_allow_html=True)
            weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])

            if not st.session_state.learning_path_generated:
                if st.button("🧠 Generate Learning Path", key="generate_learning_path_button"):
                    if weak_concepts:
                        with st.spinner("Generating learning path..."):
                            st.session_state.learning_path = generate_learning_path(weak_concepts)
                            st.session_state.learning_path_generated = True
                    else:
                        st.error("No weak concepts found!")

            if st.session_state.learning_path_generated and st.session_state.learning_path:
                display_learning_path(st.session_state.learning_path)
                if st.button("📄 Download Learning Path as PDF", key="download_learning_path_pdf_button"):
                    with st.spinner("📄 Generating PDF..."):
                        try:
                            pdf_bytes = generate_learning_path_pdf(
                                st.session_state.learning_path,
                                user_name,
                                topic_name
                            )
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_bytes,
                                file_name=f"{user_name}_Learning_Path_{topic_name}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Error creating PDF: {e}")

    # Concepts Tab (Only for Non-English Students)
    if not st.session_state.is_teacher and not st.session_state.is_english_mode:
        with tabs[2]:
            # Concepts Tab Content
            st.markdown('<div class="dashboard-header">📚 Concepts</div>', unsafe_allow_html=True)
            concept_list = st.session_state.auth_data.get('ConceptList', [])
            concept_options = {concept['ConceptText']: concept['ConceptID'] for concept in concept_list}

            # Display concepts as clickable cards
            st.markdown('<div class="concepts-container">', unsafe_allow_html=True)
            for concept_text, concept_id in concept_options.items():
                st.markdown(
                    f"""
                    <div class="concept-card" onclick="document.getElementById('concept_{concept_id}').click();">
                        <div class="concept-title">{concept_text}</div>
                    </div>
                    <form id="concept_{concept_id}" method="post">
                        <input type="hidden" name="concept_id" value="{concept_id}">
                        <button type="submit" style="display:none;"></button>
                    </form>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle form submissions
            concept_id = None
            query_params = st.experimental_get_query_params()
            if "concept_id" in query_params:
                concept_id = int(query_params["concept_id"][0])
                st.session_state.selected_concept_id = concept_id
                st.experimental_set_query_params()  # Clear the query params after handling

            if st.session_state.selected_concept_id:
                with st.spinner("🔄 Loading concept content..."):
                    load_concept_content()

    # Teacher Dashboard Tab
    if st.session_state.is_teacher:
        with tabs[1]:
            # Teacher Dashboard Content
            st.markdown('<div class="dashboard-header">📊 Teacher Dashboard</div>', unsafe_allow_html=True)
            teacher_dashboard()

# Display login or main screen based on authentication
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
