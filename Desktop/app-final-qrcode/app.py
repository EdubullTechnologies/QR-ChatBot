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
from matplotlib import rcParams

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
if "student_learning_paths" not in st.session_state:
    st.session_state.student_learning_paths = {}  # Dictionary to store multiple learning paths
if "student_weak_concepts" not in st.session_state:
    st.session_state.student_weak_concepts = []
if "branch_name" not in st.session_state:
    st.session_state.branch_name = "their class"

# Page config
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
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


# Helper function to convert LaTeX to image
def latex_to_image(latex_code, dpi=300):
    """
    Converts LaTeX code to a PNG image and returns it as a BytesIO object.
    """
    try:
        # Adjust figure size based on display or inline math
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


# ================= PDF GENERATION FUNCTIONS =================
def generate_exam_questions_pdf(questions, concept_text, user_name, resources=None):
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
            # Detect LaTeX expressions in the line
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
                # Keep track of the last index processed
                last_index = 0
                for match in latex_matches:
                    if match.group(1):
                        # Display math
                        latex = match.group(1).strip()
                        display_math = True
                    else:
                        # Inline math
                        latex = match.group(2).strip()
                        display_math = False

                    if latex:
                        # Add text before LaTeX
                        pre_text = line[last_index:match.start()]
                        if pre_text:
                            question_items.append(ListItem(Paragraph(pre_text, question_style)))

                        # Convert LaTeX to image
                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            # Adjust image size based on math type
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            question_items.append(ListItem(img))

                        # Update last_index
                        last_index = match.end()

                # Add remaining text after last LaTeX
                post_text = line[last_index:]
                if post_text:
                    question_items.append(ListItem(Paragraph(post_text, question_style)))
            else:
                # Regular text
                question_items.append(ListItem(Paragraph(line, question_style)))
        story.append(ListFlowable(question_items, bulletType='1'))
        story.append(Spacer(1, 12))

    # Add resources if available
    if resources:
        story.append(PageBreak())
        story.append(Paragraph("📚 Additional Resources", title_style))
        story.append(Spacer(1, 12))

        # Videos
        if resources.get("Video_List"):
            story.append(Paragraph("🎥 Videos", section_title_style))
            for video in resources["Video_List"]:
                video_url = video.get('LectureLink')
                video_title = video.get('LectureTitle', 'No Title')
                if video_url:
                    story.append(Paragraph(f"- <a href='{video_url}'>Video: {video_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Video: {video_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

        # Notes
        if resources.get("Notes_List"):
            story.append(Paragraph("📄 Notes", section_title_style))
            for note in resources["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                note_title = note.get("NotesTitle", "No Title")
                if note_url:
                    story.append(Paragraph(f"- <a href='{note_url}'>Notes: {note_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Notes: {note_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

        # Exercises
        if resources.get("Exercise_List"):
            story.append(Paragraph("📝 Exercises", section_title_style))
            for exercise in resources["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                exercise_title = exercise.get("ExerciseTitle", "No Title")
                if exercise_url:
                    story.append(Paragraph(f"- <a href='{exercise_url}'>Exercise: {exercise_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Exercise: {exercise_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_learning_path_pdf(learning_path, concept_text, user_name, resources=None):
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
    content_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    # Add title and subtitle
    story.append(Paragraph("Personalized Learning Path", title_style))
    user_name_display = user_name if user_name else "Student"
    concept_text_display = concept_text if concept_text else "Selected Concept"
    story.append(Paragraph(f"For {user_name_display} - {concept_text_display}", subtitle_style))
    story.append(Spacer(1, 12))

    # Add learning path content
    sections = re.split(r'\n\n', learning_path.strip())
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        # First line as section header
        story.append(Paragraph(lines[0], section_title_style))
        story.append(Spacer(1, 6))

        for line in lines[1:]:
            # Detect LaTeX expressions in the line
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
                # Keep track of the last index processed
                last_index = 0
                for match in latex_matches:
                    if match.group(1):
                        # Display math
                        latex = match.group(1).strip()
                        display_math = True
                    else:
                        # Inline math
                        latex = match.group(2).strip()
                        display_math = False

                    if latex:
                        # Add text before LaTeX
                        pre_text = line[last_index:match.start()]
                        if pre_text:
                            story.append(Paragraph(pre_text, content_style))

                        # Convert LaTeX to image
                        img_buffer = latex_to_image(latex)
                        if img_buffer:
                            # Adjust image size based on math type
                            if display_math:
                                img = RLImage(img_buffer, width=4*inch, height=1*inch)
                            else:
                                img = RLImage(img_buffer, width=2*inch, height=0.5*inch)
                            story.append(img)

                        # Update last_index
                        last_index = match.end()

                # Add remaining text after last LaTeX
                post_text = line[last_index:]
                if post_text:
                    story.append(Paragraph(post_text, content_style))
            else:
                # Regular text
                story.append(Paragraph(line, content_style))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    # Add resources if available
    if resources:
        story.append(PageBreak())
        story.append(Paragraph("📚 Additional Resources", title_style))
        story.append(Spacer(1, 12))

        # Videos
        if resources.get("Video_List"):
            story.append(Paragraph("🎥 Videos", section_title_style))
            for video in resources["Video_List"]:
                video_url = video.get('LectureLink')
                video_title = video.get('LectureTitle', 'No Title')
                if video_url:
                    story.append(Paragraph(f"- <a href='{video_url}'>Video: {video_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Video: {video_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

        # Notes
        if resources.get("Notes_List"):
            story.append(Paragraph("📄 Notes", section_title_style))
            for note in resources["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                note_title = note.get("NotesTitle", "No Title")
                if note_url:
                    story.append(Paragraph(f"- <a href='{note_url}'>Notes: {note_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Notes: {note_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

        # Exercises
        if resources.get("Exercise_List"):
            story.append(Paragraph("📝 Exercises", section_title_style))
            for exercise in resources["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                exercise_title = exercise.get("ExerciseTitle", "No Title")
                if exercise_url:
                    story.append(Paragraph(f"- <a href='{exercise_url}'>Exercise: {exercise_title}</a>", styles['Normal']))
                else:
                    story.append(Paragraph(f"- Exercise: {exercise_title} (Link not available)", styles['Normal']))
            story.append(Spacer(1, 12))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ================= LEARNING PATH GENERATION FUNCTION =================
def generate_learning_path(concept_text):
    """
    Incorporate the class/grade (branch_name) into the prompt so the content
    is pitched at the student's level.
    """
    branch_name = st.session_state.branch_name
    prompt = (
        f"You are a highly experienced educational AI assistant specializing in the NCERT curriculum. "
        f"A student in {branch_name} is struggling with the weak concept: '{concept_text}'. "
        f"Please create a structured, step-by-step learning path tailored to {branch_name} students, ensuring clarity, engagement, and curriculum alignment. "
        f"Your plan should include the following sections:\n\n"
        f"1. **Introduction to the Concept**: Briefly explain the concept in simple terms, highlighting its importance and core principles. "
        f"Emphasize how understanding it will benefit students in their studies and daily life.\n\n"
        f"2. **Step-by-Step Learning**: Provide a clear, logical progression of the subtopics or skills needed to master this concept. "
        f"Include any foundational knowledge required and tips for retaining each step.\n\n"
        f"3. **Engagement**: Suggest interactive, hands-on activities or problem-based learning tasks that reinforce the concept. "
        f"Mention creative ways to make these exercises fun and relevant to real-life scenarios.\n\n"
        f"4. **Real-World Applications**: Illustrate how the concept applies to practical situations or real-world problems. "
        f"Offer examples that resonate with {branch_name}-level students’ experiences or surroundings.\n\n"
        f"5. **Practice Problems**: Recommend specific types of problems or exercises students can work on. "
        f"Vary the difficulty level, ensuring alignment with NCERT guidelines. "
        f"Encourage students to think critically and to practice regularly.\n\n"
        f"Throughout your explanation, ensure that **all mathematical expressions are enclosed within LaTeX delimiters** "
        f"(`$...$` for inline and `$$...$$` for display math). "
        f"Your goal is to provide a clear, engaging, and age-appropriate roadmap that helps the student gain confidence and proficiency in '{concept_text}'."
    )

    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",  # Correct model name
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500
        ).choices[0].message['content'].strip()
        return gpt_response
    except Exception as e:
        st.error(f"Error generating learning path: {e}")
        return None


# ================= LEARNING PATH DISPLAY FUNCTION =================
def display_learning_path(concept_text, learning_path, resources=None):
    """
    Display the generated learning path with enhanced formatting for a single concept.
    Includes resources like videos, notes, and exercises.
    """
    branch_name = st.session_state.branch_name
    with st.expander(f"📚 Learning Path for {concept_text} according to your learning gaps for {branch_name}", expanded=False):
        st.markdown(learning_path, unsafe_allow_html=True)

        # Download Button for the specific learning path
        pdf_bytes = generate_learning_path_pdf(
            learning_path,
            concept_text,
            st.session_state.auth_data['UserInfo'][0]['FullName'],
            resources  # Pass the resources here
        )
        st.download_button(
            label="📥 Download Learning Path as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.auth_data['UserInfo'][0]['FullName']}_Learning_Path_{concept_text}.pdf",
            mime="application/pdf"
        )


# ================= RESOURCES FETCHING FUNCTION =================
def fetch_resources_by_concept_id(topic_id, concept_id):
    """
    Fetches resources for a given TopicID and ConceptID using the WeakConcept_Remedy_List_ByConceptID API.
    Returns the resources as a dictionary.
    """
    url = API_CONTENT_URL
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
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        content_data = response.json()
        return content_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching resources for ConceptID {concept_id}: {e}")
        return {}


# ================= CHAT-RELATED FUNCTIONS =================
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
        st.experimental_rerun()  # Use experimental_rerun for immediate UI update


def get_system_prompt():
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.branch_name

    if st.session_state.is_teacher:
        # TEACHER MODE PROMPT
        system_prompt = f"""
You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {topic_name}.

Teacher Mode Instructions:
- The user is a teacher instructing {branch_name} students under the NCERT curriculum.
- Provide detailed suggestions on how to explain concepts and design assessments for the {branch_name} level.
- Offer insights into common student difficulties and ways to address them.
- Encourage a teaching methodology where students learn progressively, asking guiding questions rather than providing direct answers.
- Maintain a professional, informative tone, and ensure all advice aligns with the NCERT curriculum.
- Keep all mathematical expressions within LaTeX delimiters:
  - Use `$...$` for inline math
  - Use `$$...$$` for display math
- Emphasize to the teacher the importance of fostering critical thinking and step-by-step reasoning in students.
- If the teacher requests sample questions or exercises, provide them in a progressive manner, ensuring they prompt the student to reason through each step.
- Do not provide final solutions outright; instead, suggest ways to guide students toward the solution on their own.
        """
    else:
        # STUDENT MODE PROMPT
        weak_concepts = [concept['ConceptText'] for concept in st.session_state.student_weak_concepts]
        weak_concepts_text = ", ".join(weak_concepts) if weak_concepts else "none"

        system_prompt = f"""
You are a highly knowledgeable educational assistant named EeeBee, built by iEdubull, and specialized in {topic_name}.

Student Mode Instructions:
- The student is in {branch_name}, following the NCERT curriculum.
- The student's weak concepts include: {weak_concepts_text}.
- Mention that you identified these weak concepts from the Edubull app, which are visible in the student's profile.
- Always provide the weak concepts as a list: [{weak_concepts_text}].
- Focus strictly on {topic_name} and avoid unrelated content.
- Encourage the student to solve problems step-by-step and think critically.
- Avoid giving direct, complete answers. Instead, ask guiding questions and offer hints that lead them to discover the solution.
- Support the student's reasoning and help them build confidence in their problem-solving skills.
- If the student asks for exam or practice questions, present them in a progressive manner aligned with {branch_name} NCERT guidelines.
- All mathematical expressions must be enclosed in LaTeX delimiters:
  - Use `$...$` for inline math
  - Use `$$...$$` for display math
- If the student insists on a direct solution, gently remind them that the goal is to practice problem-solving and reasoning.
        """

    return system_prompt


def get_gpt_response(user_input):
    """
    This function sends the conversation to the GPT API and appends the assistant response to chat history.
    We've added a spinner so the user sees a 'waiting' indicator while GPT is responding.
    """
    system_prompt = get_system_prompt()
    conversation_history_formatted = [{"role": "system", "content": system_prompt}]
    conversation_history_formatted += [
        {"role": role, "content": content}
        for role, content in st.session_state.chat_history
    ]

    try:
        with st.spinner("EeeBee is thinking..."):
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",  # Correct model name
                messages=conversation_history_formatted,
                max_tokens=2000
            ).choices[0].message['content'].strip()

        st.session_state.chat_history.append(("assistant", gpt_response))
    except Exception as e:
        st.error(f"Error in GPT response generation: {e}")


# ================= CONCEPT CONTENT LOADING FUNCTION =================
def load_concept_content():
    selected_concept_id = st.session_state.selected_concept_id
    selected_concept_name = next(
        (concept['ConceptText'] for concept in st.session_state.auth_data['ConceptList'] if concept['ConceptID'] == selected_concept_id),
        "Unknown Concept"
    )

    # We'll also pass the student's class/grade level (branch_name) to the prompt
    branch_name = st.session_state.branch_name
    prompt = (
        f"The student is in {branch_name}. Provide a concise and educational description of the concept "
        f"'{selected_concept_name}', pitched at the level of {branch_name} students, to help them understand it better. "
        f"Ensure that all mathematical expressions are enclosed within LaTeX delimiters (`$...$` for inline and `$$...$$` for display)."
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
        with st.spinner("🔄 Fetching concept content..."):
            content_response = requests.post(API_CONTENT_URL, json=content_payload, headers=headers)
            content_response.raise_for_status()
            content_data = content_response.json()

            # Generate concept description from GPT
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",  # Correct model name
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500
            ).choices[0].message['content'].strip()

            # Minor replacements if needed
            gpt_response = gpt_response.replace("This concept", selected_concept_name).replace("this concept", selected_concept_name)
            gpt_response += "\n\nYou can check the resources below for more information."
            st.session_state.generated_description = gpt_response

            # Fetch resources
            resources = fetch_resources_by_concept_id(
                topic_id=st.session_state.topic_id,
                concept_id=selected_concept_id
            )

            display_resources(content_data)

    except requests.exceptions.RequestException as req_err:
        st.error(f"Error fetching content: {req_err}")
    except Exception as e:
        st.error(f"Error generating concept description: {e}")


# ================= RESOURCES DISPLAY FUNCTION =================
def display_resources(content_data):
    branch_name = st.session_state.branch_name
    with st.expander("📚 Resources", expanded=True):
        concept_description = st.session_state.get("generated_description", "No description available.")
        st.markdown(f"### Concept Description for {branch_name}\n{concept_description}\n")
        
        # Display Videos
        if content_data.get("Video_List"):
            st.markdown("#### 🎥 Videos")
            for video in content_data["Video_List"]:
                video_url = video.get('LectureLink')
                video_title = video.get('LectureTitle', 'No Title')
                if video_url:
                    st.markdown(f"- [Video: {video_title}]({video_url})")
                else:
                    st.write(f"- Video: {video_title} (Link not available)")

        # Display Notes
        if content_data.get("Notes_List"):
            st.markdown("#### 📄 Notes")
            for note in content_data["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                note_title = note.get("NotesTitle", "No Title")
                if note_url:
                    st.markdown(f"- [Notes: {note_title}]({note_url})")
                else:
                    st.write(f"- Notes: {note_title} (Link not available)")

        # Display Exercises
        if content_data.get("Exercise_List"):
            st.markdown("#### 📝 Exercises")
            for exercise in content_data["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                exercise_title = exercise.get("ExerciseTitle", "No Title")
                if exercise_url:
                    st.markdown(f"- [Exercise: {exercise_title}]({exercise_url})")
                else:
                    st.write(f"- Exercise: {exercise_title} (Link not available)")


# ================= TEACHER DASHBOARD FUNCTIONS =================
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

        # Create an Altair chart
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

        # Red rule for total students
        rule = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_rule(color='red', strokeDash=[4, 4]).encode(
            y='y:Q'
        )
        # Label for the rule
        text = alt.Chart(pd.DataFrame({'y': [total_students]})).mark_text(
            align='left', dx=5, dy=-5, color='red'
        ).encode(
            y='y:Q',
            text=alt.value(f'Total Students: {total_students}')
        )

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
            index=3  # Default to L4
        )

        concept_list = {wc["ConceptText"]: wc["ConceptID"] for wc in st.session_state.teacher_weak_concepts}
        chosen_concept_text = st.radio("Select a Concept to Generate Exam Questions:", list(concept_list.keys()))

        if chosen_concept_text:
            chosen_concept_id = concept_list[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions"):
                branch_name = st.session_state.branch_name

                # Parse out the short code (L1, L2, etc.) from the selectbox choice
                bloom_short = bloom_level.split()[0]  # E.g., "L4"

                # Build the prompt
                prompt = (
                    f"You are an educational AI assistant helping a teacher. The teacher wants to create "
                    f"exam questions for the concept '{chosen_concept_text}'.\n"
                    f"The teacher is teaching students in {branch_name}, following the NCERT curriculum.\n"
                    f"Generate a set of 20 challenging and thought-provoking exam questions related to this concept.\n"
                    f"Generated questions should be aligned with NEP 2020 and NCF guidelines.\n"
                    f"Vary in difficulty.\n"
                    f"Encourage critical thinking.\n"
                    f"Be clearly formatted and numbered.\n\n"
                    f"Do not provide the answers, only the questions.\n"
                    f"Ensure that all mathematical expressions are enclosed within LaTeX delimiters (`$...$` for inline "
                    f"and `$$...$$` for display).\n"
                    f"Focus on **Bloom's Taxonomy Level {bloom_short}**.\n"
                    f"Label each question clearly with **({bloom_short})** at the end of the question.\n"
                )

                with st.spinner("Generating exam questions... Please wait."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",  # Correct model name
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=5000
                        )
                        questions = response.choices[0].message['content'].strip()
                        st.session_state.exam_questions = questions
                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

    if st.session_state.exam_questions:
        branch_name = st.session_state.branch_name
        st.markdown(f"### 📝 Generated Exam Questions for {branch_name}")
        st.markdown(st.session_state.exam_questions)

        pdf_bytes = generate_exam_questions_pdf(
            st.session_state.exam_questions,
            st.session_state.selected_teacher_concept_text,
            st.session_state.auth_data['UserInfo'][0]['FullName'],
            resources=None  # You can choose to include resources here if needed
        )
        st.download_button(
            label="📥 Download Exam Questions as PDF",
            data=pdf_bytes,
            file_name=f"Exam_Questions_{st.session_state.selected_teacher_concept_text}.pdf",
            mime="application/pdf"
        )


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
        x=alt.X('Count:Q'),
        y=alt.Y('ConceptText:N', sort='-x', title='Concepts'),
        color=alt.Color('Category:N', legend=alt.Legend(title="Category")),
        tooltip=['ConceptText:N', 'Category:N', 'Count:Q']
    ).properties(
        title='Attended vs Cleared per Concept (Horizontal View)',
        width=600
    )
    st.altair_chart(horizontal_bar, use_container_width=True)


# ================= LOGIN SCREEN FUNCTION =================
# ================= LOGIN SCREEN FUNCTION =================
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

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
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
                    st.session_state.branch_name = auth_data.get("BranchName", "their class")  # Retrieved from API
                    # If student, populate weak concepts
                    if not st.session_state.is_teacher:
                        st.session_state.student_weak_concepts = auth_data.get("WeakConceptList", [])
                    st.experimental_rerun()
                else:
                    st.error("🚫 Authentication failed. Please check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the authentication API: {e}")



# ================= MAIN SCREEN FUNCTION (POST-LOGIN) =================
def main_screen():
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']
    branch_name = st.session_state.branch_name

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;"> EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.is_teacher:
        # Teacher Mode
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
        with tabs[0]:
            display_chat()  # Utilize the streamlined chat function

        with tabs[1]:
            st.subheader("Teacher Dashboard")
            teacher_dashboard()

    else:
        # Student Mode
        if st.session_state.is_english_mode:
            # English Student: only Chat
            tab1 = st.tabs(["💬 Chat"])[0]
            with tab1:
                display_chat()
        else:
            # Non-English Student: Chat + Learning Path + Concepts
            tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Learning Path", "📚 Concepts"])
            with tab1:
                display_chat()

            with tab2:
                weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])

                if not weak_concepts:
                    st.warning("No weak concepts found.")
                else:
                    for idx, concept in enumerate(weak_concepts):
                        concept_text = concept.get("ConceptText", f"Concept {idx+1}")
                        concept_id = concept.get("ConceptID", f"id_{idx+1}")

                        st.markdown(f"#### **Weak Concept {idx+1}:** {concept_text}")

                        # Generate Learning Path Button
                        button_key = f"generate_lp_{concept_id}"
                        if st.button("🧠 Generate Learning Path", key=button_key):
                            if concept_id not in st.session_state.student_learning_paths:
                                with st.spinner(f"Generating learning path for {concept_text}..."):
                                    learning_path = generate_learning_path(concept_text)
                                    if learning_path:
                                        # Fetch resources for the concept
                                        resources = fetch_resources_by_concept_id(
                                            topic_id=st.session_state.topic_id,
                                            concept_id=concept_id
                                        )

                                        st.session_state.student_learning_paths[concept_id] = {
                                            "concept_text": concept_text,
                                            "learning_path": learning_path,
                                            "resources": resources
                                        }
                                        st.success(f"Learning path generated for {concept_text}!")
                                    else:
                                        st.error(f"Failed to generate learning path for {concept_text}.")
                            else:
                                st.info(f"Learning path for {concept_text} is already generated.")

                        # Display the learning path if it exists
                        if concept_id in st.session_state.student_learning_paths:
                            lp_data = st.session_state.student_learning_paths[concept_id]
                            display_learning_path(
                                lp_data["concept_text"],
                                lp_data["learning_path"],
                                resources=lp_data.get("resources")
                            )

            with tab3:
                concept_list = st.session_state.auth_data.get('ConceptList', [])
                concept_options = {concept['ConceptText']: concept['ConceptID'] for concept in concept_list}
                selected_concept = st.selectbox("Select a Concept to View Details:", list(concept_options.keys()))
                if selected_concept:
                    selected_concept_id = concept_options[selected_concept]
                    st.session_state.selected_concept_id = selected_concept_id
                    load_concept_content()


# ================= CHAT-RELATED FUNCTIONS =================
def display_chat():
    st.subheader("💬 Chat with your EeeBee AI buddy")
    add_initial_greeting()
    
    for role, message in st.session_state.chat_history:
        if role == "assistant":
            st.chat_message("assistant").write(message)
        else:
            st.chat_message("user").write(message)
    
    user_input = st.chat_input("Enter your question about the topic")
    if user_input:
        handle_user_input(user_input)


# ================= LOGIN SCREEN FUNCTION =================
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

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
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
                    st.session_state.branch_name = st.text_input("📚 Branch Name", key="branch_name_input") or "their class"
                    # If student, populate weak concepts
                    if not st.session_state.is_teacher:
                        st.session_state.student_weak_concepts = auth_data.get("WeakConceptList", [])
                    st.experimental_rerun()
                else:
                    st.error("🚫 Authentication failed. Please check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the authentication API: {e}")


# ================= MAIN APP LOGIC =================
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
