import warnings
import os
import re
import io
import json
import streamlit as st
import openai
import requests
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

from matplotlib import rcParams
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

# ------------------------------------------------------------------------------------
# 1. INITIAL SETUP, STREAMLIT CONFIG, IGNORE WARNINGS, LOAD OPENAI KEY, SET PAGE CONFIG
# ------------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Load OpenAI API Key from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("API key for OpenAI not found in secrets.")

openai.api_key = OPENAI_API_KEY

# Streamlit page configuration
st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Hide default Streamlit elements
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# 2. GLOBAL CONSTANTS / API URLS
# ------------------------------------------------------------------------------------
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"

# NEW endpoint that returns both concepts and students in one request
API_TEACHER_WEAK_CONCEPTS_AND_STUDENTS = (
    "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students"
)

# ------------------------------------------------------------------------------------
# 3. SESSION STATE (INITIALIZATION)
# ------------------------------------------------------------------------------------
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
if "teacher_data" not in st.session_state:
    st.session_state.teacher_data = {"Concepts": [], "Students": []}
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

# ------------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------------

# 4.1 LATEX TO IMAGE
def latex_to_image(latex_code, dpi=300):
    """
    Converts LaTeX code to a PNG image and returns it as a BytesIO object.
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

# 4.2 FETCHING RESOURCES FOR A GIVEN CONCEPT
def get_resources_for_concept(concept_text, concept_list, topic_id):
    """
    Fetch resources for a given concept text.
    Returns the resources data if found, None otherwise.
    """
    def clean_text(text):
        return text.lower().strip().replace(" ", "")

    clean_concept = clean_text(concept_text)
    matching_concept = next(
        (c for c in concept_list if clean_text(c['ConceptText']) == clean_concept),
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
            print(f"Error fetching resources: {e}")
            return None
    return None

# 4.3 FORMAT THE RESOURCES IN CHAT-FRIENDLY MESSAGE
def format_resources_message(resources):
    message = "Here are the available resources for this concept:\n\n"

    if resources.get("Video_List"):
        message += "**🎥 Video Lectures:**\n"
        for video in resources["Video_List"]:
            video_url = f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}"
            message += f"- [{video.get('LectureTitle', 'Video Lecture')}]({video_url})\n"
        message += "\n"

    if resources.get("Notes_List"):
        message += "**📄 Study Notes:**\n"
        for note in resources["Notes_List"]:
            note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
            message += f"- [{note.get('NotesTitle', 'Study Notes')}]({note_url})\n"
        message += "\n"

    if resources.get("Exercise_List"):
        message += "**📝 Practice Exercises:**\n"
        for exercise in resources["Exercise_List"]:
            exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
            message += f"- [{exercise.get('ExerciseTitle', 'Practice Exercise')}]({exercise_url})\n"

    return message

# 4.4 GENERATE EXAM QUESTIONS PDF
def generate_exam_questions_pdf(questions, concept_text, user_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
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

    # Title and subtitle
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
        # First line as section title
        story.append(Paragraph(lines[0], section_title_style))
        story.append(Spacer(1, 8))

        # Add questions as a numbered list
        question_items = []
        for line in lines[1:]:
            # Detect LaTeX
            latex_matches = re.finditer(r'\$\$(.*?)\$\$|\$(.*?)\$', line)
            if latex_matches:
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

                    # Add text before LaTeX
                    pre_text = line[last_index:match.start()]
                    if pre_text:
                        question_items.append(ListItem(Paragraph(pre_text, question_style)))

                    img_buffer = latex_to_image(latex)
                    if img_buffer:
                        if display_math:
                            img = RLImage(img_buffer, width=4 * inch, height=1 * inch)
                        else:
                            img = RLImage(img_buffer, width=2 * inch, height=0.5 * inch)
                        question_items.append(ListItem(img))

                    last_index = match.end()

                # Remaining text
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

# 4.5 GENERATE LEARNING PATH PDF
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

    # Break the learning path into sections
    sections = re.split(r'\n\n', learning_path.strip())
    for section in sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            continue
        # First line as section header
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

                    pre_text = line[last_index:match.start()]
                    if pre_text:
                        story.append(Paragraph(pre_text, content_style))

                    img_buffer = latex_to_image(latex)
                    if img_buffer:
                        if display_math:
                            img = RLImage(img_buffer, width=4 * inch, height=1 * inch)
                        else:
                            img = RLImage(img_buffer, width=2 * inch, height=0.5 * inch)
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

# 4.6 GENERATE LEARNING PATH CONTENT (GPT)
def generate_learning_path(concept_text):
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    prompt = (
        f"You are a highly experienced educational AI assistant specializing in the NCERT curriculum. "
        f"A student in {branch_name} is struggling with the weak concept: '{concept_text}'. "
        f"Please create a structured, step-by-step learning path tailored to {branch_name} students, ensuring clarity, engagement, and curriculum alignment. "
        f"Your plan should include:\n\n"
        f"1. **Introduction to the Concept**\n"
        f"2. **Step-by-Step Learning**\n"
        f"3. **Engagement**\n"
        f"4. **Real-World Applications**\n"
        f"5. **Practice Problems**\n\n"
        f"All mathematical expressions enclosed in LaTeX delimiters ($...$ or $$...$$)."
    )
    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",  # or whichever GPT model you have
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500
        ).choices[0].message['content'].strip()
        return gpt_response
    except Exception as e:
        st.error(f"Error generating learning path: {e}")
        return None

# 4.7 DISPLAY LEARNING PATH + RESOURCES
def display_learning_path_with_resources(concept_text, learning_path, concept_list, topic_id):
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    with st.expander(f"📚 Learning Path for {concept_text} ({branch_name} level)", expanded=False):
        st.markdown(learning_path, unsafe_allow_html=True)

        # Attempt to fetch resources
        resources = get_resources_for_concept(
            concept_text,
            concept_list,
            topic_id
        )
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

        # Download PDF
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

# 4.8 DISPLAY CONCEPT CONTENT (if needed)
def display_resources(content_data):
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')
    with st.expander("📚 Resources", expanded=True):
        concept_description = st.session_state.get("generated_description", "No description available.")
        st.markdown(f"### Concept Description ({branch_name})\n{concept_description}\n")

        if content_data.get("Video_List"):
            for video in content_data["Video_List"]:
                video_url = video.get("LectureLink", f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}")
                st.write(f"- [Video 🎥]({video_url})")
        if content_data.get("Notes_List"):
            for note in content_data["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                st.write(f"- [Notes 📄]({note_url})")
        if content_data.get("Exercise_List"):
            for exercise in content_data["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                st.write(f"- [Exercise 📝]({exercise_url})")

# ------------------------------------------------------------------------------------
# 5. TEACHER DASHBOARD
# ------------------------------------------------------------------------------------
def teacher_dashboard():
    """
    Fetch both Concepts and Students data from the new API endpoint,
    display chart(s), table(s), and allow generating exam questions & suggestions.
    """
    batches = st.session_state.auth_data.get("BatchList", [])
    if not batches:
        st.warning("No batches found for the teacher.")
        return

    # 5.1 Batch selection
    batch_options = {b['BatchName']: b for b in batches}
    selected_batch_name = st.selectbox("Select a Batch:", list(batch_options.keys()))
    selected_batch = batch_options.get(selected_batch_name)
    selected_batch_id = selected_batch["BatchID"]
    total_students = selected_batch.get("StudentCount", 0)

    # 5.2 API call for teacher data if batch changes
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
        with st.spinner("🔄 Fetching teacher data (concepts & students)..."):
            try:
                response = requests.post(
                    API_TEACHER_WEAK_CONCEPTS_AND_STUDENTS,
                    json=params,
                    headers=headers
                )
                response.raise_for_status()
                teacher_data = response.json()  # { "Concepts": [...], "Students": [...] }
                st.session_state.teacher_data = teacher_data
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.session_state.teacher_data = {"Concepts": [], "Students": []}

    # 5.3 Load local copy
    teacher_data = st.session_state.teacher_data
    concepts = teacher_data.get("Concepts", [])
    students = teacher_data.get("Students", [])

    # ---- CONCEPT-LEVEL OVERVIEW ----
    st.header("Concept-Level Overview")
    if concepts:
        df_concepts = pd.DataFrame(concepts)

        # Melt data for Attended vs. Cleared chart
        df_melted = df_concepts.melt(
            id_vars=["ConceptText"],
            value_vars=["AttendedStudentCount", "ClearedStudentCount"],
            var_name="Category",
            value_name="Count"
        )
        df_melted["Category"] = df_melted["Category"].replace({
            "AttendedStudentCount": "Attended",
            "ClearedStudentCount": "Cleared"
        })

        # Altair bar chart
        concept_chart = (
            alt.Chart(df_melted)
            .mark_bar()
            .encode(
                x=alt.X("ConceptText:N", sort=None, title="Concept"),
                y=alt.Y("Count:Q"),
                color="Category:N",
                tooltip=["ConceptText:N", "Category:N", "Count:Q"]
            )
            .properties(width=700, height=400, title="Attended vs. Cleared per Concept")
            .interactive()
        )
        st.altair_chart(concept_chart, use_container_width=True)

        # Summaries in text form
        st.write("**Concept Stats**:")
        for c in concepts:
            st.write(
                f"• {c['ConceptText']}: "
                f"Attended = {c['AttendedStudentCount']}, "
                f"Cleared = {c['ClearedStudentCount']}"
            )
    else:
        st.info("No concept data available for this batch/topic.")

    # ---- STUDENT-LEVEL OVERVIEW ----
    st.header("Student-Level Overview")
    if students:
        df_students = pd.DataFrame(students)
        st.dataframe(
            df_students[
                ["FullName", "TotalConceptCount", "WeakConceptCount", "ClearedConceptCount"]
            ]
        )

        # Quick chart (Weak vs Cleared) per student
        df_students_melted = df_students.melt(
            id_vars=["FullName"],
            value_vars=["WeakConceptCount", "ClearedConceptCount"],
            var_name="Status",
            value_name="Count"
        )
        student_chart = (
            alt.Chart(df_students_melted)
            .mark_bar()
            .encode(
                x=alt.X("FullName:N", sort=None, title="Student"),
                y=alt.Y("Count:Q"),
                color="Status:N",
                tooltip=["FullName:N", "Status:N", "Count:Q"]
            )
            .properties(width=700, height=400, title="Weak vs. Cleared Concepts per Student")
            .interactive()
        )
        st.altair_chart(student_chart, use_container_width=True)
    else:
        st.info("No student data available for this batch/topic.")

    # ---- TEACHER ACTIONS: EXAM QUESTIONS & INDIVIDUAL SUGGESTIONS ----
    st.header("Additional Teacher Tools")

    # 5.3.1 EXAM QUESTIONS GENERATION
    if concepts:
        bloom_level = st.radio(
            "Select Bloom's Taxonomy Level for the Questions",
            ["L1 (Remember)", "L2 (Understand)", "L3 (Apply)", "L4 (Analyze)", "L5 (Evaluate)"],
            index=3
        )

        # Map concept_text -> concept_id
        concept_map = {c["ConceptText"]: c["ConceptID"] for c in concepts}
        chosen_concept_text = st.selectbox(
            "Select a Concept to Generate Exam Questions:",
            list(concept_map.keys())
        )

        if chosen_concept_text:
            chosen_concept_id = concept_map[chosen_concept_text]
            st.session_state.selected_teacher_concept_id = chosen_concept_id
            st.session_state.selected_teacher_concept_text = chosen_concept_text

            if st.button("Generate Exam Questions"):
                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                bloom_short = bloom_level.split()[0]  # e.g., "L4"

                # Prompt
                prompt = (
                    f"You are an educational AI assistant helping a teacher with {branch_name} students. "
                    f"The teacher wants to create **20** exam questions for the concept '{chosen_concept_text}'.\n"
                    f"Students follow the NCERT curriculum. Questions should vary in difficulty, align with NEP 2020, "
                    f"and encourage critical thinking. Provide them as a numbered list, no solutions. "
                    f"Use LaTeX for math ($...$ or $$...$$), focusing on Bloom's Taxonomy Level {bloom_short}.\n"
                    f"Label each question with **({bloom_short})** at the end."
                )

                with st.spinner("Generating exam questions..."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",  # or whichever GPT model
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=2000
                        )
                        questions = response.choices[0].message['content'].strip()
                        st.session_state.exam_questions = questions
                    except Exception as e:
                        st.error(f"Error generating exam questions: {e}")

    if st.session_state.exam_questions:
        st.subheader("📝 Generated Exam Questions")
        st.markdown(st.session_state.exam_questions)

        pdf_bytes = generate_exam_questions_pdf(
            st.session_state.exam_questions,
            st.session_state.selected_teacher_concept_text,
            st.session_state.auth_data['UserInfo'][0]['FullName']
        )
        st.download_button(
            label="📥 Download Exam Questions as PDF",
            data=pdf_bytes,
            file_name=f"Exam_Questions_{st.session_state.selected_teacher_concept_text}.pdf",
            mime="application/pdf"
        )

    # 5.3.2 INDIVIDUALIZED TEACHING SUGGESTIONS
    st.subheader("Get Individualized Teaching Suggestions")
    if students:
        student_names = {s["FullName"]: s for s in students}
        chosen_student_name = st.selectbox(
            "Select a Student to get teaching suggestions:",
            list(student_names.keys())
        )

        if chosen_student_name:
            student_info = student_names[chosen_student_name]
            if st.button("Ask EeeBee for Suggestions"):
                suggestions = generate_student_suggestions(student_info)
                if suggestions:
                    st.session_state[f"suggestions_{student_info['UserID']}"] = suggestions
                    st.success("Suggestions generated.")

            # Display suggestions if they exist
            existing_suggestions = st.session_state.get(f"suggestions_{student_info['UserID']}", "")
            if existing_suggestions:
                with st.expander(f"EeeBee's Suggestions for {chosen_student_name}", expanded=True):
                    st.markdown(existing_suggestions)

# GPT function for suggestions
def generate_student_suggestions(student_info):
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data.get('TopicName', 'the subject')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    prompt = (
        f"You are an educational AI (EeeBee) specialized in {topic_name}.\n\n"
        f"The teacher, {user_name}, wants individualized suggestions for the student:\n"
        f"• Name: {student_info['FullName']}\n"
        f"• Total Concepts: {student_info['TotalConceptCount']}\n"
        f"• Weak Concepts: {student_info['WeakConceptCount']}\n"
        f"• Cleared Concepts: {student_info['ClearedConceptCount']}\n\n"
        f"Assume the student is in {branch_name} (NCERT). Please provide:\n"
        f"1) Strategies to strengthen any weak concepts.\n"
        f"2) Practice/homework tasks at their level.\n"
        f"3) Motivation/engagement ideas.\n\n"
        f"All suggestions must be actionable, aligned with {branch_name}, and avoid providing direct answers.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        suggestions = response.choices[0].message["content"].strip()
        return suggestions
    except Exception as e:
        st.error(f"Error generating suggestions: {e}")
        return None

# ------------------------------------------------------------------------------------
# 6. LOGIN SCREEN
# ------------------------------------------------------------------------------------
def login_screen():
    # Display a login graphic (optional)
    try:
        image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image_url, width=160)
        st.markdown(
            """
            <style>
            @media only screen and (max-width: 600px) {
                .title { font-size: 2.5em; margin-top: 20px; text-align: center; }
            }
            @media only screen and (min-width: 601px) {
                .title { font-size: 4em; font-weight: bold; margin-top: 90px; margin-left: -125px; text-align: left; }
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        with col2:
            st.markdown('<div class="title">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading login image: {e}")

    st.markdown(
        '<h3 style="font-size: 1.5em;">🦾 Please enter your credentials to access your AI Buddy!</h3>',
        unsafe_allow_html=True
    )

    user_type = st.radio("Select User Type", ["Student", "Teacher"])
    user_type_value = 2 if user_type == "Teacher" else None  # 2 for Teacher, None for Student

    org_code = st.text_input("🏫 School Code", key="org_code")
    login_id = st.text_input("👤 Login ID", key="login_id")
    password = st.text_input("🔒 Password", type="password", key="password")

    # Query params for E (English) or T (Non-English)
    query_params = st.experimental_get_query_params()
    E_params = query_params.get("E", [None])
    T_params = query_params.get("T", [None])

    E_value = E_params[0]
    T_value = T_params[0]

    api_url = None
    topic_id = None

    # Decide which API to use (English vs Non-English)
    if E_value is not None and T_value is not None:
        st.warning("Please provide either E for English OR T for Non-English, not both.")
    elif E_value is not None and T_value is None:
        st.session_state.is_english_mode = True
        api_url = API_AUTH_URL_ENGLISH
        topic_id = E_value
    elif E_value is None and T_value is not None:
        st.session_state.is_english_mode = False
        api_url = API_AUTH_URL_MATH_SCIENCE
        topic_id = T_value
    else:
        st.warning("Please provide E for English mode or T for Non-English mode in the URL.")

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if topic_id is None or api_url is None:
            st.warning("Please ensure correct E or T parameter is provided.")
            return

        auth_payload = {
            'OrgCode': org_code,
            'TopicID': int(topic_id),
            'LoginID': login_id,
            'Password': password
        }
        if user_type_value:
            auth_payload['UserType'] = user_type_value  # Teacher only

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

                    # If student, store weak concepts
                    if not st.session_state.is_teacher:
                        st.session_state.student_weak_concepts = auth_data.get("WeakConceptList", [])
                    st.rerun()
                else:
                    st.error("🚫 Authentication failed. Check your credentials.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the authentication API: {e}")

# ------------------------------------------------------------------------------------
# 7. CHAT-RELATED FUNCTIONS (STUDENT/TEACHER)
# ------------------------------------------------------------------------------------
def add_initial_greeting():
    """
    Adds a greeting from EeeBee if the chat is empty.
    """
    if len(st.session_state.chat_history) == 0 and st.session_state.auth_data:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data['TopicName']

        # Build concept list text
        concept_list = st.session_state.auth_data.get('ConceptList', [])
        concept_options = "\n\n**📚 Available Concepts:**\n"
        for concept in concept_list:
            concept_options += f"- {concept['ConceptText']}\n"

        # Build weak concept text
        weak_concepts = st.session_state.auth_data.get('WeakConceptList', [])
        weak_concepts_text = ""
        if weak_concepts:
            weak_concepts_text = "\n\n**🎯 Your Current Learning Gaps:**\n"
            for concept in weak_concepts:
                weak_concepts_text += f"- {concept['ConceptText']}\n"

        st.session_state.available_concepts = {
            concept['ConceptText']: concept['ConceptID']
            for concept in concept_list
        }

        greeting_message = (
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy, here to help with **{topic_name}**.\n\n"
            f"You can:\n"
            f"1. Ask me questions about any concept\n"
            f"2. Request learning resources (videos, notes, exercises)\n"
            f"3. Ask for help understanding specific topics\n"
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
    topic_name = st.session_state.auth_data.get('TopicName', 'Unknown Topic')
    branch_name = st.session_state.auth_data.get('BranchName', 'their class')

    if st.session_state.is_teacher:
        system_prompt = f"""
You are EeeBee, a highly knowledgeable educational AI assistant from iEdubull, specialized in {topic_name}.

Teacher Mode:
- The user is a teacher instructing {branch_name} students under the NCERT curriculum.
- Provide detailed suggestions on how to explain concepts and design assessments.
- Offer insights into common student difficulties.
- Encourage step-by-step reasoning, not direct solutions.
- Keep math in LaTeX: $...$ for inline, $$...$$ for display.
- Focus on aligning with NCERT guidelines.
"""
    else:
        weak_concepts = [c['ConceptText'] for c in st.session_state.student_weak_concepts]
        weak_concepts_text = ", ".join(weak_concepts) if weak_concepts else "none"
        system_prompt = f"""
You are EeeBee, a knowledgeable educational AI assistant from iEdubull, specialized in {topic_name}.

Student Mode:
- The student is in {branch_name}, NCERT curriculum.
- Weak concepts include: [{weak_concepts_text}].
- Provide guidance, practice suggestions, step-by-step reasoning.
- Avoid giving direct final solutions; use prompting/hints.
- Keep math in LaTeX: $...$ or $$...$$.
- Provide resources from Edubull only if asked.
"""
    return system_prompt

def get_gpt_response(user_input):
    system_prompt = get_system_prompt()
    conversation_history_formatted = [{"role": "system", "content": system_prompt}] + [
        {"role": role, "content": content}
        for role, content in st.session_state.chat_history
    ]

    try:
        with st.spinner("EeeBee is thinking..."):
            concept_list = st.session_state.auth_data.get('ConceptList', [])
            mentioned_concept = None

            # Check if user mentions a known concept
            for concept in concept_list:
                if concept['ConceptText'].lower() in user_input.lower():
                    mentioned_concept = concept['ConceptText']
                    break

            # GPT response
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=conversation_history_formatted,
                max_tokens=2000
            ).choices[0].message['content'].strip()

            st.session_state.chat_history.append(("assistant", gpt_response))

            # If user asked for resources about a concept
            if mentioned_concept and any(
                word in user_input.lower() for word in ['resource', 'material', 'video', 'note', 'exercise']
            ):
                resources = get_resources_for_concept(
                    mentioned_concept, concept_list, st.session_state.topic_id
                )
                if resources:
                    resource_message = format_resources_message(resources)
                    st.session_state.chat_history.append(("assistant", resource_message))

    except Exception as e:
        st.error(f"Error in GPT response: {e}")

# ------------------------------------------------------------------------------------
# 8. MAIN SCREEN (POST-LOGIN)
# ------------------------------------------------------------------------------------
def main_screen():
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, 
        <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;">
        EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    # TEACHER MODE
    if st.session_state.is_teacher:
        tabs = st.tabs(["💬 Chat", "📊 Teacher Dashboard"])
        # Chat tab
        with tabs[0]:
            st.subheader("Chat with your EeeBee AI buddy")
            add_initial_greeting()

            chat_container = st.container()
            with chat_container:
                chat_history_html = """
                <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                """
                for role, message in st.session_state.chat_history:
                    if role == "assistant":
                        chat_history_html += (
                            f"<div style='text-align: left; color: #000; background-color: #e0e7ff; "
                            f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                            f"<b>EeeBee:</b> {message}</div>"
                        )
                    else:
                        chat_history_html += (
                            f"<div style='text-align: left; color: #fff; background-color: #2563eb; "
                            f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                            f"<b>{user_name}:</b> {message}</div>"
                        )
                chat_history_html += "</div>"
                st.markdown(chat_history_html, unsafe_allow_html=True)

            user_input = st.chat_input("Enter your question about the topic")
            if user_input:
                handle_user_input(user_input)

        # Teacher dashboard tab
        with tabs[1]:
            st.subheader("Teacher Dashboard (Concepts & Students)")
            teacher_dashboard()

    # STUDENT MODE
    else:
        if st.session_state.is_english_mode:
            # English Students: only Chat
            tab = st.tabs(["💬 Chat"])[0]
            with tab:
                st.subheader("Chat with your EeeBee AI buddy")
                add_initial_greeting()

                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += (
                                f"<div style='text-align: left; color: #000; background-color: #e0e7ff; "
                                f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                                f"<b>EeeBee:</b> {message}</div>"
                            )
                        else:
                            chat_history_html += (
                                f"<div style='text-align: left; color: #fff; background-color: #2563eb; "
                                f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                                f"<b>{user_name}:</b> {message}</div>"
                            )
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)

                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)
        else:
            # Non-English Students: Chat + Learning Path tab
            tab_chat, tab_learning = st.tabs(["💬 Chat", "🧠 Learning Path"])

            with tab_chat:
                st.subheader("Chat with your EeeBee AI buddy")
                add_initial_greeting()

                chat_container = st.container()
                with chat_container:
                    chat_history_html = """
                    <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
                    """
                    for role, message in st.session_state.chat_history:
                        if role == "assistant":
                            chat_history_html += (
                                f"<div style='text-align: left; color: #000; background-color: #e0e7ff; "
                                f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                                f"<b>EeeBee:</b> {message}</div>"
                            )
                        else:
                            chat_history_html += (
                                f"<div style='text-align: left; color: #fff; background-color: #2563eb; "
                                f"padding: 8px; border-radius: 8px; margin-bottom: 5px;'>"
                                f"<b>{user_name}:</b> {message}</div>"
                            )
                    chat_history_html += "</div>"
                    st.markdown(chat_history_html, unsafe_allow_html=True)

                user_input = st.chat_input("Enter your question about the topic")
                if user_input:
                    handle_user_input(user_input)

            with tab_learning:
                weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
                concept_list = st.session_state.auth_data.get('ConceptList', [])

                if not weak_concepts:
                    st.warning("No weak concepts found.")
                else:
                    for idx, concept in enumerate(weak_concepts):
                        concept_text = concept.get("ConceptText", f"Concept {idx+1}")
                        concept_id = concept.get("ConceptID", f"id_{idx+1}")

                        st.markdown(f"#### Weak Concept {idx+1}: {concept_text}")

                        # Button to generate learning path
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

                        # If learning path is generated, display it + resources
                        if concept_id in st.session_state.student_learning_paths:
                            lp_data = st.session_state.student_learning_paths[concept_id]
                            display_learning_path_with_resources(
                                lp_data["concept_text"],
                                lp_data["learning_path"],
                                concept_list,
                                st.session_state.topic_id
                            )

# ------------------------------------------------------------------------------------
# 9. MAIN APP ENTRY POINT
# ------------------------------------------------------------------------------------
def main():
    if st.session_state.is_authenticated:
        main_screen()
    else:
        login_screen()

if __name__ == "__main__":
    main()
