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
# 1. INITIAL SETUP AND STREAMLIT CONFIG
# ------------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# Load OpenAI API Key from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found in secrets.")

openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide"
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
# 2. GLOBAL API URLs
# ------------------------------------------------------------------------------------
API_AUTH_URL_ENGLISH = "https://webapi.edubull.com/api/EnglishLab/Auth_with_topic_for_chatbot"
API_AUTH_URL_MATH_SCIENCE = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"

# New endpoint returning both Concepts & Students
API_TEACHER_WEAK_CONCEPTS_AND_STUDENTS = (
    "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students"
)

# ------------------------------------------------------------------------------------
# 3. SESSION STATE
# ------------------------------------------------------------------------------------
if "auth_data" not in st.session_state:
    st.session_state.auth_data = None
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "is_teacher" not in st.session_state:
    st.session_state.is_teacher = False
if "topic_id" not in st.session_state:
    st.session_state.topic_id = None

# For the chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# For teacher data & selection
if "teacher_data" not in st.session_state:
    st.session_state.teacher_data = {"Concepts": [], "Students": []}
if "selected_batch_id" not in st.session_state:
    st.session_state.selected_batch_id = None
if "selected_student_id" not in st.session_state:
    st.session_state.selected_student_id = None
if "selected_student_info" not in st.session_state:
    st.session_state.selected_student_info = None

# For the student's perspective (if user logs in as a student)
if "student_weak_concepts" not in st.session_state:
    st.session_state.student_weak_concepts = []
if "is_english_mode" not in st.session_state:
    st.session_state.is_english_mode = False

# For exam questions & learning paths, etc.
if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = ""
if "student_learning_paths" not in st.session_state:
    st.session_state.student_learning_paths = {}

# ------------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------------
def latex_to_image(latex_code, dpi=300):
    """
    Convert LaTeX code to a PNG image in memory.
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
        st.error(f"Error converting LaTeX: {e}")
        return None

def format_resources_message(resources):
    """
    Example helper to format concept resources for the chat.
    """
    message = "Here are the available resources:\n\n"
    if resources.get("Video_List"):
        message += "**🎥 Video Lectures**\n"
        for vid in resources["Video_List"]:
            url = f"https://www.edubull.com/courses/videos/{vid.get('LectureID','')}"
            message += f"- [{vid.get('LectureTitle','Video')}]({url})\n"
    if resources.get("Notes_List"):
        message += "\n**📄 Study Notes**\n"
        for note in resources["Notes_List"]:
            url = f"{note.get('FolderName','')}{note.get('PDFFileName','')}"
            message += f"- [{note.get('NotesTitle','Note')}]({url})\n"
    if resources.get("Exercise_List"):
        message += "\n**📝 Practice Exercises**\n"
        for ex in resources["Exercise_List"]:
            url = f"{ex.get('FolderName','')}{ex.get('ExerciseFileName','')}"
            message += f"- [{ex.get('ExerciseTitle','Exercise')}]({url})\n"
    return message

# ------------------------------------------------------------------------------------
# 5. LOGIN SCREEN
# ------------------------------------------------------------------------------------
def login_screen():
    st.markdown("## EeeBee AI Buddy Login")
    user_type = st.radio("User Type", ["Student", "Teacher"])
    user_type_value = 2 if user_type == "Teacher" else None

    org_code = st.text_input("🏫 School Code", key="org_code")
    login_id = st.text_input("👤 Login ID", key="login_id")
    password = st.text_input("🔒 Password", type="password", key="password")

    # Read E or T query param
    query_params = st.experimental_get_query_params()
    E_value = query_params.get("E", [None])[0]
    T_value = query_params.get("T", [None])[0]

    api_url = None
    topic_id = None

    if E_value and T_value:
        st.warning("Please provide only E or T, not both.")
    elif E_value and not T_value:
        st.session_state.is_english_mode = True
        api_url = API_AUTH_URL_ENGLISH
        topic_id = E_value
    elif T_value and not E_value:
        st.session_state.is_english_mode = False
        api_url = API_AUTH_URL_MATH_SCIENCE
        topic_id = T_value
    else:
        st.warning("Missing E or T param in URL")

    if st.button("🚀 Login"):
        if not api_url or not topic_id:
            st.warning("Please provide correct E or T param.")
            return

        payload = {
            "OrgCode": org_code,
            "TopicID": int(topic_id),
            "LoginID": login_id,
            "Password": password
        }
        if user_type_value:
            payload["UserType"] = user_type_value

        # Define headers here
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        with st.spinner("🔄 Authenticating..."):
            try:
                # Include headers in the POST request
                resp = requests.post(api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if data.get("statusCode") == 1:
                    st.session_state.auth_data = data
                    st.session_state.is_authenticated = True
                    st.session_state.is_teacher = (user_type_value == 2)
                    st.session_state.topic_id = int(topic_id)

                    if not st.session_state.is_teacher:
                        # If a student logs in, store their weak concepts
                        st.session_state.student_weak_concepts = data.get("WeakConceptList", [])

                    st.rerun()
                else:
                    st.error("🚫 Authentication failed. Check your credentials.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error in authentication: {e}")

# ------------------------------------------------------------------------------------
# 6. TEACHER SETUP & DASHBOARD
# ------------------------------------------------------------------------------------
def teacher_setup_and_dashboard():
    """
    - Lets the teacher pick a batch
    - Fetches 'teacher_data' from the combined endpoint
    - Shows concept-level analytics (Attended vs Cleared)
    - Lets teacher pick a student
    - Allows generating exam questions or suggestions for that student
    """
    auth_data = st.session_state.auth_data
    batches = auth_data.get("BatchList", [])
    if not batches:
        st.warning("No batches found.")
        return

    batch_map = {b["BatchName"]: b for b in batches}
    selected_batch_name = st.selectbox("Select Batch (Class):", list(batch_map.keys()))
    selected_batch = batch_map[selected_batch_name]
    batch_id = selected_batch["BatchID"]
    total_students = selected_batch.get("StudentCount", 0)

    # 6.1. Fetch teacher_data if new batch is chosen
    if batch_id and st.session_state.selected_batch_id != batch_id:
        st.session_state.selected_batch_id = batch_id
        user_info = auth_data.get("UserInfo", [{}])[0]
        org_code = user_info.get("OrgCode", "012")

        payload = {
            "BatchID": batch_id,
            "TopicID": st.session_state.topic_id,
            "OrgCode": org_code
        }

        # Define headers here
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        with st.spinner("🔄 Fetching class data..."):
            try:
                # Include headers in the POST request
                resp = requests.post(API_TEACHER_WEAK_CONCEPTS_AND_STUDENTS, json=payload, headers=headers)
                resp.raise_for_status()
                teacher_data = resp.json()  # {Concepts: [...], Students: [...]}
                st.session_state.teacher_data = teacher_data
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching teacher data: {e}")
                st.session_state.teacher_data = {"Concepts": [], "Students": []}

    # 6.2. Display concept-level analytics
    teacher_data = st.session_state.teacher_data
    concepts = teacher_data.get("Concepts", [])
    students = teacher_data.get("Students", [])

    st.markdown("### 📊 Concept-Level Overview")
    if concepts:
        df_concepts = pd.DataFrame(concepts)
        # Melt for charting Attended vs Cleared
        df_melted = df_concepts.melt(
            id_vars=["ConceptText"],
            value_vars=["AttendedStudentCount", "ClearedStudentCount"],
            var_name="Category", value_name="Count"
        )
        df_melted["Category"] = df_melted["Category"].replace({
            "AttendedStudentCount": "Attended",
            "ClearedStudentCount": "Cleared"
        })

        chart = (
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
        st.altair_chart(chart, use_container_width=True)

        st.write("**📈 Per Concept Stats:**")
        for c in concepts:
            st.write(
                f"- **{c['ConceptText']}**: Attended = {c['AttendedStudentCount']}, "
                f"Cleared = {c['ClearedStudentCount']}"
            )
    else:
        st.info("No concept data available.")

    # 6.3. Student-level table
    st.markdown("### 👥 Student-Level Data")
    if students:
        df_students = pd.DataFrame(st.session_state.teacher_data["Students"])
        st.dataframe(df_students[["FullName", "TotalConceptCount", "WeakConceptCount", "ClearedConceptCount"]])

        # Horizontal bar or grouped bar for Weak vs Cleared
        df_students_melted = df_students.melt(
            id_vars=["FullName"],
            value_vars=["WeakConceptCount", "ClearedConceptCount"],
            var_name="Status", value_name="Count"
        )
        chart2 = (
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
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No student data available.")

    # 6.4. Select a single student to focus on
    if students:
        student_map = {s["FullName"]: s for s in students}
        selected_student_name = st.selectbox("🎯 Select Student for Detailed Interaction:", list(student_map.keys()))
        selected_student = student_map[selected_student_name]

        # If new student is selected, store in session
        if st.session_state.selected_student_id != selected_student["UserID"]:
            st.session_state.selected_student_id = selected_student["UserID"]
            st.session_state.selected_student_info = selected_student

        st.write(f"**Selected Student**: {selected_student_name}")
        st.write(
            f"- Weak Concepts: {selected_student['WeakConceptCount']}, "
            f"Cleared Concepts: {selected_student['ClearedConceptCount']} "
            f"(Total Concepts: {selected_student['TotalConceptCount']})"
        )

        # 6.5. Generate exam questions for a concept
        st.subheader("📝 Generate Exam Questions for a Concept")
        if concepts:
            concept_map = {c["ConceptText"]: c["ConceptID"] for c in concepts}
            bloom_level = st.radio(
                "Select Bloom's Taxonomy Level for the Questions",
                ["L1 (Remember)", "L2 (Understand)", "L3 (Apply)", "L4 (Analyze)", "L5 (Evaluate)"],
                index=3
            )
            chosen_concept_text = st.selectbox("Select Concept:", list(concept_map.keys()))

            if st.button("Generate Exam Questions"):
                short_bloom = bloom_level.split()[0]  # e.g., "L4"
                branch_name = st.session_state.auth_data.get("BranchName", "their class")
                prompt = (
                    f"You are an educational AI assisting a teacher. "
                    f"Please create 20 exam questions on the concept '{chosen_concept_text}', "
                    f"for {branch_name} students, aligned with NCERT & NEP 2020. "
                    f"Use Bloom's {short_bloom} difficulty. "
                    f"Do not provide solutions, only questions. Use LaTeX for math."
                )
                try:
                    with st.spinner("Generating questions..."):
                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": prompt}],
                            max_tokens=2000
                        )
                        st.session_state.exam_questions = response.choices[0].message["content"].strip()
                    st.success("Exam questions generated successfully!")
                except Exception as e:
                    st.error(f"Error generating questions: {e}")

        # Show any generated exam questions
        if st.session_state.exam_questions:
            st.markdown("### 📄 Generated Exam Questions")
            st.markdown(st.session_state.exam_questions)

            # PDF Download Option
            pdf_bytes = generate_exam_questions_pdf(
                st.session_state.exam_questions,
                st.session_state.selected_student_info["FullName"],
                st.session_state.selected_student_info["FullName"]
            )
            st.download_button(
                label="📥 Download Exam Questions as PDF",
                data=pdf_bytes,
                file_name=f"Exam_Questions_{st.session_state.selected_student_info['FullName']}.pdf",
                mime="application/pdf"
            )

        # 6.6. Generate Individualized Teaching Suggestions
        st.subheader("💡 Get Individualized Teaching Suggestions")
        if selected_student:
            if st.button("Ask EeeBee for Suggestions"):
                suggestions = generate_student_suggestions(selected_student)
                if suggestions:
                    st.session_state[f"suggestions_{selected_student['UserID']}"] = suggestions
                    st.success("Suggestions generated successfully!")

            # Display suggestions if they exist
            existing_suggestions = st.session_state.get(f"suggestions_{selected_student['UserID']}", "")
            if existing_suggestions:
                with st.expander(f"EeeBee's Suggestions for {selected_student_name}", expanded=True):
                    st.markdown(existing_suggestions)

# ------------------------------------------------------------------------------------
# 7. SYSTEM PROMPT FOR GPT (CONTEXTUALIZED)
# ------------------------------------------------------------------------------------
def get_system_prompt():
    """
    If teacher & a student is selected, incorporate that student's data
    so EeeBee can tailor the conversation to that specific student.
    """
    topic_name = st.session_state.auth_data.get("TopicName", "Unknown Topic")
    branch_name = st.session_state.auth_data.get("BranchName", "their class")

    if st.session_state.is_teacher:
        student_info = st.session_state.selected_student_info
        if student_info:
            sname = student_info["FullName"]
            wcount = student_info["WeakConceptCount"]
            ccount = student_info["ClearedConceptCount"]
            tcount = student_info["TotalConceptCount"]
            student_context = (
                f"Focus on student {sname}. They have {wcount} weak concepts, {ccount} cleared, out of {tcount} total."
            )
        else:
            student_context = "No specific student selected yet."

        prompt = f"""
You are EeeBee, an educational AI from iEdubull, specialized in {topic_name}.

Teacher Mode:
- The user is a teacher instructing {branch_name} students (NCERT).
- Provide suggestions for explaining concepts, designing questions, addressing the student's learning gaps.
- Avoid providing direct solutions; prefer step-by-step reasoning and guidance.
- Math in LaTeX: $...$ or $$...$$.
{student_context}
"""
    else:
        # Student mode
        weak_concepts_list = [wc["ConceptText"] for wc in st.session_state.student_weak_concepts]
        w_text = ", ".join(weak_concepts_list) if weak_concepts_list else "none"
        prompt = f"""
You are EeeBee, an educational AI from iEdubull, specialized in {topic_name}.

Student Mode:
- The student is in {branch_name} (NCERT).
- Their weak concepts: {w_text}.
- Provide step-by-step hints, no direct solutions.
- Math in LaTeX: $...$ or $$...$$.
"""

    return prompt.strip()

# ------------------------------------------------------------------------------------
# 8. CHAT FUNCTIONALITY
# ------------------------------------------------------------------------------------
def add_initial_greeting():
    if not st.session_state.chat_history and st.session_state.auth_data:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        greeting = f"Hello {user_name}, I'm EeeBee! How can I help you today?"
        st.session_state.chat_history.append(("assistant", greeting))

def handle_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()

def get_gpt_response(user_input):
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": role, "content": content}
        for (role, content) in st.session_state.chat_history
    ]
    try:
        with st.spinner("🤖 EeeBee is thinking..."):
            # Example: if the user input mentions a concept resource
            # you could optionally detect that and fetch resources, etc.

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # or whichever model you have access to
                messages=messages,
                max_tokens=2000
            )
            answer = response.choices[0].message["content"].strip()
            st.session_state.chat_history.append(("assistant", answer))
    except Exception as e:
        st.error(f"Error from GPT: {e}")

def generate_student_suggestions(student_info):
    """
    Generate personalized teaching suggestions for a specific student.
    """
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data.get("TopicName", "the subject")
    branch_name = st.session_state.auth_data.get("BranchName", "their class")

    prompt = (
        f"You are an educational AI assistant named EeeBee, specialized in {topic_name}.\n\n"
        f"The teacher, {user_name}, is looking for individualized suggestions to help the student:\n"
        f"• Name: {student_info['FullName']}\n"
        f"• Total Concepts: {student_info['TotalConceptCount']}\n"
        f"• Weak Concepts: {student_info['WeakConceptCount']}\n"
        f"• Cleared Concepts: {student_info['ClearedConceptCount']}\n\n"
        f"Assume the student is in {branch_name} following NCERT curriculum. Provide:\n"
        f"1) Strategies to strengthen their weak concepts.\n"
        f"2) Suggestions for practice or homework tailored to their level.\n"
        f"3) Methods of motivation and engagement based on their progress.\n\n"
        f"Ensure all suggestions are actionable and relevant to {branch_name}-level education.\n"
        f"Avoid providing direct solutions; focus on step-by-step learning."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or whichever model you have access to
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
# 9. MAIN SCREEN (POST-LOGIN)
# ------------------------------------------------------------------------------------
def main_screen():
    user_info = st.session_state.auth_data['UserInfo'][0]
    user_name = user_info['FullName']
    topic_name = st.session_state.auth_data.get("TopicName", "the topic")

    # A quick logout button
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # Display header with icon
    icon_img = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/icon.png"
    st.markdown(
        f"""
        # Hello {user_name}, 
        <img src="{icon_img}" alt="EeeBee AI" style="width:55px; vertical-align:middle;">
        EeeBee AI buddy is here to help you with :blue[{topic_name}]
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.is_teacher:
        # Teacher sees two tabs: "Teacher Setup & Dashboard" and "Chat with EeeBee"
        tab1, tab2 = st.tabs(["Teacher Setup & Dashboard", "💬 Chat with EeeBee"])

        with tab1:
            st.subheader("1. 📚 Class & Student Selection")
            teacher_setup_and_dashboard()

        with tab2:
            st.subheader("2. 💬 Chat (Context: Selected Student)")
            add_initial_greeting()

            # Chat UI
            chat_box = "<div style='height:400px; overflow-y: auto; border:1px solid #ddd; padding:8px; background:#f9f9f9; border-radius:8px;'>"
            for role, msg in st.session_state.chat_history:
                if role == "assistant":
                    chat_box += f"<div style='background:#e0e7ff; color:#000; margin:5px; padding:8px; border-radius:8px;'>🤖 EeeBee: {msg}</div>"
                else:
                    chat_box += f"<div style='background:#2563eb; color:#fff; margin:5px; padding:8px; border-radius:8px;'>👤 You: {msg}</div>"
            chat_box += "</div>"
            st.markdown(chat_box, unsafe_allow_html=True)

            user_input = st.chat_input("Ask EeeBee about this student or any concept.")
            if user_input:
                handle_user_input(user_input)

    else:
        # Student mode
        if st.session_state.is_english_mode:
            # English Students: only Chat
            tab = st.tabs(["💬 Chat"])[0]
            with tab:
                st.subheader("💬 Chat with your EeeBee AI buddy")
                add_initial_greeting()

                # Chat UI
                chat_box = "<div style='height:400px; overflow-y: auto; border:1px solid #ddd; padding:8px; background:#f9f9f9; border-radius:8px;'>"
                for role, msg in st.session_state.chat_history:
                    if role == "assistant":
                        chat_box += f"<div style='background:#e0e7ff; color:#000; margin:5px; padding:8px; border-radius:8px;'>🤖 EeeBee: {msg}</div>"
                    else:
                        chat_box += f"<div style='background:#2563eb; color:#fff; margin:5px; padding:8px; border-radius:8px;'>👤 You: {msg}</div>"
                chat_box += "</div>"
                st.markdown(chat_box, unsafe_allow_html=True)

                user_input = st.chat_input("Ask EeeBee anything about the topic.")
                if user_input:
                    handle_user_input(user_input)
        else:
            # Non-English Students: Chat + Learning Path
            tab1, tab2 = st.tabs(["💬 Chat", "🧠 Learning Path"])

            with tab1:
                st.subheader("💬 Chat with your EeeBee AI buddy")
                add_initial_greeting()

                # Chat UI
                chat_box = "<div style='height:400px; overflow-y: auto; border:1px solid #ddd; padding:8px; background:#f9f9f9; border-radius:8px;'>"
                for role, msg in st.session_state.chat_history:
                    if role == "assistant":
                        chat_box += f"<div style='background:#e0e7ff; color:#000; margin:5px; padding:8px; border-radius:8px;'>🤖 EeeBee: {msg}</div>"
                    else:
                        chat_box += f"<div style='background:#2563eb; color:#fff; margin:5px; padding:8px; border-radius:8px;'>👤 You: {msg}</div>"
                chat_box += "</div>"
                st.markdown(chat_box, unsafe_allow_html=True)

                user_input = st.chat_input("Ask EeeBee anything about the topic.")
                if user_input:
                    handle_user_input(user_input)

            with tab2:
                st.subheader("🧠 Generate Your Learning Path")
                weak_concepts = st.session_state.student_weak_concepts
                if not weak_concepts:
                    st.warning("No weak concepts found.")
                else:
                    for idx, concept in enumerate(weak_concepts):
                        concept_text = concept.get("ConceptText", f"Concept {idx+1}")
                        concept_id = concept.get("ConceptID", f"id_{idx+1}")

                        st.markdown(f"#### **Weak Concept {idx+1}:** {concept_text}")

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
                                st.session_state.auth_data.get('ConceptList', []),
                                st.session_state.topic_id
                            )

# ------------------------------------------------------------------------------------
# 10. GENERATE LEARNING PATH PDF
# ------------------------------------------------------------------------------------
def generate_learning_path_pdf(learning_path, concept_text, user_name):
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
    content_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    # Title and subtitle
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

# ------------------------------------------------------------------------------------
# 11. DISPLAY LEARNING PATH + RESOURCES
# ------------------------------------------------------------------------------------
def display_learning_path_with_resources(concept_text, learning_path, concept_list, topic_id):
    branch_name = st.session_state.auth_data.get("BranchName", "their class")
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

# ------------------------------------------------------------------------------------
# 12. GENERATE EXAM QUESTIONS PDF
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# 13. GENERATE LEARNING PATH CONTENT (GPT)
# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------
# 14. DISPLAY LEARNING PATH + RESOURCES
# ------------------------------------------------------------------------------------
def display_learning_path_with_resources(concept_text, learning_path, concept_list, topic_id):
    branch_name = st.session_state.auth_data.get("BranchName", "their class")
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

# ------------------------------------------------------------------------------------
# 15. GENERATE STUDENT SUGGESTIONS
# ------------------------------------------------------------------------------------
def generate_student_suggestions(student_info):
    """
    Generate personalized teaching suggestions for a specific student.
    """
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data.get("TopicName", "the subject")
    branch_name = st.session_state.auth_data.get("BranchName", "their class")

    prompt = (
        f"You are an educational AI assistant named EeeBee, specialized in {topic_name}.\n\n"
        f"The teacher, {user_name}, is looking for individualized suggestions to help the student:\n"
        f"• Name: {student_info['FullName']}\n"
        f"• Total Concepts: {student_info['TotalConceptCount']}\n"
        f"• Weak Concepts: {student_info['WeakConceptCount']}\n"
        f"• Cleared Concepts: {student_info['ClearedConceptCount']}\n\n"
        f"Assume the student is in {branch_name} following NCERT curriculum. Provide:\n"
        f"1) Strategies to strengthen their weak concepts.\n"
        f"2) Suggestions for practice or homework tailored to their level.\n"
        f"3) Methods of motivation and engagement based on their progress.\n\n"
        f"Ensure all suggestions are actionable and relevant to {branch_name}-level education.\n"
        f"Avoid providing direct solutions; focus on step-by-step learning."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or whichever model you have access to
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
# 16. MAIN APP ENTRY POINT
# ------------------------------------------------------------------------------------
def main():
    if not st.session_state.is_authenticated:
        login_screen()
    else:
        main_screen()

if __name__ == "__main__":
    main()
