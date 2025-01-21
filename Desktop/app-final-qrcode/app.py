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

    org_code = st.text_input("School Code")
    login_id = st.text_input("Login ID")
    password = st.text_input("Password", type="password")

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

    if st.button("Login"):
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

        with st.spinner("Authenticating..."):
            try:
                resp = requests.post(api_url, json=payload)
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

                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials.")
            except Exception as e:
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
        with st.spinner("Fetching class data..."):
            try:
                resp = requests.post(API_TEACHER_WEAK_CONCEPTS_AND_STUDENTS, json=payload)
                resp.raise_for_status()
                teacher_data = resp.json()  # {Concepts: [...], Students: [...]}
                st.session_state.teacher_data = teacher_data
            except Exception as e:
                st.error(f"Error fetching teacher data: {e}")
                st.session_state.teacher_data = {"Concepts": [], "Students": []}

    # 6.2. Display concept-level analytics
    teacher_data = st.session_state.teacher_data
    concepts = teacher_data.get("Concepts", [])
    students = teacher_data.get("Students", [])

    st.markdown("### Concept-Level Overview")
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

        st.write("**Per Concept Stats**:")
        for c in concepts:
            st.write(
                f"- {c['ConceptText']}: Attended={c['AttendedStudentCount']}, "
                f"Cleared={c['ClearedStudentCount']}"
            )
    else:
        st.info("No concept data available.")

    # 6.3. Student-level table
    st.markdown("### Student-Level Data")
    if students:
        df_students = pd.DataFrame(students)
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
        selected_student_name = st.selectbox("Select a Student:", list(student_map.keys()))
        selected_student = student_map[selected_student_name]

        # If new student is selected, store in session
        if st.session_state.selected_student_id != selected_student["UserID"]:
            st.session_state.selected_student_id = selected_student["UserID"]
            st.session_state.selected_student_info = selected_student

        st.write(f"**Selected Student**: {selected_student_name}")
        st.write(
            f"- Weak Concepts: {selected_student['WeakConceptCount']}, "
            f"Cleared: {selected_student['ClearedConceptCount']} "
            f"(out of {selected_student['TotalConceptCount']})"
        )

        # 6.5. Generate exam questions for a concept
        st.subheader("Generate Exam Questions for a Concept")
        if concepts:
            concept_map = {c["ConceptText"]: c["ConceptID"] for c in concepts}
            bloom_level = st.radio(
                "Bloom's Level",
                ["L1 (Remember)", "L2 (Understand)", "L3 (Apply)", "L4 (Analyze)", "L5 (Evaluate)"],
                index=3
            )
            chosen_concept_text = st.selectbox("Select Concept:", list(concept_map.keys()))

            if st.button("Generate Exam Questions"):
                short_bloom = bloom_level.split()[0]
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
                    st.success("Exam questions generated!")
                except Exception as e:
                    st.error(f"Error generating questions: {e}")

        # Show any generated exam questions
        if st.session_state.exam_questions:
            st.markdown("### Generated Exam Questions")
            st.markdown(st.session_state.exam_questions)

    else:
        st.info("Select a student to see more options.")


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
        st.experimental_rerun()

def get_gpt_response(user_input):
    system_prompt = get_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": role, "content": content}
        for (role, content) in st.session_state.chat_history
    ]
    try:
        with st.spinner("EeeBee is thinking..."):
            # Example: if the user input mentions a concept resource
            # you could optionally detect that and fetch resources, etc.

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000
            )
            answer = response.choices[0].message["content"].strip()
            st.session_state.chat_history.append(("assistant", answer))

    except Exception as e:
        st.error(f"Error from GPT: {e}")


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
            st.experimental_rerun()

    st.markdown(f"### Hello {user_name}, EeeBee is here to help with **{topic_name}**.")

    if st.session_state.is_teacher:
        # Teacher sees two tabs: "Teacher Dashboard" and "Chat"
        tab1, tab2 = st.tabs(["Teacher Setup & Dashboard", "Chat with EeeBee"])

        with tab1:
            st.subheader("1. Class/Student Selection & Dashboard")
            teacher_setup_and_dashboard()

        with tab2:
            st.subheader("2. Chat (Context: Selected Student)")
            add_initial_greeting()

            # Chat UI
            chat_box = "<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 8px; background: #f9f9f9; border-radius: 8px;'>"
            for role, msg in st.session_state.chat_history:
                if role == "assistant":
                    chat_box += f"<div style='background:#e0e7ff; color:#000; margin:5px; padding:8px; border-radius:8px;'>EeeBee: {msg}</div>"
                else:
                    chat_box += f"<div style='background:#2563eb; color:#fff; margin:5px; padding:8px; border-radius:8px;'>You: {msg}</div>"
            chat_box += "</div>"
            st.markdown(chat_box, unsafe_allow_html=True)

            user_input = st.chat_input("Ask EeeBee about this student or any concept.")
            if user_input:
                handle_user_input(user_input)

    else:
        # Student mode
        st.subheader("Student Mode: Chat with EeeBee")
        add_initial_greeting()

        chat_box = "<div style='height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 8px; background: #f9f9f9; border-radius: 8px;'>"
        for role, msg in st.session_state.chat_history:
            if role == "assistant":
                chat_box += f"<div style='background:#e0e7ff; color:#000; margin:5px; padding:8px; border-radius:8px;'>EeeBee: {msg}</div>"
            else:
                chat_box += f"<div style='background:#2563eb; color:#fff; margin:5px; padding:8px; border-radius:8px;'>You: {msg}</div>"
        chat_box += "</div>"
        st.markdown(chat_box, unsafe_allow_html=True)

        user_input = st.chat_input("Ask EeeBee anything about the topic.")
        if user_input:
            handle_user_input(user_input)


# ------------------------------------------------------------------------------------
# 10. APP ENTRY POINT
# ------------------------------------------------------------------------------------
def main():
    if not st.session_state.is_authenticated:
        login_screen()
    else:
        main_screen()

if __name__ == "__main__":
    main()
