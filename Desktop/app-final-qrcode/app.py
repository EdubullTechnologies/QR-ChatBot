import os
import json
import streamlit as st
import openai
import requests
import warnings
import streamlit.components.v1 as components

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)


# Load OpenAI API key from st.secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("API key for OpenAI not found in secrets.")

openai.api_key = OPENAI_API_KEY

# API URLs
API_AUTH_URL = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"

# Initialize session state variables if they don't exist
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



# Streamlit page settings
st.set_page_config(
    page_title="AI Buddy",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="auto"
)


# Define login screen
def login_screen():
    st.title("🤖 AI Buddy Login")
    st.markdown("# Scanned Topic: Rational Numbers")
    st.write("🦾 Welcome! Please enter your credentials to chat with your AI Buddy!")
    
    # Input fields for organization code, login ID, and password
    org_code = st.text_input("🏫 Organization Code", key="org_code")  # No default value
    login_id = st.text_input("👤 Login ID", key="login_id")           # No default value
    password = st.text_input("🔒 Password", type="password", key="password")  # No default value

    query_params = st.experimental_get_query_params()  # Replace with st.query_params after April 2024
    topic_id = query_params.get("T", [None])[0]

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if topic_id:
            auth_payload = {
                'OrgCode': org_code,
                'TopicID': int(topic_id),
                'LoginID': login_id,
                'Password': password
            }
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }
            try:
                auth_response = requests.post(API_AUTH_URL, json=auth_payload, headers=headers)
                auth_response.raise_for_status()
                auth_data = auth_response.json()
                if auth_data.get("statusCode") == 1:
                    st.session_state.auth_data = auth_data
                    st.session_state.is_authenticated = True
                    st.session_state.topic_id = int(topic_id)
                    st.rerun()
                else:
                    st.error("🚫 Authentication failed. Please check your credentials.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the authentication API: {e}")
        else:
            st.warning("❗Please enter a valid Topic ID.")


# Add initial greeting message
def add_initial_greeting():
    if len(st.session_state.chat_history) == 0:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data['TopicName']
        greeting_message = (
            f"Hello {user_name}! I'm your AI assistant. "
            f"How can I help you with {topic_name} today?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

# Callback function for handling user input
def handle_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()  # Force rerun to immediately display the new message





def main_screen():

    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']
    
    # Custom title greeting the user
    st.title(f"Hello {user_name}, your AI Buddy is here to help you", anchor=None)
    
    # Display the scanned topic in a larger size
    st.subheader(f"Scanned Topic: {topic_name}", anchor=None)
    
    # Display available concepts with topic name
    st.subheader(f"Available Concepts:", anchor=None)

    # List of available concepts
    concept_options = {concept['ConceptText']: concept['ConceptID'] for concept in st.session_state.auth_data['ConceptList']}
    for concept_text, concept_id in concept_options.items():
        if st.button(concept_text, key=f"concept_{concept_id}"):
            st.session_state.selected_concept_id = concept_id
            load_concept_content()

    add_initial_greeting()
    
    # Chatbox interface
    st.subheader("Chat with your AI Buddy", anchor=None)
    chat_container = st.container()
    with chat_container:
        chat_history_html = """
        <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
        """
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>AI:</b> {message}</div>"
            else:
                chat_history_html += f"<div style='text-align: right; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>You:</b> {message}</div>"
        chat_history_html += "</div>"
        st.markdown(chat_history_html, unsafe_allow_html=True)

    # User input with st.chat_input
    user_input = st.chat_input("Enter your question about the topic")
    if user_input:
        handle_user_input(user_input)

# Function to get GPT-4 response
def get_gpt_response(user_input):
    conversation_history_formatted = [
        {"role": "system", "content": f"You are a highly knowledgeable educational assistant. The student is asking questions related to the topic '{st.session_state.auth_data.get('TopicName', 'Unknown Topic')}'. Please only discuss information directly related to this topic and avoid answering any unrelated questions. Ensure that all responses are appropriate for a school setting."}
    ]
    conversation_history_formatted += [{"role": role, "content": content} for role, content in st.session_state.chat_history]

    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation_history_formatted,
            max_tokens=150
        ).choices[0].message['content'].strip()

        # Append GPT-4's response to chat history
        st.session_state.chat_history.append(("assistant", gpt_response))
    except Exception as e:
        st.error(f"Error in GPT-4 response generation: {e}")

# Function to load content based on the selected concept
def load_concept_content():
    content_payload = {
        'TopicID': st.session_state.topic_id,
        'ConceptID': int(st.session_state.selected_concept_id)
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
        display_resources(content_data)
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error fetching content: {req_err}")

# Function to display resources (videos, notes, exercises)
def display_resources(content_data):
    with st.expander("Resources", expanded=True):
        # Display video resources
        if content_data.get("Video_List"):
            st.write("*Videos*")
            for video in content_data["Video_List"]:
                video_url = video.get("LectureLink", f"https://example.com/{video.get('LectureID', '')}")
                st.write(f"[{video.get('LectureTitle', 'Untitled Video')}]({video_url})")

        # Display notes resources
        if content_data.get("Notes_List"):
            st.write("*Notes*")
            for note in content_data["Notes_List"]:
                note_url = f"{note.get('FolderName', '')}{note.get('PDFFileName', '')}"
                note_title = note.get("NotesTitle", "Untitled Note")
                st.write(f"[{note_title}]({note_url})")

        # Display exercises resources
        if content_data.get("Exercise_List"):
            st.write("*Exercises*")
            for exercise in content_data["Exercise_List"]:
                exercise_url = f"{exercise.get('FolderName', '')}{exercise.get('ExerciseFileName', '')}"
                exercise_title = exercise.get("ExerciseTitle", "Untitled Exercise")
                st.write(f"[{exercise_title}]({exercise_url})")

# Display login or main screen based on authentication
if st.session_state.is_authenticated:
    main_screen()
else:
    login_screen()
