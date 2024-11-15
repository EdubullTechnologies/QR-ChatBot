import os
import json
import streamlit as st
import openai
import requests
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="EeeBee AI Buddy", page_icon="🤖", layout="wide")

# Load OpenAI API key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("API key for OpenAI not found in secrets.")

# API URLs
API_AUTH_URL = "https://webapi.edubull.com/api/eProfessor/eProf_Org_StudentVerify_with_topic_for_chatbot"
API_CONTENT_URL = "https://webapi.edubull.com/api/eProfessor/WeakConcept_Remedy_List_ByConceptID"

# Initialize session state variables if they don't exist
def initialize_session():
    session_defaults = {
        "auth_data": None,
        "selected_concept_id": None,
        "conversation_history": [],
        "is_authenticated": False,
        "chat_history": [],
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session()

# Custom CSS for styling the chat UI
st.markdown("""
    <style>
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #f9fafb;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 16px;
            line-height: 1.5;
        }
        .chat-bubble.assistant {
            text-align: left;
            background-color: #e0e7ff;
            color: #000;
        }
        .chat-bubble.user {
            text-align: left;
            background-color: #2563eb;
            color: #fff;
        }
        .logout-btn {
            background-color: #ff4b4b;
            color: #fff;
            font-size: 16px;
            padding: 8px 20px;
            border-radius: 5px;
            margin-left: auto;
            margin-right: 0;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

def login_screen():
    st.title("🤖 EeeBee AI Buddy Login")
    st.write("🦾 Welcome! Enter your credentials to chat with your AI Buddy!")
    org_code = st.text_input("🏫 Organization Code")
    login_id = st.text_input("👤 Login ID")
    password = st.text_input("🔒 Password", type="password")
    topic_id = st.experimental_get_query_params().get("T", [None])[0]

    if st.button("🚀 Login and Start Chatting!") and not st.session_state.is_authenticated:
        if topic_id:
            authenticate_user(org_code, login_id, password, topic_id)
        else:
            st.warning("❗Please enter a valid Topic ID.")

def authenticate_user(org_code, login_id, password, topic_id):
    """Authenticate user via API and set session state upon success."""
    payload = {'OrgCode': org_code, 'TopicID': int(topic_id), 'LoginID': login_id, 'Password': password}
    try:
        response = requests.post(API_AUTH_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        if data.get("statusCode") == 1:
            st.session_state.auth_data = data
            st.session_state.is_authenticated = True
            st.session_state.topic_id = int(topic_id)
            st.experimental_rerun()
        else:
            st.error("🚫 Authentication failed. Check your credentials.")
    except requests.RequestException as e:
        st.error(f"Error connecting to the API: {e}")

def main_screen():
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Logout", key="logout", help="Click to logout"):
            st.session_state.clear()
            st.rerun()

    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']
    st.title(f"Hello {user_name}, 🤖 EeeBee is here to help!")
    st.subheader(f"Scanned Topic: {topic_name}")
    add_initial_greeting()
    display_chatbox()

def display_chatbox():
    st.subheader("Chat with EeeBee")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, message in st.session_state.chat_history:
        bubble_class = "assistant" if role == "assistant" else "user"
        st.markdown(f"<div class='chat-bubble {bubble_class}'>{message}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    user_input = st.chat_input("Enter your question")
    if user_input:
        handle_user_input(user_input)

def add_initial_greeting():
    if not st.session_state.chat_history:
        user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
        topic_name = st.session_state.auth_data['TopicName']
        greeting_message = (
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy. "
            f"How can I help you with {topic_name} today?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

def handle_user_input(user_input):
    st.session_state.chat_history.append(("user", user_input))
    get_gpt_response(user_input)
    st.experimental_rerun()

def get_gpt_response(user_input):
    conversation_history_formatted = [
        {
            "role": "system",
            "content": (
                "You are a highly knowledgeable educational assistant created by Edubull and your name is EeeBee. "
                f"The student is asking questions related to the topic '{st.session_state.auth_data.get('TopicName', 'Unknown Topic')}'. "
                "Engage with the student by asking guiding questions that encourage them to understand the problem step-by-step. "
                "Avoid providing direct answers; instead, prompt them to think critically and break down the problem. "
                "Offer hints or pose questions like 'What do you think the first step might be?' or 'How would you approach this part of the problem?' "
                "Continue prompting them through each part until they arrive at the answer on their own. "
                "Ensure that your tone is supportive and encouraging, aiming to build the student's confidence and understanding, "
                "and make sure all responses are relevant to the topic and suitable for a school setting."
            )
        }
    ]
    conversation_history_formatted += [{"role": role, "content": content} for role, content in st.session_state.chat_history[-5:]]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation_history_formatted,
            max_tokens=150
        ).choices[0].message['content']
        st.session_state.chat_history.append(("assistant", response.strip()))
    except Exception as e:
        st.error(f"Error in GPT response: {e}")

# Display the appropriate screen based on authentication
if st.session_state.is_authenticated:
    main_screen()
else:
    login_screen()
