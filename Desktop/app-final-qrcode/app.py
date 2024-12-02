import os
import re
import json
import streamlit as st
import openai
import requests
import warnings
import streamlit.components.v1 as components
from PIL import Image
import requests
from io import BytesIO

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

config_path = 'config.json'  # Correct assignment


working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, 'config.json')

# try:
#     with open(config_path, 'r') as config_file:
#         config_data = json.load(config_file)
#     OPENAI_API_KEY = config_data["openai_api_key"]
#     openai.api_key = OPENAI_API_KEY
# except (FileNotFoundError, KeyError) as e:
#     st.error(f"Configuration error: {e}")
#     st.stop()


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
    page_title="EeeBee AI Buddy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Utility Function to Generate Learning Path
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
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1000
            ).choices[0].message['content'].strip()
            learning_path[concept_text] = gpt_response
        except Exception as e:
            learning_path[concept_text] = f"Error generating learning path: {e}"
    return learning_path

# Utility Function to Display Learning Path
def display_learning_path(learning_path):
    """
    Display the generated learning path with collapsible functionality.
    """
    # Regex for detecting LaTeX math expressions
    MATH_REGEX = r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])"  
    
    with st.expander("📚 Generated Learning Path", expanded=True):
        for concept, path in learning_path.items():
            # Display the concept as a subheader
            st.markdown(f"### Weak Concept: {concept}")
            
            # Split the learning path into parts (math and non-math)
            parts = re.split(MATH_REGEX, path)
            
            for part in parts:
                part = part.strip()
                if re.match(MATH_REGEX, part):  # Check if the part is LaTeX math
                    try:
                        # Remove extra \(, \), $, etc., as Streamlit doesn't need them explicitly
                        clean_part = part.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "").strip("$")
                        st.latex(clean_part)
                    except Exception as e:
                        st.markdown(f"**Math Error:** Unable to render `{part}`. Error: {e}")
                elif part:  # Handle non-empty non-math text
                    st.markdown(part)



# Define login screen
def login_screen():
    try:
        # External image URL
        image_url = "https://raw.githubusercontent.com/EdubullTechnologies/QR-ChatBot/master/Desktop/app-final-qrcode/assets/login_page_img.png"

        # Display the image from the URL with a caption and width
        col1, col2 = st.columns([1, 2])  # Adjusted to make the image column smaller

        with col1:
            st.image(image_url, width=160)

        # Apply custom CSS to increase font size and adjust positioning
        st.markdown("""
        <style>
        .title {
            font-size: 4em;  /* Adjust font size to make it bigger */
            font-weight: bold;
            margin-top: 90px;  /* Move the text closer to the image */
            margin-left: -125px;
            text-align: left;  /* Align the text to the left */
        }
        </style>
        """, unsafe_allow_html=True)

        # Display the title in the second column
        with col2:
            st.markdown('<div class="title">EeeBee AI Buddy Login</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading image: {e}")

    # st.title("🤖 EeeBee AI Buddy Login")
    st.markdown('<h3 style="font-size: 1.5em;">🦾 Welcome! Please enter your credentials to chat with your AI Buddy!</h3>', unsafe_allow_html=True)
    
    # Input fields for organization code, login ID, and password
    org_code = st.text_input("🏫 School Code", key="org_code")  # No default value
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
            f"Hello {user_name}! I'm your 🤖 EeeBee AI buddy. "
            f"How can I help you with {topic_name} today?"
        )
        st.session_state.chat_history.append(("assistant", greeting_message))

# Callback function for handling user input
def handle_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        get_gpt_response(user_input)
        st.rerun()  # Force rerun to immediately display the new message



# Define the main screen
def main_screen():
    user_name = st.session_state.auth_data['UserInfo'][0]['FullName']
    topic_name = st.session_state.auth_data['TopicName']
    
    # Custom title greeting the user
    
    col1, col2 = st.columns([9, 1])  # Adjust the proportions as needed
    with col2:
        if st.button("Logout"):
            # Clear session states related to authentication and conversation history
            st.session_state.clear()  # Clears all session states
            st.rerun()  # Refresh the app to go back to the login screen
    
    st.title(f"Hello {user_name}, 🤖 EeeBee AI buddy is here to help you", anchor=None)

    # Debugging: Display the auth data


     # Button to generate learning path
    if st.button("🧠 Generate Learning Path"):
        weak_concepts = st.session_state.auth_data.get("WeakConceptList", [])
        if weak_concepts:
            learning_path = generate_learning_path(weak_concepts)
            display_learning_path(learning_path)
        else:
            st.error("No weak concepts found!")

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
    st.subheader("Chat with your 🤖 EeeBee AI buddy", anchor=None)
    chat_container = st.container()
    with chat_container:
        chat_history_html = """
        <div style="height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f3f4f6; border-radius: 10px;">
        """
        for role, message in st.session_state.chat_history:
            if role == "assistant":
                chat_history_html += f"<div style='text-align: left; color: #000; background-color: #e0e7ff; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>EeeBee:</b> {message}</div>"
            else:
                chat_history_html += f"<div style='text-align: left; color: #fff; background-color: #2563eb; padding: 8px; border-radius: 8px; margin-bottom: 5px;'><b>{st.session_state.auth_data['UserInfo'][0]['FullName']}:</b> {message}</div>"
        chat_history_html += "</div>"
        st.markdown(chat_history_html, unsafe_allow_html=True)

    # User input with st.chat_input
    user_input = st.chat_input("Enter your question about the topic")
    if user_input:
        handle_user_input(user_input)


# Function to get GPT-4 response
def get_gpt_response(user_input):
    conversation_history_formatted = [
        {"role": "system",  "content": f"You are a highly knowledgeable educational assistant created by edubull and your name is EeeBee. The student is asking questions related to the topic '{st.session_state.auth_data.get('TopicName', 'Unknown Topic')}'. Engage with the student by asking guiding questions that encourage them to understand the problem step-by-step. Avoid providing direct answers; instead, prompt them to think critically and break down the problem. Offer hints or pose questions like 'What do you think the first step might be?' or 'How would you approach this part of the problem?' Continue prompting them through each part until they arrive at the answer on their own. Ensure that your tone is supportive and encouraging, aiming to build the student's confidence and understanding, and make sure all responses are relevant to the topic and suitable for a school setting."}
    ]
    conversation_history_formatted += [{"role": role, "content": content} for role, content in st.session_state.chat_history]

    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=conversation_history_formatted,
            max_tokens=1000
        ).choices[0].message['content'].strip()

        # Append GPT-4's response to chat history
        st.session_state.chat_history.append(("assistant", gpt_response))
    except Exception as e:
        st.error(f"Error in GPT-4 response generation: {e}")

# Function to load content and generate a description for the selected concept
def load_concept_content():
    # Get the selected concept name and ID from ConceptList
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
        # Fetch content data from API
        content_response = requests.post(API_CONTENT_URL, json=content_payload, headers=headers)
        content_response.raise_for_status()
        content_data = content_response.json()

        # Generate a description for the selected concept using ChatGPT
        prompt = f"Provide a concise and educational description of the concept '{selected_concept_name}' to help students understand it better."

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=150
        ).choices[0].message['content'].strip()

        # Replace any generic references to "this concept" with the actual concept name
        gpt_response = gpt_response.replace("This concept", selected_concept_name).replace("this concept", selected_concept_name)

        gpt_response += "\n\nYou can check the resources below for more information."

        # Save the description in session state to display later
        st.session_state.generated_description = gpt_response

        # Display resources
        display_resources(content_data)

    except requests.exceptions.RequestException as req_err:
        st.error(f"Error fetching content: {req_err}")
    except Exception as e:
        st.error(f"Error generating concept description: {e}")

# Function to display resources (videos, notes, exercises) with generated concept description
def display_resources(content_data):
    with st.expander("Resources", expanded=True):
        
        # Display the generated concept description from ChatGPT
        concept_description = st.session_state.get("generated_description", "No description available.")
        st.markdown(f"### Concept Description\n{concept_description}\n")

        # Display video resources
        if content_data.get("Video_List"):
            st.write("*Videos*")
            for video in content_data["Video_List"]:
                video_url = video.get("LectureLink", f"https://www.edubull.com/courses/videos/{video.get('LectureID', '')}")
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
