import streamlit as st
import time
from openai import OpenAI

########################################
# Gemini-Like Client Setup
########################################

# Create a Gemini-like client by overriding the base_url
client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

########################################
# Session State & Helper Functions
########################################

# Conversation messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# The name of the model you want to use (change if needed)
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-2.0-flash"

# Control whether the file uploader is visible
if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

# We'll store file content in session state, once uploaded
if "file_content" not in st.session_state:
    st.session_state["file_content"] = None

def toggle_upload_visibility(visible: bool):
    st.session_state["uploader_visible"] = visible


########################################
# Page Layout
########################################

st.title("Gemini-Like Chat App with File Upload & Processing")

# Ask if user wants to upload a file
with st.chat_message("system"):
    cols = st.columns((3,1,1))
    cols[0].write("Would you like to upload a file?")
    cols[1].button("Yes", use_container_width=True,
                   on_click=toggle_upload_visibility, args=[True])
    cols[2].button("No", use_container_width=True,
                   on_click=toggle_upload_visibility, args=[False])

# If user selected "Yes", show the uploader
if st.session_state["uploader_visible"]:
    with st.chat_message("system"):
        file = st.file_uploader("Upload your data here (text file for demo)")
        if file is not None:
            with st.spinner("Reading/processing your file..."):
                # Read the file into session state (assuming it's text)
                file_content = file.read().decode("utf-8", errors="ignore")
                st.session_state["file_content"] = file_content
                time.sleep(2)  # Simulate some processing time
            st.success("File uploaded and stored successfully!")

########################################
# Show Past Chat Messages
########################################

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

########################################
# Chat Input & Gemini Processing
########################################

user_input = st.chat_input("Ask something, or request a summary of your uploaded file...")

if user_input:
    # 1) User's message goes to conversation history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display the user's message right away
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Build the message list we'll send to the Gemini-like model
    messages_for_gemini = []

    # (Optional) Provide the file's text via a system message
    if st.session_state["file_content"]:
        system_msg = (
            "The user has uploaded a file with the following text:\n\n"
            f"{st.session_state['file_content']}\n\n"
            "You can use this file content to answer the user's questions."
        )
        messages_for_gemini.append({"role": "system", "content": system_msg})

    # Add all conversation messages (including user’s latest)
    messages_for_gemini.extend(
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
    )

    # 3) Streaming response from Gemini-like endpoint
    response_buffer = ""
    with st.chat_message("assistant"):
        # Attempt a streaming call; your endpoint must support it
        response_stream = client.chat.completions.create(
            model=st.session_state["gemini_model"],
            messages=messages_for_gemini,
            stream=True,
        )
        response_container = st.empty()

        for chunk in response_stream:
            # Standard OpenAI-like streaming format
            chunk_message = chunk["choices"][0].get("delta", {}).get("content", "")
            response_buffer += chunk_message
            response_container.markdown(response_buffer)

    # 4) Store the assistant's reply
    st.session_state["messages"].append(
        {"role": "assistant", "content": response_buffer}
    )
