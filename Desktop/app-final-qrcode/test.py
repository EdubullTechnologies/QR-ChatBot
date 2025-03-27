import streamlit as st
import openai
import time

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

########################################
# Session State & Helper Functions
########################################

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "openai_model" not in st.session_state:
    # The name of the model you want to use. 
    st.session_state["openai_model"] = "gpt-4o"

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

st.title("ChatGPT-like Clone with File Upload & Processing")

# “System” block asking whether user wants to upload a file
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
                time.sleep(2)
            st.success("File uploaded and stored successfully!")

########################################
# Show Past Chat Messages
########################################

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


########################################
# Chat Input & AI Processing
########################################

user_input = st.chat_input("Ask something, or request a summary of your uploaded file...")

if user_input:
    # Append the user's message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the messages array we'll send to OpenAI
    # Optionally, if you want the assistant to have access to the file content,
    # you can inject a "system" message telling the AI what the file is about:
    messages_for_openai = []

    # 1) If we have file content, pass it into a system message so GPT can "see" it.
    if st.session_state["file_content"]:
        system_msg = (
            "The user has uploaded a file with the following text:\n\n"
            f"{st.session_state['file_content']}\n\n"
            "You can use this file content to answer the user's questions."
        )
        messages_for_openai.append({"role": "system", "content": system_msg})

    # 2) Add all previous messages in the conversation
    messages_for_openai.extend(
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"]
    )

    # Make the streaming call
    response_buffer = ""
    with st.chat_message("assistant"):
        stream = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=messages_for_openai,
            stream=True,
        )
        response_container = st.empty()
        for chunk in stream:
            chunk_message = chunk["choices"][0].get("delta", {}).get("content", "")
            response_buffer += chunk_message
            response_container.markdown(response_buffer)

    # Save the assistant response
    st.session_state["messages"].append(
        {"role": "assistant", "content": response_buffer}
    )
