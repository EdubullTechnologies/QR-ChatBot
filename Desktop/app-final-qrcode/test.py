import streamlit as st
import openai

st.title("ChatGPT-like Clone")

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Set a default model if not already defined
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"  # or "gpt-4" if preferred

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Append user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare to stream the assistant's response
    assistant_response = ""
    # Use a placeholder to update the response in real time
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Create a streaming response using OpenAI's API
        response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=st.session_state.messages,
            stream=True,
        )
        # Process each chunk in the streaming response
        for chunk in response:
            delta = chunk.choices[0].delta
            if "content" in delta:
                assistant_response += delta["content"]
                placeholder.markdown(assistant_response)

    # Append the complete assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
