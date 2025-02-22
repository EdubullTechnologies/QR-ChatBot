import streamlit as st
from google import genai

# Retrieve API key from Streamlit secrets
api_key = st.secrets["google"]["api_key"]

# Initialize the GenAI client with the secret API key
client = genai.Client(api_key=api_key)

def stream_response(prompt):
    """
    Generate a streaming response from the GenAI model.
    This function yields partial responses as they are received.
    """
    response_stream = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        stream=True  # Enable streaming mode if supported
    )
    full_response = ""
    for chunk in response_stream:
        full_response += chunk.text
        yield full_response

# Initialize session state to keep conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with Gemini-2.0 Flash (Streaming)")

# Form for user input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your message here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Save user's message in the chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    
    # Create a placeholder for the assistant's streaming response
    placeholder = st.empty()
    assistant_message = ""
    
    # Stream the response and update the placeholder in real time
    for partial in stream_response(user_input):
        assistant_message = partial
        placeholder.markdown(f"**Assistant:** {assistant_message}")
    
    # Save the complete assistant response in the chat history
    st.session_state.chat_history.append({"role": "assistant", "message": assistant_message})

# Display the entire conversation history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['message']}")
    else:
        st.markdown(f"**Assistant:** {chat['message']}")
