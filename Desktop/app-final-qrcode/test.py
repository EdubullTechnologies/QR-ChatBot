import streamlit as st
import time
from google import genai

# Retrieve API key from Streamlit secrets
api_key = st.secrets["google"]["api_key"]

# Initialize the GenAI client with the secret API key
client = genai.Client(api_key=api_key)

def simulate_stream_response(prompt, chunk_size=20, delay=0.1):
    """
    Simulate streaming by splitting the full response text into chunks.
    """
    # Get the complete response from the API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    full_text = response.text
    
    # Yield partial text chunks to simulate streaming
    for i in range(0, len(full_text), chunk_size):
        yield full_text[:i+chunk_size]
        time.sleep(delay)

# Initialize session state to keep conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with Gemini-2.0 Flash (Simulated Streaming)")

# Form for user to submit their message
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your message here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Append the user's message to the chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    
    # Create a placeholder for the assistant's simulated streaming response
    placeholder = st.empty()
    assistant_message = ""
    
    # Simulate streaming by iterating over response chunks
    for partial in simulate_stream_response(user_input):
        assistant_message = partial
        placeholder.markdown(f"**Assistant:** {assistant_message}")
    
    # Once complete, save the full assistant message in chat history
    st.session_state.chat_history.append({"role": "assistant", "message": assistant_message})

# Display the entire conversation history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['message']}")
    else:
        st.markdown(f"**Assistant:** {chat['message']}")
