import streamlit as st
import os
from google import genai
import tempfile
import fitz  # PyMuPDF for PDF handling
import docx  # python-docx for DOCX handling
import pandas as pd
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="EeeBee Pro",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_content" not in st.session_state:
    st.session_state.document_content = ""

# App title and description
st.title("🤖 EeeBee Pro")
st.markdown("Your AI assistant powered by EeeBee Pro")

# Sidebar for document upload
with st.sidebar:
    st.header("Settings")
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", 
                                    type=["txt", "pdf", "docx", "csv"])
    
    if uploaded_file is not None:
        # Process different file types
        try:
            if uploaded_file.type == "text/plain":
                # Handle TXT files
                content = uploaded_file.read().decode("utf-8")
                st.session_state.document_content = content
                
            elif uploaded_file.type == "application/pdf":
                # Handle PDF files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                text_content = ""
                with fitz.open(tmp_path) as doc:
                    for page in doc:
                        text_content += page.get_text()
                
                os.unlink(tmp_path)  # Delete the temporary file
                st.session_state.document_content = text_content
                
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Handle DOCX files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                doc = docx.Document(tmp_path)
                text_content = "\n".join([para.text for para in doc.paragraphs])
                
                os.unlink(tmp_path)  # Delete the temporary file
                st.session_state.document_content = text_content
                
            elif uploaded_file.type == "text/csv":
                # Handle CSV files
                df = pd.read_csv(uploaded_file)
                buffer = StringIO()
                df.info(buf=buffer)
                text_content = f"CSV Summary:\n{buffer.getvalue()}\n\nFirst 5 rows:\n{df.head().to_string()}"
                st.session_state.document_content = text_content
            
            st.success(f"File '{uploaded_file.name}' processed successfully!")
            
            # Show document content preview
            with st.expander("Document Content Preview"):
                st.text(st.session_state.document_content[:1000] + 
                      ("..." if len(st.session_state.document_content) > 1000 else ""))
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Button to clear document content
    if st.session_state.document_content:
        if st.button("Clear Document"):
            st.session_state.document_content = ""
            st.success("Document cleared!")

# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate response from Gemini
def generate_gemini_response(prompt, document_content=""):
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["gemini_api_key"]
        
        # Initialize the client with the API key
        client = genai.GenerativeModel(model_name="gemini-2.0-flash", api_key=api_key)
        
        # Prepare the prompt with document content if available
        if document_content:
            full_prompt = f"Document content:\n{document_content}\n\nUser query: {prompt}\n\nPlease respond based on the document content."
        else:
            full_prompt = prompt
        
        response = client.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Generate response
        response = generate_gemini_response(prompt, st.session_state.document_content)
        
        # Update placeholder with response
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("EeeBee Pro - Powered by Google Gemini")
