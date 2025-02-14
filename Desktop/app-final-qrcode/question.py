import streamlit as st
import google.generativeai as genai
from typing import List
import pandas as pd
from io import BytesIO

# Initialize Gemini
def initialize_genai():
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return genai.GenerativeModel('gemini-pro')

# Updated constants
CLASSES = list(range(6, 13))  # 6th to 12th
SUBJECTS = ["Mathematics", "Science", "Physics", "Chemistry", "Biology"]
BLOOMS_LEVELS = [
    "Remember",
    "Understand", 
    "Apply",
    "Analyze",
    "Evaluate"
]


def parse_questions(text: str) -> List[dict]:
    """Parse the generated text into structured question data"""
    # This is a simple parser - you might need to adjust based on actual output format
    questions = []
    current_question = {}
    
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith(('Question', 'Q', '1.', '2.', '3.', '4.', '5.')):
            if current_question:
                questions.append(current_question)
            current_question = {'question': line}
        elif 'Bloom' in line:
            current_question['blooms_level'] = line
        elif 'Difficulty' in line:
            current_question['difficulty'] = line
            
    if current_question:
        questions.append(current_question)
    
    return questions

def get_questions(model, subject: str, class_num: str, chapter: str, 
                 blooms_levels: List[str], num_questions: int = 5):
    prompt = f"""Generate {num_questions} questions for CBSE NCERT Class {class_num} 
    {subject}, Chapter: {chapter}. The questions should be of the following Bloom's 
    taxonomy levels: {', '.join(blooms_levels)}. 

    For each question, please provide:
    1. The question
    2. Bloom's Level: [level]
    3. Difficulty: [Easy/Medium/Hard]

    Format each question clearly with these three components.
    """
    
    response = model.generate_content(prompt)
    return response.text

def export_to_excel(questions_data: List[dict]) -> BytesIO:
    """Convert questions to Excel file"""
    df = pd.DataFrame(questions_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Questions')
    output.seek(0)
    return output

def get_chapters(model, class_num: str, subject: str) -> List[str]:
    """Get list of chapters for given class and subject using Gemini"""
    prompt = f"""Based on your knowledge, provide a general list of typical chapters that might be covered in {subject} for Class {class_num}.
    Please provide only chapter names in a simple list format.
    For example:
    1. Chapter Name
    2. Chapter Name
    etc.
    
    Note: This is a general list, not specific to any particular textbook."""
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            # Fallback if no response
            return ["Please enter chapter name manually"]
        
        # Parse the response to get chapter names
        chapters = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and any(line.startswith(str(i) + '.') for i in range(1, 21)):
                chapter_name = line.split('.', 1)[1].strip()
                chapters.append(chapter_name)
        
        return chapters if chapters else ["Please enter chapter name manually"]
        
    except Exception as e:
        st.error(f"Error generating chapter list: {str(e)}")
        return ["Please enter chapter name manually"]

def main():
    st.title("NCERT Question Generator")
    st.write("Generate questions based on NCERT curriculum and Bloom's Taxonomy")
    
    # Initialize Gemini
    model = initialize_genai()
    
    # Sidebar for inputs
    with st.sidebar:
        class_num = st.selectbox("Select Class", CLASSES)
        subject = st.selectbox("Select Subject", SUBJECTS)
        
        # Get chapters dynamically using Gemini
        with st.spinner("Loading chapters..."):
            # Use session state to store chapters and prevent reloading
            if 'chapters' not in st.session_state or \
               'last_class' not in st.session_state or \
               'last_subject' not in st.session_state or \
               st.session_state.last_class != class_num or \
               st.session_state.last_subject != subject:
                
                st.session_state.chapters = get_chapters(model, str(class_num), subject)
                st.session_state.last_class = class_num
                st.session_state.last_subject = subject
            
            chapter = st.selectbox("Select Chapter", st.session_state.chapters)
        
        # Multiple selection for Bloom's levels
        blooms_selected = st.multiselect(
            "Select Bloom's Taxonomy Levels",
            BLOOMS_LEVELS,
            default=["Remember", "Understand"]
        )
        
        num_questions = st.slider("Number of Questions", 1, 10, 5)
        
        generate = st.button("Generate Questions")
    
    # Generate and display questions
    if generate and blooms_selected:
        with st.spinner("Generating questions..."):
            questions_text = get_questions(
                model,
                subject,
                str(class_num),
                chapter,
                blooms_selected,
                num_questions
            )
            
            # Display raw text
            st.write("Generated Questions:")
            st.write(questions_text)
            
            # Parse questions and create Excel download
            questions_data = parse_questions(questions_text)
            if questions_data:
                excel_file = export_to_excel(questions_data)
                
                st.download_button(
                    label="Download Questions as Excel",
                    data=excel_file,
                    file_name=f"questions_{subject}_class{class_num}_{chapter}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
