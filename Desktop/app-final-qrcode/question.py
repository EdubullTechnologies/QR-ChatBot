import streamlit as st
import google.generativeai as genai
from typing import List
import pandas as pd
from io import BytesIO

# Initialize Gemini
def initialize_genai():
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return genai.GenerativeModel('gemini-2.0-flash')

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
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# Add new constants for question types
MATH_QUESTION_TYPES = [
    "Numerical Problems",
    "Word Problems",
    "Logical Reasoning",
    "Proof-Based Questions"
]

SCIENCE_QUESTION_TYPES = [
    "Numerical Problems",
    "Word Problems",
    "Assertion and Reason",
    "Process-Based Questions"
]

def parse_questions(text: str) -> List[dict]:
    """Parse the generated text into structured question data"""
    questions = []
    current_question = {}
    current_options = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_question:
                current_question['options'] = current_options
                questions.append(current_question)
                current_question = {}
                current_options = []
        elif line.startswith('Question:'):
            if current_question:
                current_question['options'] = current_options
                questions.append(current_question)
                current_options = []
            current_question = {'question': line.replace('Question:', '').strip()}
        elif line.startswith(('a)', 'b)', 'c)', 'd)')):
            current_options.append(line)
        elif line.startswith('Correct Answer:'):
            current_question['correct_answer'] = line.replace('Correct Answer:', '').strip()
        elif line.startswith("Bloom's Level:"):
            current_question['blooms_level'] = line.replace("Bloom's Level:", '').strip()
        elif line.startswith('Difficulty:'):
            current_question['difficulty'] = line.replace('Difficulty:', '').strip()
        elif line.startswith('Question Type:'):
            current_question['question_type'] = line.replace('Question Type:', '').strip()
    
    # Add the last question
    if current_question:
        current_question['options'] = current_options
        questions.append(current_question)
    
    return questions

def get_questions(model, subject: str, class_num: str, chapter: str, 
                 blooms_levels: List[str], difficulty_levels: List[str],
                 question_types: List[str], num_questions: int = 5):
    """Updated to include question types and MCQ format"""
    prompt = f"""Generate {num_questions} multiple choice questions about {chapter} suitable for Class {class_num} 
    {subject} students. 
    
    Requirements:
    - Bloom's taxonomy levels: {', '.join(blooms_levels)}
    - Difficulty levels: {', '.join(difficulty_levels)}
    - Question types: {', '.join(question_types)}
    
    For each question, provide in this exact format:
    Question: [The question text]
    Options:
    a) [option a]
    b) [option b]
    c) [option c]
    d) [option d]
    Correct Answer: [a/b/c/d]
    Bloom's Level: [specific level]
    Difficulty: [specific difficulty]
    Question Type: [specific type]
    
    Ensure each question:
    1. Is in MCQ format with 4 options
    2. Matches the requested difficulty, Bloom's level, and question type
    3. Has clear and distinct options
    4. For numerical problems, include step-wise solution
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text if response.text else "Unable to generate questions. Please try again."
    except Exception as e:
        return f"Error generating questions: {str(e)}"

def export_to_excel(questions_data: List[dict]) -> BytesIO:
    """Convert questions to Excel file with proper formatting"""
    # Create a DataFrame with all the question details
    df = pd.DataFrame({
        'Question': [q.get('question', '') for q in questions_data],
        'Option A': [q.get('options', [])[0].replace('a) ', '') if len(q.get('options', [])) > 0 else '' for q in questions_data],
        'Option B': [q.get('options', [])[1].replace('b) ', '') if len(q.get('options', [])) > 1 else '' for q in questions_data],
        'Option C': [q.get('options', [])[2].replace('c) ', '') if len(q.get('options', [])) > 2 else '' for q in questions_data],
        'Option D': [q.get('options', [])[3].replace('d) ', '') if len(q.get('options', [])) > 3 else '' for q in questions_data],
        'Correct Answer': [q.get('correct_answer', '') for q in questions_data],
        "Bloom's Level": [q.get('blooms_level', '') for q in questions_data],
        'Difficulty': [q.get('difficulty', '') for q in questions_data],
        'Question Type': [q.get('question_type', '') for q in questions_data]
    })
    
    # Create Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Questions')
        
        # Get the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Questions']
        
        # Auto-adjust columns width
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
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
            if 'chapters' not in st.session_state or \
               'last_class' not in st.session_state or \
               'last_subject' not in st.session_state or \
               st.session_state.last_class != class_num or \
               st.session_state.last_subject != subject:
                
                st.session_state.chapters = get_chapters(model, str(class_num), subject)
                st.session_state.last_class = class_num
                st.session_state.last_subject = subject
            
            chapter = st.selectbox("Select Chapter", st.session_state.chapters)
        
        # Dynamic question type selection based on subject
        question_types = MATH_QUESTION_TYPES if subject == "Mathematics" else SCIENCE_QUESTION_TYPES
        question_types_selected = st.multiselect(
            "Select Question Types",
            question_types,
            default=[question_types[0]]
        )
        
        blooms_selected = st.multiselect(
            "Select Bloom's Taxonomy Levels",
            BLOOMS_LEVELS,
            default=["Remember", "Understand"]
        )
        
        difficulty_selected = st.multiselect(
            "Select Difficulty Levels",
            DIFFICULTY_LEVELS,
            default=["Easy", "Medium"]
        )
        
        num_questions = st.slider("Number of Questions", 1, 10, 5)
        
        generate = st.button("Generate Questions")
    
    # Generate and display questions
    if generate and blooms_selected and difficulty_selected and question_types_selected:
        with st.spinner("Generating questions..."):
            questions_text = get_questions(
                model,
                subject,
                str(class_num),
                chapter,
                blooms_selected,
                difficulty_selected,
                question_types_selected,
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
            else:
                st.error("No questions were generated. Please try again.")

if __name__ == "__main__":
    main()
