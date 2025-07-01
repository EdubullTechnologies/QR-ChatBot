import streamlit as st
import pandas as pd
import altair as alt
import requests
import logging
from datetime import datetime
import json

# API endpoints
API_TEACHER_DROPDOWNS = "https://webapi.edubull.com/api/eeebee/TeacherDropdowns"
API_SUBJECT_WISE_DROPDOWNS = "https://webapi.edubull.com/api/eeebee/SubjectWiseDropdowns"
API_CONCEPT_STUDENT_STATUS = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Teacher_Topic_Wise_Weak_Concepts_AND_Students"
API_BASELINE_REPORT_BATCH = "https://webapi.edubull.com/api/eeebee/Baseline_Report_Batch_Subject"
API_BASELINE_REPORT_STUDENT = "https://webapi.edubull.com/api/eProfessor/eProf_Org_Baseline_Report_Single_Student"

def fetch_teacher_data_with_params(org_code, user_id, topic_id=None, batch_id=None):
    """Fetch teacher dropdown data with optional filtering parameters"""
    try:
        params = {
            "OrgCode": org_code,
            "UserID": user_id
        }
        
        # Always include TopicID (use 0 for getting all data)
        if topic_id is None or topic_id == 0:
            params["TopicID"] = 0  # This gets all data but empty Topics array
        elif topic_id != "all":
            params["TopicID"] = topic_id
            
        if batch_id is not None:
            params["BatchID"] = batch_id
            
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        
        # Log the request
        logging.info(f"Fetching teacher data with params: {params}")
        
        response = requests.post(API_TEACHER_DROPDOWNS, json=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Log what we received
        logging.info(f"Received {len(data.get('Topics', []))} topics from API")
        logging.info(f"Received {len(data.get('Students', []))} students from API")
        
        # Check if student data looks empty
        students = data.get('Students', [])
        if students:
            sample_student = students[0]
            logging.info(f"Sample student data: {sample_student.get('FullName', 'Unknown')} - Total: {sample_student.get('TotalConceptCount', 0)}")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching teacher data: {e}")
        return None

def fetch_concept_student_status(org_code, batch_id, topic_id):
    """Fetch concept and student performance status data"""
    try:
        params = {
            "OrgCode": org_code,
            "BatchID": batch_id,
            "TopicID": topic_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        
        logging.info(f"Fetching concept and student status with params: {params}")
        
        response = requests.post(API_CONCEPT_STUDENT_STATUS, json=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Log what we received
        logging.info(f"Received {len(data.get('Concepts', []))} concepts with performance data")
        logging.info(f"Received {len(data.get('Students', []))} students with performance data")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching concept and student status: {e}")
        return None

def fetch_batch_baseline_report(org_code, subject_id, batch_id):
    """
    Fetch batch baseline report data from the API
    
    Args:
        org_code (str): Organization code
        subject_id (int): Subject ID
        batch_id (int): Batch ID
        
    Returns:
        dict: Baseline report data or None on error
    """
    payload = {
        "OrgCode": org_code,
        "SubjectID": subject_id,
        "BatchID": batch_id
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    try:
        logging.info(f"Fetching baseline report with params: {payload}")
        response = requests.post(API_BASELINE_REPORT_BATCH, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"Received baseline data for {data.get('admin_baseline_summary', [{}])[0].get('TotalStudent', 0)} students")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching batch baseline data: {e}")
        return None

def fetch_student_baseline_report(org_code, subject_id, user_id):
    """
    Fetch individual student baseline report data
    
    Args:
        org_code (str): Organization code
        subject_id (int): Subject ID
        user_id (int): Student User ID
        
    Returns:
        dict: Student baseline report data or None on error
    """
    payload = {
        "UserID": user_id,
        "SubjectID": subject_id,
        "OrgCode": org_code
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    try:
        logging.info(f"Fetching student baseline report for UserID: {user_id}")
        response = requests.post(API_BASELINE_REPORT_STUDENT, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error fetching student baseline data: {e}")
        return None

def fetch_subject_wise_data(org_code, subject_id):
    """Fetch subject-wise dropdown data to get topics"""
    try:
        params = {
            "OrgCode": org_code,
            "SubjectID": subject_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        
        logging.info(f"Fetching subject-wise data with params: {params}")
        
        response = requests.post(API_SUBJECT_WISE_DROPDOWNS, json=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Log what we received
        logging.info(f"Received {len(data.get('Topics', []))} topics from subject API")
        
        return data
    except Exception as e:
        logging.error(f"Error fetching subject-wise data: {e}")
        return None

def display_class_overview_redesigned():
    """Redesigned class overview with hierarchical selection flow: Batch -> Subject -> Topic -> Concept"""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 Class Performance Dashboard")
    with col2:
        if st.button("🔄 Refresh", help="Reload data from server"):
            # Clear all related session state
            keys_to_clear = ['teacher_dropdown_data', 'class_overview_selections', 
                           'teacher_selections', 'current_view', 'subject_topics_cache',
                           'initial_student_data', 'all_topics_performance_cache']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Get authentication data
    org_code = st.session_state.get('stored_org_code', '012')
    user_id = st.session_state.get('user_id')
    
    if not user_id:
        st.error("User authentication required")
        return
    
    # Force data refresh on first load of class overview
    if 'current_view' not in st.session_state or st.session_state.get('current_view') != 'class_overview':
        st.session_state.current_view = 'class_overview'
        if 'teacher_dropdown_data' in st.session_state:
            del st.session_state.teacher_dropdown_data
    
    # Initialize session state for selections
    if 'class_overview_selections' not in st.session_state:
        st.session_state.class_overview_selections = {
            'batch_id': None,
            'subject_id': None,
            'topic_id': None,
            'concept_id': None,
            'student_id': None,
            'achiever_type_id': None
        }
    
    # Initialize cache for subject topics
    if 'subject_topics_cache' not in st.session_state:
        st.session_state.subject_topics_cache = {}
    
    # Fetch initial data if not in session or if we need to refresh
    if 'teacher_dropdown_data' not in st.session_state or 'teacher_data_needs_refresh' in st.session_state:
        with st.spinner("Loading class data..."):
            # Pass TopicID: 0 as suggested by the user
            data = fetch_teacher_data_with_params(org_code, user_id, topic_id=0, batch_id=None)
            if data:
                st.session_state.teacher_dropdown_data = data
                # Store the initial student data separately
                if data.get('Students'):
                    st.session_state.initial_student_data = data['Students'].copy()
                if 'teacher_data_needs_refresh' in st.session_state:
                    del st.session_state.teacher_data_needs_refresh
            else:
                st.error("Failed to load teacher data")
                return
    
    dropdown_data = st.session_state.teacher_dropdown_data
    
    # STEP 1: Batch Selection (Primary Filter)
    st.markdown("### 1️⃣ Select Class/Batch")
    batches = dropdown_data.get('Batches', [])
    
    if batches:
        batch_options = {b['BatchID']: f"{b['BatchName']} - {b['BranchName']} ({b.get('StudentCount', 0)} students)" 
                        for b in batches}
        
        selected_batch_id = st.selectbox(
            "Choose a batch to view:",
            options=list(batch_options.keys()),
            format_func=lambda x: batch_options[x],
            key="batch_selector_redesign"
        )
        
        if selected_batch_id != st.session_state.class_overview_selections['batch_id']:
            st.session_state.class_overview_selections['batch_id'] = selected_batch_id
            # Reset dependent selections
            st.session_state.class_overview_selections.update({
                'subject_id': None,
                'topic_id': None,
                'concept_id': None,
                'student_id': None
            })
            # Clear performance cache when batch changes
            if 'all_topics_performance_cache' in st.session_state:
                del st.session_state.all_topics_performance_cache
            st.rerun()
    else:
        st.warning("No batches available")
        return
    
    # Display batch overview metrics
    if selected_batch_id:
        selected_batch = next((b for b in batches if b['BatchID'] == selected_batch_id), None)
        if selected_batch:
            col1, col2, col3 = st.columns(3)
            col1.metric("Batch", selected_batch['BatchName'])
            col2.metric("Grade", selected_batch['BranchName'])
            col3.metric("Total Students", selected_batch.get('StudentCount', 0))
            
            st.markdown("---")
    
    # STEP 2: Subject Selection (after batch is selected)
    if selected_batch_id:
        st.markdown("### 2️⃣ Select Subject")
        
        # Get subjects from the data
        subjects = dropdown_data.get('Subjects', [])
        
        if subjects:
            subject_options = {"all": "📚 All Subjects (Overview)"}
            subject_options.update({
                s['SubjectID']: s['SubjectName']
                for s in subjects
            })
            
            selected_subject_id = st.selectbox(
                "Choose a subject:",
                options=list(subject_options.keys()),
                format_func=lambda x: subject_options[x],
                key="subject_selector_redesign"
            )
            
            if selected_subject_id != st.session_state.class_overview_selections['subject_id']:
                st.session_state.class_overview_selections['subject_id'] = selected_subject_id
                # Reset dependent selections
                st.session_state.class_overview_selections.update({
                    'topic_id': None,
                    'concept_id': None
                })
                st.rerun()
                
            # Display subject overview if "all" is selected
            if selected_subject_id == "all":
                display_all_subjects_overview(dropdown_data, selected_batch_id)
        else:
            st.warning("No subjects available for this batch")
            return
    
    # STEP 3: Topic Selection (after subject is selected)
    if st.session_state.class_overview_selections['subject_id'] and st.session_state.class_overview_selections['subject_id'] != "all":
        st.markdown("---")
        st.markdown("### 3️⃣ Select Topic/Chapter")
        
        selected_subject_id = st.session_state.class_overview_selections['subject_id']
        
        # Check cache first
        cache_key = f"{selected_subject_id}_{org_code}"
        if cache_key in st.session_state.subject_topics_cache:
            topics = st.session_state.subject_topics_cache[cache_key]
        else:
            # Fetch topics for the selected subject
            with st.spinner(f"Loading topics for subject..."):
                subject_data = fetch_subject_wise_data(org_code, selected_subject_id)
                if subject_data and subject_data.get('Topics'):
                    topics = subject_data['Topics']
                    # Cache the topics
                    st.session_state.subject_topics_cache[cache_key] = topics
                else:
                    topics = []
        
        if topics:
            # Add "All Topics" option
            topic_options = {"all": "📚 All Topics (Overview)"}
            topic_options.update({
                t['TopicID']: t['TopicName']
                for t in topics
            })
            
            selected_topic = st.selectbox(
                "Choose a topic to analyze:",
                options=list(topic_options.keys()),
                format_func=lambda x: topic_options[x],
                key="topic_selector_redesign"
            )
            
            if selected_topic != st.session_state.class_overview_selections['topic_id']:
                st.session_state.class_overview_selections['topic_id'] = selected_topic
                # Reset dependent selections
                st.session_state.class_overview_selections.update({
                    'concept_id': None
                })
                
                # Fetch updated data for the selected topic
                if selected_topic != "all":
                    with st.spinner("Loading topic data..."):
                        # Keep the base data for merging
                        base_data = st.session_state.get('teacher_dropdown_data', {})
                        
                        updated_data = fetch_teacher_data_with_params(
                            org_code, user_id, 
                            topic_id=selected_topic,
                            batch_id=selected_batch_id
                        )
                        
                        # Fetch performance data for concepts and students
                        performance_data = fetch_concept_student_status(
                            org_code, selected_batch_id, selected_topic
                        )
                        
                        if updated_data:
                            # Merge performance data with dropdown data
                            if performance_data:
                                # Update concepts with actual performance metrics
                                perf_concepts = performance_data.get('Concepts', [])
                                dropdown_concepts = updated_data.get('ConceptList', [])
                                
                                # Create a map of performance data by ConceptID
                                perf_map = {c['ConceptID']: c for c in perf_concepts}
                                
                                # Update dropdown concepts with performance data
                                for concept in dropdown_concepts:
                                    concept_id = concept['ConceptID']
                                    if concept_id in perf_map:
                                        perf = perf_map[concept_id]
                                        concept['AttendedStudentCount'] = perf.get('AttendedStudentCount', 0)
                                        concept['ClearedStudentCount'] = perf.get('ClearedStudentCount', 0)
                                        concept['DurationTaken_SS'] = perf.get('DurationTaken_SS', 0)
                                        concept['TotalQuestion'] = perf.get('TotalQuestion', 0)
                                        concept['AttendedQuestion'] = perf.get('AttendedQuestion', 0)
                                        concept['CorrectQuestion'] = perf.get('CorrectQuestion', 0)
                                        concept['AvgMarksPercent'] = perf.get('AvgMarksPercent', 0)
                                
                                # Update students with actual performance metrics
                                perf_students = performance_data.get('Students', [])
                                dropdown_students = updated_data.get('Students', [])
                                
                                # Create a map of performance data by UserID
                                perf_student_map = {s['UserID']: s for s in perf_students}
                                
                                # Update dropdown students with performance data
                                for student in dropdown_students:
                                    user_id = student['UserID']
                                    if user_id in perf_student_map:
                                        perf = perf_student_map[user_id]
                                        student['TotalConceptCount'] = perf.get('TotalConceptCount', 0)
                                        student['ClearedConceptCount'] = perf.get('ClearedConceptCount', 0)
                                        student['WeakConceptCount'] = perf.get('WeakConceptCount', 0)
                                        student['AvgMarksPercent'] = perf.get('AvgMarksPercent', 0)
                            
                            # Preserve student data if the new data has empty metrics
                            new_students = updated_data.get('Students', [])
                            base_students = base_data.get('Students', [])
                            
                            # Check if new student data has zero metrics
                            if new_students and base_students:
                                all_zero = all(
                                    s.get('TotalConceptCount', 0) == 0 for s in new_students
                                )
                                if all_zero:
                                    # Create a map of base student data
                                    base_map = {s['UserID']: s for s in base_students}
                                    # Update new students with base metrics
                                    for student in new_students:
                                        if student['UserID'] in base_map:
                                            base_student = base_map[student['UserID']]
                                            student['TotalConceptCount'] = base_student.get('TotalConceptCount', 0)
                                            student['ClearedConceptCount'] = base_student.get('ClearedConceptCount', 0)
                                            student['WeakConceptCount'] = base_student.get('WeakConceptCount', 0)
                            
                            st.session_state.teacher_dropdown_data = updated_data
                st.rerun()
            
            # Display topic visualizations
            if selected_topic == "all":
                display_all_topics_for_subject(dropdown_data, selected_subject_id, topics)
            else:
                display_topic_overview(dropdown_data, selected_topic)
        else:
            st.info("No topics found for this subject.")
    
    # STEP 4: Concept Selection (after topic is selected)
    if st.session_state.class_overview_selections['topic_id'] and st.session_state.class_overview_selections['topic_id'] != "all":
        st.markdown("---")
        st.markdown("### 4️⃣ Analyze Specific Concept (Optional)")
        
        concepts = dropdown_data.get('ConceptList', [])
        topic_concepts = [c for c in concepts if c['TopicID'] == st.session_state.class_overview_selections['topic_id']]
        
        if topic_concepts:
            concept_options = {"none": "None - View all concepts"}
            concept_options.update({
                c['ConceptID']: c['ConceptText']
                for c in topic_concepts
            })
            
            selected_concept = st.selectbox(
                "Select a concept for detailed analysis:",
                options=list(concept_options.keys()),
                format_func=lambda x: concept_options[x],
                key="concept_selector_redesign"
            )
            
            if selected_concept != "none" and selected_concept != st.session_state.class_overview_selections['concept_id']:
                st.session_state.class_overview_selections['concept_id'] = selected_concept
                st.rerun()
            
            if selected_concept != "none":
                display_concept_analysis(dropdown_data, selected_concept)
    
    # STEP 5: Student/Group Analysis
    st.markdown("---")
    st.markdown("### 5️⃣ Student & Group Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Individual Students", "Performance Groups", "Baseline Testing Report"])
    
    with tab1:
        display_student_analysis(dropdown_data, selected_batch_id)
    
    with tab2:
        display_achiever_groups(dropdown_data, selected_batch_id)
        
    with tab3:
        display_batch_baseline_report(dropdown_data, selected_batch_id)
    
    # Debug information at the bottom
    with st.expander("🔍 Debug Information"):
        st.write("Current selections:")
        st.write(f"- Teacher User ID: {user_id}")
        st.write(f"- Batch ID: {st.session_state.class_overview_selections.get('batch_id')}")
        st.write(f"- Subject ID: {st.session_state.class_overview_selections.get('subject_id')}")
        st.write(f"- Topic ID: {st.session_state.class_overview_selections.get('topic_id')}")
        st.write(f"- Concept ID: {st.session_state.class_overview_selections.get('concept_id')}")
        
        st.write("\nData summary:")
        st.write(f"- Batches: {len(dropdown_data.get('Batches', []))}")
        st.write(f"- Subjects: {len(dropdown_data.get('Subjects', []))}")
        st.write(f"- Topics in cache: {sum(len(v) for v in st.session_state.subject_topics_cache.values())}")
        st.write(f"- Concepts: {len(dropdown_data.get('ConceptList', []))}")
        st.write(f"- Students: {len(dropdown_data.get('Students', []))}")
        
        # Show sample student data
        students = dropdown_data.get('Students', [])
        if students:
            st.write("\nSample student data:")
            for s in students[:3]:
                st.write(f"- {s.get('FullName')}: Total={s.get('TotalConceptCount', 0)}, Cleared={s.get('ClearedConceptCount', 0)}, AvgMarks={s.get('AvgMarksPercent', 0)}%")
        
        # Show concept performance
        concepts = dropdown_data.get('ConceptList', [])
        if concepts:
            st.write("\nSample concept data:")
            for c in concepts[:3]:
                st.write(f"- {c.get('ConceptText', 'Unknown')[:30]}...: Attended={c.get('AttendedStudentCount', 0)}, Cleared={c.get('ClearedStudentCount', 0)}")
            
            # Check if we have actual performance data
            has_performance = any(c.get('AttendedStudentCount', 0) > 0 or c.get('ClearedStudentCount', 0) > 0 for c in concepts)
            st.write(f"\nPerformance data loaded: {'✅ Yes' if has_performance else '❌ No - Select a specific topic to load performance data'}")

def display_all_subjects_overview(data, batch_id):
    """Display overview of all subjects for the batch"""
    st.markdown("#### 📊 All Subjects Performance Overview")
    
    subjects = data.get('Subjects', [])
    concepts = data.get('ConceptList', [])
    students = data.get('Students', [])
    
    if not subjects:
        st.info("No subjects data available")
        return
    
    # Calculate subject-wise metrics (simplified since we might not have all topics loaded)
    subject_metrics = []
    for subject in subjects:
        subject_id = subject['SubjectID']
        subject_name = subject['SubjectName']
        
        # For now, just show basic subject info
        subject_metrics.append({
            'Subject': subject_name,
            'Subject ID': subject_id,
            'Status': 'Available'
        })
    
    # Create visualization
    df_subjects = pd.DataFrame(subject_metrics)
    
    if not df_subjects.empty:
        # Simple table display
        st.markdown("##### Available Subjects")
        st.dataframe(df_subjects, use_container_width=True)
        
        st.info("Select a specific subject to view detailed topic and concept analysis.")

def display_all_topics_for_subject(data, subject_id, topics):
    """Display all topics for a specific subject"""
    st.markdown("#### 📊 All Topics Overview")
    
    # Get cached performance data if available
    cached_performance_data = st.session_state.get('all_topics_performance_cache', {})
    
    # Get batch and org info from session state
    org_code = st.session_state.get('stored_org_code', '012')
    batch_id = st.session_state.class_overview_selections.get('batch_id')
    
    if not batch_id:
        st.error("Please select a batch first")
        return
    
    if not topics:
        st.info("No topics available")
        return
    
    # Fetch performance data for all topics if not in cache
    cache_key = f"{batch_id}_{subject_id}"
    if cache_key not in cached_performance_data:
        with st.spinner("Loading performance data for all topics..."):
            topic_performance_data = {}
            topic_students_data = {}
            
            # Fetch performance data for each topic
            for topic in topics:
                topic_id = topic['TopicID']
                perf_data = fetch_concept_student_status(org_code, batch_id, topic_id)
                
                if perf_data:
                    if perf_data.get('Concepts'):
                        topic_performance_data[topic_id] = perf_data['Concepts']
                    else:
                        topic_performance_data[topic_id] = []
                    
                    # Also store student performance data
                    if perf_data.get('Students'):
                        topic_students_data[topic_id] = perf_data['Students']
                else:
                    topic_performance_data[topic_id] = []
                    topic_students_data[topic_id] = []
            
            # Cache the performance data
            if 'all_topics_performance_cache' not in st.session_state:
                st.session_state.all_topics_performance_cache = {}
            st.session_state.all_topics_performance_cache[cache_key] = {
                'concepts': topic_performance_data,
                'students': topic_students_data
            }
            cached_performance_data = st.session_state.all_topics_performance_cache
    
    # Get the cached data for this batch and subject
    cached_data = cached_performance_data.get(cache_key, {})
    
    # Handle both old and new cache formats
    if isinstance(cached_data, dict) and 'concepts' in cached_data:
        topic_performance_data = cached_data['concepts']
        topic_students_data = cached_data['students']
    else:
        # Old format compatibility
        topic_performance_data = cached_data
        topic_students_data = {}
    
    # Calculate topic-wise metrics using actual performance data
    topic_metrics = []
    for topic in topics:
        topic_id = topic['TopicID']
        topic_concepts = topic_performance_data.get(topic_id, [])
        
        total_concepts = len(topic_concepts)
        
        if total_concepts > 0:
            total_attended = sum(c.get('AttendedStudentCount', 0) for c in topic_concepts)
            total_cleared = sum(c.get('ClearedStudentCount', 0) for c in topic_concepts)
            avg_attendance = total_attended / total_concepts
            avg_cleared = total_cleared / total_concepts
        else:
            avg_attendance = 0
            avg_cleared = 0
        
        topic_metrics.append({
            'Topic': topic['TopicName'],
            'Topic ID': topic_id,
            'Total Concepts': total_concepts,
            'Avg Students Attended': round(avg_attendance, 1),
            'Avg Students Cleared': round(avg_cleared, 1),
            'Mastery Rate': round((avg_cleared / max(avg_attendance, 1)) * 100, 1) if avg_attendance > 0 else 0
        })
    
    # Create visualization
    df_topics = pd.DataFrame(topic_metrics)
    
    if not df_topics.empty:
        # Bar chart for mastery rates
        mastery_chart = alt.Chart(df_topics).mark_bar().encode(
            y=alt.Y('Topic:N', sort='-x', title='Topics'),
            x=alt.X('Mastery Rate:Q', title='Average Mastery Rate (%)'),
            color=alt.Color('Mastery Rate:Q', 
                          scale=alt.Scale(scheme='redyellowgreen'),
                          legend=None),
            tooltip=['Topic', 'Total Concepts', 'Mastery Rate']
        ).properties(
            title='Topic-wise Mastery Rates',
            height=400
        )
        
        st.altair_chart(mastery_chart, use_container_width=True)
        
        # Display detailed table
        with st.expander("View Detailed Topic Performance"):
            # Topic Summary
            st.markdown("##### Topic Summary")
            st.dataframe(df_topics, use_container_width=True)
            
            # Student Performance by Topic
            if topic_students_data:
                st.markdown("##### Student Performance by Topic")
                
                # Create a comprehensive student performance table
                student_performance_data = []
                
                # Get all unique students
                all_students = {}
                for topic_id, students in topic_students_data.items():
                    for student in students:
                        user_id = student.get('UserID')
                        if user_id not in all_students:
                            all_students[user_id] = student.get('FullName', f'Student {user_id}')
                
                # Build performance matrix
                for user_id, student_name in all_students.items():
                    student_row = {
                        'Student Name': student_name,
                        'User ID': user_id
                    }
                    
                    total_concepts_sum = 0
                    cleared_concepts_sum = 0
                    weak_concepts_sum = 0
                    
                    for topic in topics:
                        topic_id = topic['TopicID']
                        topic_name = topic['TopicName']
                        
                        # Find this student's performance for this topic
                        topic_student_data = None
                        if topic_id in topic_students_data:
                            for s in topic_students_data[topic_id]:
                                if s.get('UserID') == user_id:
                                    topic_student_data = s
                                    break
                        
                        if topic_student_data:
                            # Get performance metrics
                            total_concepts = topic_student_data.get('TotalConceptCount', 0)
                            cleared_concepts = topic_student_data.get('ClearedConceptCount', 0)
                            weak_concepts = topic_student_data.get('WeakConceptCount', 0)
                            avg_marks = topic_student_data.get('AvgMarksPercent', 0)
                            
                            # Add to row
                            student_row[f'{topic_name[:20]}... Cleared'] = f"{cleared_concepts}/{total_concepts}"
                            student_row[f'{topic_name[:20]}... Marks'] = f"{avg_marks:.0f}%"
                            
                            # Add to totals
                            total_concepts_sum += total_concepts
                            cleared_concepts_sum += cleared_concepts
                            weak_concepts_sum += weak_concepts
                        else:
                            student_row[f'{topic_name[:20]}... Cleared'] = "N/A"
                            student_row[f'{topic_name[:20]}... Marks'] = "N/A"
                    
                    # Add overall metrics
                    if total_concepts_sum > 0:
                        student_row['Overall Progress'] = f"{cleared_concepts_sum}/{total_concepts_sum} ({(cleared_concepts_sum/total_concepts_sum*100):.0f}%)"
                    else:
                        student_row['Overall Progress'] = "No Data"
                    
                    student_performance_data.append(student_row)
                
                if student_performance_data:
                    df_student_performance = pd.DataFrame(student_performance_data)
                    
                    # Sort by student name
                    df_student_performance = df_student_performance.sort_values('Student Name')
                    
                    # Display the dataframe
                    st.dataframe(
                        df_student_performance,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info("No student performance data available for these topics.")

def display_topic_overview(data, topic_id):
    """Display overview for a specific topic"""
    topic_info = data.get('TopicInfo', {})
    concepts = data.get('ConceptList', [])
    students = data.get('Students', [])
    
    # Filter concepts for this topic
    topic_concepts = [c for c in concepts if c.get('TopicID') == topic_id]
    
    if not topic_concepts:
        st.info("No concept data available for this topic")
        return
    
    # Display topic info
    st.markdown(f"#### 📚 Topic: {topic_info.get('TopicName', 'Unknown')}")
    st.markdown(f"**Subject:** {topic_info.get('SubjectName', 'Unknown')}")
    
    # Calculate and display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_concepts = len(topic_concepts)
    total_attended = sum(c.get('AttendedStudentCount', 0) for c in topic_concepts)
    total_cleared = sum(c.get('ClearedStudentCount', 0) for c in topic_concepts)
    avg_marks = sum(c.get('AvgMarksPercent', 0) for c in topic_concepts) / max(total_concepts, 1)
    
    col1.metric("Total Concepts", total_concepts)
    col2.metric("Students Attempted", f"{total_attended // max(total_concepts, 1)}")
    col3.metric("Average Cleared", f"{total_cleared // max(total_concepts, 1)}")
    col4.metric("Avg Marks", f"{avg_marks:.1f}%")
    
    # If all metrics are 0, show a note
    if total_attended == 0 and total_cleared == 0:
        st.warning("📊 No student activity recorded for this topic yet. Students may need to attempt assessments for data to appear.")
    
    # Concept mastery visualization
    st.markdown("##### Concept-wise Performance")
    
    # Prepare data for visualization
    concept_data = []
    for concept in topic_concepts:
        attended = concept.get('AttendedStudentCount', 0)
        cleared = concept.get('ClearedStudentCount', 0)
        not_cleared = attended - cleared
        
        concept_data.extend([
            {
                'Concept': concept['ConceptText'][:30] + '...' if len(concept['ConceptText']) > 30 else concept['ConceptText'],
                'Status': 'Cleared',
                'Count': cleared
            },
            {
                'Concept': concept['ConceptText'][:30] + '...' if len(concept['ConceptText']) > 30 else concept['ConceptText'],
                'Status': 'Not Cleared',
                'Count': not_cleared
            }
        ])
    
    df_concepts = pd.DataFrame(concept_data)
    
    # Stacked bar chart
    concept_chart = alt.Chart(df_concepts).mark_bar().encode(
        y=alt.Y('Concept:N', sort='-x', title='Concepts'),
        x=alt.X('Count:Q', title='Number of Students'),
        color=alt.Color('Status:N', 
                       scale=alt.Scale(domain=['Cleared', 'Not Cleared'], 
                                     range=['#4CAF50', '#FF9800']),
                       legend=alt.Legend(title='Status')),
        tooltip=['Concept', 'Status', 'Count']
    ).properties(
        title='Concept Mastery Overview',
        height=max(400, len(topic_concepts) * 30)
    )
    
    st.altair_chart(concept_chart, use_container_width=True)

def display_concept_analysis(data, concept_id):
    """Display detailed analysis for a specific concept"""
    concepts = data.get('ConceptList', [])
    concept = next((c for c in concepts if c['ConceptID'] == concept_id), None)
    
    if not concept:
        st.warning("Concept data not found")
        return
    
    st.markdown(f"#### 🎯 Concept: {concept['ConceptText']}")
    
    # Display concept metrics
    col1, col2, col3, col4 = st.columns(4)
    
    attended = concept.get('AttendedStudentCount', 0)
    cleared = concept.get('ClearedStudentCount', 0)
    avg_marks = concept.get('AvgMarksPercent', 0)
    time_spent = concept.get('DurationTaken_SS', 0) / 60  # Convert to minutes
    
    col1.metric("Students Attempted", attended)
    col2.metric("Students Cleared", cleared)
    col3.metric("Success Rate", f"{(cleared/max(attended,1)*100):.1f}%")
    col4.metric("Avg Time", f"{time_spent:.1f} min")
    
    # Question-level analysis if available
    if concept.get('TotalQuestion', 0) > 0:
        st.markdown("##### Question Analysis")
        
        question_data = {
            'Total Questions': concept.get('TotalQuestion', 0),
            'Attempted Questions': concept.get('AttendedQuestion', 0),
            'Correct Answers': concept.get('CorrectQuestion', 0),
            'Average Marks': f"{avg_marks:.1f}%"
        }
        
        # Create a simple metrics display
        cols = st.columns(len(question_data))
        for i, (key, value) in enumerate(question_data.items()):
            cols[i].metric(key, value)

def display_student_analysis(data, batch_id):
    """Display individual student analysis"""
    students = data.get('Students', [])
    # Students from dropdown API might not have BatchID, so use all students if filtering returns empty
    batch_students = [s for s in students if s.get('BatchID') == batch_id]
    if not batch_students:
        batch_students = students  # Use all students if BatchID filtering fails
    
    if not batch_students:
        st.info("No student data available")
        return
    
    # Add info about the data being displayed
    current_topic_id = st.session_state.class_overview_selections.get('topic_id')
    if current_topic_id and current_topic_id != 'all':
        st.info("📈 Showing overall student progress across all topics. Individual topic assessment data may not be available.")
    
    # Student selector
    student_options = {s['UserID']: s['FullName'] for s in batch_students}
    selected_student_id = st.selectbox(
        "Select a student:",
        options=list(student_options.keys()),
        format_func=lambda x: student_options[x],
        key="student_selector_analysis"
    )
    
    if selected_student_id:
        student = next((s for s in batch_students if s['UserID'] == selected_student_id), None)
        if student:
            # Display student metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Check if we're viewing a specific topic
            current_topic_id = st.session_state.class_overview_selections.get('topic_id')
            if current_topic_id and current_topic_id != 'all':
                # Show both overall and topic-specific metrics
                total = student.get('TotalConceptCount', 0)
                cleared = student.get('ClearedConceptCount', 0)
                weak = student.get('WeakConceptCount', 0)
                avg_marks = student.get('AvgMarksPercent', 0)
                
                col1.metric("Total Concepts", total)
                col2.metric("Cleared", cleared)
                col3.metric("Weak", weak)
                col4.metric("Avg Marks", f"{avg_marks:.1f}%")
                
                # Show topic context
                if 'TopicTotalConcepts' in student:
                    st.markdown("##### Topic-Specific Estimates")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Topic Concepts", student.get('TopicTotalConcepts', 0))
                    col2.metric("Est. Topic Cleared", student.get('TopicClearedConcepts', 0))
                    col3.metric("Est. Topic Progress", f"{student.get('TopicProgress', 0):.1f}%")
            else:
                # Show overall metrics
                total = student.get('TotalConceptCount', 0)
                cleared = student.get('ClearedConceptCount', 0)
                weak = student.get('WeakConceptCount', 0)
                avg_marks = student.get('AvgMarksPercent', 0)
                
                col1.metric("Total Concepts", total)
                col2.metric("Concepts Cleared", cleared)
                col3.metric("Weak Concepts", weak)
                col4.metric("Average Marks", f"{avg_marks:.1f}%")
            
            # Add button to view detailed baseline report
            st.markdown("---")
            
            # Get the subject ID from session state
            subject_id = st.session_state.class_overview_selections.get('subject_id')
            org_code = st.session_state.get('stored_org_code', '012')
            
            if subject_id and subject_id != 'all':
                if st.button(f"📊 View Detailed Baseline Report for {student['FullName']}", key=f"baseline_btn_{selected_student_id}"):
                    # Display the same baseline report that students see
                    with st.spinner("Loading baseline report..."):
                        baseline_data = fetch_student_baseline_report(org_code, subject_id, selected_student_id)
                    
                    if baseline_data:
                        st.markdown("---")
                        # Use the same display function as in eeebee.py
                        display_individual_baseline_report(org_code, subject_id, selected_student_id)
                    else:
                        st.warning("No baseline data available for this student.")
            else:
                # Progress visualization for when no specific subject is selected
                if total > 0:
                    progress_data = pd.DataFrame([
                        {'Category': 'Cleared', 'Count': cleared, 'Percentage': cleared/total*100},
                        {'Category': 'Weak', 'Count': weak, 'Percentage': weak/total*100},
                        {'Category': 'Not Attempted', 'Count': total-cleared-weak, 'Percentage': (total-cleared-weak)/total*100}
                    ])
                    
                    # Pie chart
                    pie_chart = alt.Chart(progress_data).mark_arc().encode(
                        theta=alt.Theta('Count:Q'),
                        color=alt.Color('Category:N', 
                                      scale=alt.Scale(domain=['Cleared', 'Weak', 'Not Attempted'],
                                                   range=['#4CAF50', '#FF9800', '#E0E0E0'])),
                        tooltip=['Category', 'Count', alt.Tooltip('Percentage:Q', format='.1f')]
                    ).properties(
                        title=f"Concept Status for {student['FullName']}",
                        width=300,
                        height=300
                    )
                    
                    st.altair_chart(pie_chart, use_container_width=True)
                    
                st.info("Select a specific subject to view detailed baseline report for this student.")

def display_achiever_groups(data, batch_id):
    """Display performance group analysis"""
    achiever_types = data.get('AchieverTypes', [])
    students = data.get('Students', [])
    
    if not achiever_types:
        st.info("No performance group data available")
        return
    
    # Calculate student distribution across groups
    batch_students = [s for s in students if s.get('BatchID') == batch_id]
    if not batch_students:
        batch_students = students  # Use all students if BatchID filtering fails
    
    # Create categorization based on average marks and progress
    group_data = []
    for student in batch_students:
        total = student.get('TotalConceptCount', 0)
        cleared = student.get('ClearedConceptCount', 0)
        avg_marks = student.get('AvgMarksPercent', 0)
        progress = (cleared / max(total, 1)) * 100 if total > 0 else 0
        
        # Use average marks if available, otherwise use progress
        score = avg_marks if avg_marks > 0 else progress
        
        # Categorize based on score (matching the achiever types from API)
        if score >= 75.001:
            group = "Outstanding"
        elif score >= 50.001:
            group = "Achiever"
        elif score >= 35.001:
            group = "Average"
        else:
            group = "Need Improvement"
        
        group_data.append({
            'Student': student['FullName'],
            'Group': group,
            'Average Marks': avg_marks,
            'Score': score
        })
    
    df_groups = pd.DataFrame(group_data)
    
    # Group distribution chart
    group_counts = df_groups['Group'].value_counts().reset_index()
    group_counts.columns = ['Group', 'Count']
    
    # Define the order
    group_order = ["Outstanding", "Achiever", "Average", "Need Improvement"]
    
    bar_chart = alt.Chart(group_counts).mark_bar().encode(
        x=alt.X('Group:N', sort=group_order, title='Performance Group'),
        y=alt.Y('Count:Q', title='Number of Students'),
        color=alt.Color('Group:N', 
                       scale=alt.Scale(domain=group_order,
                                     range=['#4CAF50', '#8BC34A', '#FFC107', '#FF5722']),
                       legend=None),
        tooltip=['Group', 'Count']
    ).properties(
        title='Student Distribution by Performance Group',
        height=400
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Show detailed list for each group
    selected_group = st.selectbox(
        "View students in group:",
        options=group_order,
        key="group_selector"
    )
    
    if selected_group:
        group_students = df_groups[df_groups['Group'] == selected_group].sort_values('Score', ascending=False)
        if not group_students.empty:
            # Show relevant columns based on available data
            columns_to_show = ['Student']
            format_dict = {}
            
            if group_students['Average Marks'].sum() > 0:
                columns_to_show.append('Average Marks')
                format_dict = {'Average Marks': '{:.1f}%'}
            else:
                # If no average marks, show a note
                st.info("Average marks data will be available when students complete assessments.")
            
            st.dataframe(
                group_students[columns_to_show].style.format(format_dict),
                use_container_width=True
            )
        else:
            st.info(f"No students in {selected_group} group")

def display_batch_baseline_report(data, batch_id):
    """
    Display baseline report for a batch with improved visuals
    
    Args:
        data: Teacher dropdown data (for context)
        batch_id (int): Batch ID
    """
    st.markdown("#### 📊 bACEline Testing Report")
    
    # Get the selected subject ID from session state
    subject_id = st.session_state.class_overview_selections.get('subject_id')
    org_code = st.session_state.get('stored_org_code', '012')
    
    if not subject_id or subject_id == 'all':
        st.info("Please select a specific subject to view the baseline testing report.")
        return
    
    # Fetch the baseline data for this batch if not already in session state
    key_name = f"baseline_batch_{batch_id}_{subject_id}"
    
    if key_name not in st.session_state:
        with st.spinner("Loading baseline testing data..."):
            st.session_state[key_name] = fetch_batch_baseline_report(
                org_code=org_code,
                subject_id=subject_id,
                batch_id=batch_id
            )
    
    baseline_data = st.session_state[key_name]
    if not baseline_data:
        st.warning("No baseline data available for this batch and subject.")
        return
    
    # Extract data sections
    achiever_types = baseline_data.get("Achiever_Types", [])
    admin_summary = baseline_data.get("admin_baseline_summary", [{}])[0]
    students = baseline_data.get("baseline_batches_students", [])
    skills = baseline_data.get("s_skills", [])
    concepts = baseline_data.get("concept_list", [])
    
    # Add custom CSS for better metric cards
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0068c9;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Overall Performance Summary
    st.markdown("##### Overall Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Format numbers with proper decimal places
    avg_marks = f"{admin_summary.get('AvgMarksPercent', 0):.1f}%"
    
    # Metrics with better formatting
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{admin_summary.get("TotalStudent", 0)}</div>
            <div class="metric-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{admin_summary.get("AttendedStudentCount", 0)}</div>
            <div class="metric-label">Students Attended</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{admin_summary.get("StudentAtRiskCount", 0)}</div>
            <div class="metric-label">Students at Risk</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_marks}</div>
            <div class="metric-label">Average Marks</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Student Performance Distribution
    if achiever_types:
        st.markdown("---")
        st.markdown("##### Student Performance Distribution")
        
        if len(achiever_types) > 0:
            # Prepare data
            chart_data = pd.DataFrame({
                'Performance Level': [at['AchieverText'] for at in achiever_types],
                'Students': [at['StudentCount'] for at in achiever_types],
                'Color': [at.get('AchieverColor', '#0068c9') for at in achiever_types]
            })
            
            # Use a horizontal bar chart for better readability
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Students:Q', title='Number of Students'),
                y=alt.Y('Performance Level:N', sort='-x', title=None),
                color=alt.Color('Performance Level:N', 
                               scale=alt.Scale(
                                   domain=chart_data['Performance Level'].tolist(),
                                   range=['#4CAF50', '#8BC34A', '#FFC107', '#FF5722']
                               ),
                               legend=None),
                tooltip=['Performance Level', 'Students']
            ).properties(
                height=min(300, len(achiever_types) * 60)
            )
            
            st.altair_chart(chart, use_container_width=True)
    
    # Student Performance Details
    if students:
        st.markdown("---")
        st.markdown("##### Student Performance Details")
        
        # Add a selectbox to choose between table view and individual report
        view_mode = st.radio(
            "View Mode:",
            ["Summary Table", "Individual Student Report"],
            horizontal=True,
            key="baseline_view_mode"
        )
        
        if view_mode == "Summary Table":
            # Convert to dataframe for display
            df_students = pd.DataFrame(students)
            
            # Select relevant columns if they exist
            columns_to_show = []
            if 'FullName' in df_students.columns:
                columns_to_show.append('FullName')
            if 'TotalScore' in df_students.columns:
                columns_to_show.append('TotalScore')
            if 'ObtainedScore' in df_students.columns:
                columns_to_show.append('ObtainedScore')
            if 'PercentObt' in df_students.columns:
                columns_to_show.append('PercentObt')
            if 'Status' in df_students.columns:
                columns_to_show.append('Status')
            
            if columns_to_show:
                # Rename columns for better display
                column_mapping = {
                    'FullName': 'Student Name',
                    'TotalScore': 'Total Score',
                    'ObtainedScore': 'Obtained Score',
                    'PercentObt': 'Percentage',
                    'Status': 'Status'
                }
                
                df_display = df_students[columns_to_show].copy()
                df_display.columns = [column_mapping.get(col, col) for col in columns_to_show]
                
                # Format percentage column if it exists
                if 'Percentage' in df_display.columns:
                    df_display['Percentage'] = df_display['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Sort by percentage/score descending
                if 'Percentage' in df_display.columns:
                    df_display = df_display.sort_values('Percentage', ascending=False)
                
                st.dataframe(df_display, use_container_width=True)
            else:
                st.info("Detailed student performance data will be available after baseline testing.")
        
        else:  # Individual Student Report mode
            # Create a selectbox for student selection
            student_options = {s.get('UserID', i): s.get('FullName', f'Student {i}') 
                             for i, s in enumerate(students) if s.get('FullName')}
            
            if student_options:
                selected_student_id = st.selectbox(
                    "Select a student to view detailed report:",
                    options=list(student_options.keys()),
                    format_func=lambda x: student_options[x],
                    key="baseline_student_selector"
                )
                
                if selected_student_id:
                    display_individual_baseline_report(org_code, subject_id, selected_student_id)
            else:
                st.info("No students available for individual reports.")
    
    # Concept-wise Performance
    if concepts:
        st.markdown("---")
        st.markdown("##### Concept-wise Performance")
        
        with st.expander("View Concept Performance Details"):
            concept_df = pd.DataFrame(concepts)
            
            # Select relevant columns
            concept_columns = []
            if 'ConceptText' in concept_df.columns:
                concept_columns.append('ConceptText')
            if 'TotalQuestion' in concept_df.columns:
                concept_columns.append('TotalQuestion')
            if 'CorrectAnswer' in concept_df.columns:
                concept_columns.append('CorrectAnswer')
            if 'PercentObt' in concept_df.columns:
                concept_columns.append('PercentObt')
            
            if concept_columns:
                concept_display = concept_df[concept_columns].copy()
                
                # Rename columns based on what's actually present
                column_mapping = {
                    'ConceptText': 'Concept',
                    'TotalQuestion': 'Total Questions',
                    'CorrectAnswer': 'Correct Answers',
                    'PercentObt': 'Success Rate'
                }
                
                new_columns = []
                for col in concept_display.columns:
                    new_columns.append(column_mapping.get(col, col))
                concept_display.columns = new_columns
                
                # Format percentage
                if 'Success Rate' in concept_display.columns:
                    concept_display['Success Rate'] = concept_display['Success Rate'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(concept_display, use_container_width=True)

def display_blooms_taxonomy_visualization(taxonomy_list):
    """
    Display a visually appealing Bloom's Taxonomy visualization
    
    Args:
        taxonomy_list: List of dictionaries containing taxonomy performance data
    """
    # Create a mapping of taxonomy levels to user-friendly names and colors
    taxonomy_mapping = {
        "L1": {"name": "Remember", "color": "#FF6B6B", "icon": "📝", "description": "Recall facts and basic concepts"},
        "L2": {"name": "Understand", "color": "#4ECDC4", "icon": "🧩", "description": "Explain ideas or concepts"},
        "L3": {"name": "Apply", "color": "#FFD166", "icon": "🛠️", "description": "Use information in new situations"},
        "L4": {"name": "Analyze", "color": "#118AB2", "icon": "🔍", "description": "Draw connections among ideas"},
        "L5": {"name": "Evaluate", "color": "#6A0572", "icon": "⚖️", "description": "Justify a stand or decision"},
        "L6": {"name": "Create", "color": "#06D6A0", "icon": "💡", "description": "Produce new or original work"}
    }
    
    # Convert the taxonomy data to a dictionary for easier lookup
    taxonomy_dict = {}
    if taxonomy_list:
        for item in taxonomy_list:
            # The taxonomy text might be in various formats, so let's extract the level more carefully
            taxonomy_text = item.get("TaxonomyText", "")
            
            # Try to find which level this entry corresponds to
            level_code = None
            for code in taxonomy_mapping.keys():
                if code in taxonomy_text or taxonomy_mapping[code]["name"] in taxonomy_text:
                    level_code = code
                    break
            
            # If we couldn't determine the level from the text, try to infer from the position
            if not level_code and "L1" not in taxonomy_dict and "remember" in taxonomy_text.lower():
                level_code = "L1"
            elif not level_code and "L2" not in taxonomy_dict and "understand" in taxonomy_text.lower():
                level_code = "L2"
            elif not level_code and "L3" not in taxonomy_dict and "apply" in taxonomy_text.lower():
                level_code = "L3"
            elif not level_code and "L4" not in taxonomy_dict and "analyze" in taxonomy_text.lower():
                level_code = "L4"
            elif not level_code and "L5" not in taxonomy_dict and "evaluate" in taxonomy_text.lower():
                level_code = "L5"
            elif not level_code and "L6" not in taxonomy_dict and "create" in taxonomy_text.lower():
                level_code = "L6"
                
            # If we found a level code, store the data
            if level_code:
                taxonomy_dict[level_code] = {
                    "percent": item.get("PercentObt", 0),
                    "total": item.get("TotalQuestion", 0),
                    "correct": item.get("CorrectAnswer", 0)
                }
    
    st.markdown("### Bloom's Taxonomy Performance")
    
    # Create columns for the Bloom's pyramid
    cols = st.columns(6)
    
    # Display each level with its data
    for i, (level_code, level_info) in enumerate(taxonomy_mapping.items()):
        with cols[i]:
            data = taxonomy_dict.get(level_code, {"percent": 0, "total": 0, "correct": 0})
            
            # Create a metric card for each level
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: {level_info['color']}20; border-radius: 10px; border: 2px solid {level_info['color']};">
                <div style="font-size: 30px;">{level_info['icon']}</div>
                <div style="font-weight: bold; font-size: 14px; margin: 5px 0;">{level_info['name']}</div>
                <div style="font-size: 24px; font-weight: bold; color: {level_info['color']};">{data['percent']:.1f}%</div>
                <div style="font-size: 12px; color: #666;">{data['correct']}/{data['total']} correct</div>
                <div style="font-size: 10px; color: #888; margin-top: 5px;">{level_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add an overall summary
    if taxonomy_dict:
        st.markdown("---")
        
        # Calculate overall performance
        total_questions = sum(d['total'] for d in taxonomy_dict.values())
        total_correct = sum(d['correct'] for d in taxonomy_dict.values())
        overall_percent = (total_correct / max(total_questions, 1)) * 100
        
        # Create a simple bar chart showing performance across levels
        levels_data = []
        for level_code in ["L1", "L2", "L3", "L4", "L5", "L6"]:
            if level_code in taxonomy_dict:
                levels_data.append({
                    'Level': taxonomy_mapping[level_code]['name'],
                    'Performance': taxonomy_dict[level_code]['percent']
                })
        
        if levels_data:
            df_taxonomy = pd.DataFrame(levels_data)
            
            chart = alt.Chart(df_taxonomy).mark_bar().encode(
                x=alt.X('Performance:Q', scale=alt.Scale(domain=[0, 100]), title='Performance (%)'),
                y=alt.Y('Level:N', sort=['Remember', 'Understand', 'Apply', 'Analyze', 'Evaluate', 'Create'], title='Cognitive Level'),
                color=alt.Color('Performance:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                tooltip=['Level', alt.Tooltip('Performance:Q', format='.1f')]
            ).properties(
                height=250,
                title='Performance Across Bloom\'s Taxonomy Levels'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Performance", f"{overall_percent:.1f}%")
            col2.metric("Total Questions", total_questions)
            col3.metric("Correct Answers", total_correct)

def display_individual_baseline_report(org_code, subject_id, user_id):
    """
    Display individual student baseline report
    
    Args:
        org_code (str): Organization code
        subject_id (int): Subject ID
        user_id (int): Student User ID
    """
    # Fetch individual student baseline data
    with st.spinner("Loading student baseline report..."):
        baseline_data = fetch_student_baseline_report(org_code, subject_id, user_id)
    
    if not baseline_data:
        st.warning("No baseline data available for this student.")
        return
    
    # Extract data sections
    u_list = baseline_data.get("u_list", [])
    s_skills = baseline_data.get("s_skills", [])
    concept_wise_data = baseline_data.get("concept_wise_data", [])
    taxonomy_list = baseline_data.get("taxonomy_list", [])
    
    # Student Summary
    if u_list:
        user_summary = u_list[0]
        st.markdown("### 📊 Overall Performance Summary")
        
        # Create a structured layout with better spacing
        # First row - Basic Information
        st.markdown("#### Student Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p style="margin: 0; color: #333333; font-size: 14px;">Student Name</p>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #000000;">{user_summary.get("FullName", "N/A")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
                <p style="margin: 0; color: #333333; font-size: 14px;">Subject</p>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #000000;">{user_summary.get("SubjectName", "N/A")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with info_col2:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <p style="margin: 0; color: #333333; font-size: 14px;">Batch/Class</p>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #000000;">{user_summary.get("BatchName", "N/A")}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
                <p style="margin: 0; color: #333333; font-size: 14px;">Test Date</p>
                <p style="margin: 0; font-size: 18px; font-weight: bold; color: #000000;">{user_summary.get("AttendDate", "N/A")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Second section - Performance Metrics
        st.markdown("#### Performance Metrics")
        
        # Key performance indicators in a grid
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            marks_percent = user_summary.get('MarksPercent', 0)
            color = "#4CAF50" if marks_percent >= 60 else "#FF9800" if marks_percent >= 35 else "#f44336"
            st.markdown(f"""
            <div style="background-color: {color}15; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color};">
                <h2 style="margin: 0; color: {color};">{marks_percent}%</h2>
                <p style="margin: 0; color: #333333;">Overall Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col2:
            st.markdown(f"""
            <div style="background-color: #2196F315; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #2196F3;">
                <h2 style="margin: 0; color: #2196F3;">{user_summary.get("TotalQuestion", 0)}</h2>
                <p style="margin: 0; color: #333333;">Total Concepts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col3:
            st.markdown(f"""
            <div style="background-color: #4CAF5015; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #4CAF50;">
                <h2 style="margin: 0; color: #4CAF50;">{user_summary.get("CorrectQuestion", 0)}</h2>
                <p style="margin: 0; color: #333333;">Cleared</p>
            </div>
            """, unsafe_allow_html=True)
        
        with perf_col4:
            st.markdown(f"""
            <div style="background-color: #FF980015; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #FF9800;">
                <h2 style="margin: 0; color: #FF9800;">{user_summary.get("WeakConceptCount", 0)}</h2>
                <p style="margin: 0; color: #333333;">Weak Areas</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Third section - Additional Details
        st.markdown("#### Test Analysis")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            diff_percent = user_summary.get('DiffQuesPercent', 0)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #9C27B0;">{diff_percent}%</p>
                <p style="margin: 0; color: #333333; font-size: 14px;">Difficult Questions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col2:
            easy_percent = user_summary.get('EasyQuesPercent', 0)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #00BCD4;">{easy_percent}%</p>
                <p style="margin: 0; color: #333333; font-size: 14px;">Easy Questions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col3:
            duration_hh = user_summary.get("DurationHH", 0)
            duration_mm = user_summary.get("DurationMM", 0)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center;">
                <p style="margin: 0; font-size: 24px; font-weight: bold; color: #FF5722;">{duration_hh}h {duration_mm}m</p>
                <p style="margin: 0; color: #333333; font-size: 14px;">Time Taken</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Skill-wise Performance (matching student view exactly)
    st.markdown("---")
    st.markdown("### Skill-wise Performance")
    if s_skills:
        df_skills = pd.DataFrame(s_skills)

        skill_chart = alt.Chart(df_skills).mark_bar().encode(
            x=alt.X('RightAnswerPercent:Q', title='Correct %', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('SubjectSkillName:N', sort='-x'),
            tooltip=['SubjectSkillName:N', 'TotalQuestion:Q', 
                     'RightAnswerCount:Q', 'RightAnswerPercent:Q']
        ).properties(
            width=700,
            height=400,
            title="Skill-wise Correct Percentage"
        )
        st.altair_chart(skill_chart, use_container_width=True)
    else:
        st.info("No skill-wise data available.")
    
    # Concept-wise Performance (matching student view exactly)
    st.markdown("---")
    st.markdown("### Concept-wise Performance")
    if concept_wise_data:
        df_concepts = pd.DataFrame(concept_wise_data).copy()
        df_concepts["S.No."] = range(1, len(df_concepts) + 1)
        df_concepts["Concept Status"] = df_concepts["RightAnswerPercent"].apply(
            lambda x: "✅" if x == 100.0 else "❌"
        )
        df_concepts.rename(columns={"ConceptText": "Concept Name", 
                                    "BranchName": "Class"}, inplace=True)
        df_display = df_concepts[["S.No.", "Concept Name","Concept Status", "Class"]]
        st.dataframe(df_display, hide_index=True)
    else:
        st.info("No concept-wise data available.")
    
    # Bloom's Taxonomy Analysis
    st.markdown("---")
    if taxonomy_list:
        display_blooms_taxonomy_visualization(taxonomy_list)
    else:
        st.info("No taxonomy data available for Bloom's analysis.")

