import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import logging
from typing import Dict, List, Any
import numpy as np

# Import API functions from existing modules
from cor_prod import (
    fetch_teacher_data_with_params,
    fetch_concept_student_status,
    fetch_batch_baseline_report,
    fetch_subject_wise_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page config
st.set_page_config(
    page_title="Principal Dashboard - EeeBee",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #0068c9;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
        margin-bottom: 5px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin: 20px 0;
        padding: 10px 0;
        border-bottom: 2px solid #e0e0e0;
    }
    .insight-card {
        background-color: #f5f3ff;
        border-left: 4px solid #6d28d9;
        padding: 16px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .alert-card {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 16px;
        margin: 10px 0;
        border-radius: 8px;
    }
    .success-card {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 16px;
        margin: 10px 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_view' not in st.session_state:
    st.session_state.selected_view = 'overview'

def fetch_school_data(org_code: str) -> Dict[str, Any]:
    """Fetch comprehensive school data using available APIs"""
    school_data = {
        'total_students': 0,
        'total_teachers': 0,
        'total_batches': 0,
        'subjects': [],
        'batch_performance': [],
        'weak_concepts': [],
        'student_metrics': {}
    }
    
    try:
        # Get teacher data to find all batches and subjects
        # Using a sample teacher ID - in production, you'd aggregate data from all teachers
        teacher_data = fetch_teacher_data_with_params(org_code, 1, topic_id=0)
        
        if teacher_data:
            school_data['total_batches'] = len(teacher_data.get('Batches', []))
            school_data['subjects'] = teacher_data.get('Subjects', [])
            
            # Calculate total students across batches
            for batch in teacher_data.get('Batches', []):
                school_data['total_students'] += batch.get('StudentCount', 0)
                
                # Get batch performance data
                batch_id = batch['BatchID']
                # For each subject, get baseline report
                for subject in school_data['subjects']:
                    baseline_data = fetch_batch_baseline_report(
                        org_code, 
                        subject['SubjectID'], 
                        batch_id
                    )
                    if baseline_data:
                        summary = baseline_data.get('admin_baseline_summary', [{}])[0]
                        school_data['batch_performance'].append({
                            'batch_name': batch['BatchName'],
                            'subject': subject['SubjectName'],
                            'avg_marks': summary.get('AvgMarksPercent', 0),
                            'students_at_risk': summary.get('StudentAtRiskCount', 0),
                            'total_students': summary.get('TotalStudent', 0)
                        })
    
    except Exception as e:
        logging.error(f"Error fetching school data: {e}")
    
    return school_data

def display_sidebar():
    """Display navigation sidebar"""
    st.sidebar.title("🏫 Principal Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation menu
    menu_items = {
        'overview': {'icon': '📊', 'label': "Bird's-Eye View"},
        'engagement': {'icon': '📈', 'label': 'Engagement'},
        'gaps': {'icon': '🔍', 'label': 'Learning Gaps'},
        'pedagogy': {'icon': '👩‍🏫', 'label': 'Pedagogy'},
        'atrisk': {'icon': '⚠️', 'label': 'At-Risk Students'},
        'cohorts': {'icon': '👨‍👩‍👧‍👦', 'label': 'Cohorts'},
        'teacher': {'icon': '🧑‍💼', 'label': 'Teacher Efficiency'},
        'tech': {'icon': '💻', 'label': 'Tech Adoption'},
        'nep': {'icon': '🇮🇳', 'label': 'NEP 2020'}
    }
    
    for key, item in menu_items.items():
        if st.sidebar.button(f"{item['icon']} {item['label']}", key=f"nav_{key}", use_container_width=True):
            st.session_state.selected_view = key
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        key="date_filter"
    )
    
    # Grade filter
    grade_filter = st.sidebar.multiselect(
        "Grades",
        options=['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10'],
        default=['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10'],
        key="grade_filter"
    )
    
    # Subject filter
    subject_filter = st.sidebar.multiselect(
        "Subjects",
        options=['Mathematics', 'Science', 'English', 'History'],
        default=['Mathematics', 'Science', 'English', 'History'],
        key="subject_filter"
    )

def display_overview(school_data: Dict[str, Any]):
    """Display Bird's-Eye View"""
    st.title("📊 Bird's-Eye View")
    st.markdown("Holistic overview of your institution's key performance indicators")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">2,450</div>
            <div style="color: #16a34a; font-size: 14px;">↑ 5% from last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Overall Proficiency</div>
            <div class="metric-value">82%</div>
            <div style="color: #16a34a; font-size: 14px;">↑ 3% improvement</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Engagement Rate</div>
            <div class="metric-value">91%</div>
            <div style="color: #16a34a; font-size: 14px;">↑ 2% increase</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">NEP 2020 Alignment</div>
            <div class="metric-value">75%</div>
            <div style="color: #f59e0b; font-size: 14px;">On track</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Engagement Trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        engagement_data = [85, 88, 86, 90, 92, 91]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months,
            y=engagement_data,
            mode='lines+markers',
            name='Engagement %',
            line=dict(color='#4f46e5', width=3),
            marker=dict(size=8)
        ))
        fig_trend.update_layout(
            title="Engagement Trend (Last 6 Months)",
            xaxis_title="Month",
            yaxis_title="Engagement %",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Proficiency Distribution
        proficiency_labels = ['Exceeding', 'Meeting', 'Approaching', 'Below']
        proficiency_values = [25, 57, 12, 6]
        colors = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=proficiency_labels,
            values=proficiency_values,
            marker=dict(colors=colors)
        )])
        fig_pie.update_layout(
            title="Student Proficiency Levels",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Quick Insights
    st.markdown("---")
    st.markdown("### 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="success-card">
            <strong>🎯 High Achievement:</strong> Grade 10 Mathematics showing 15% improvement in average scores
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
            <strong>📚 Resource Utilization:</strong> 85% of teachers actively using digital learning tools
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-card">
            <strong>⚠️ Attention Needed:</strong> Grade 7 Science has 18% students in at-risk category
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
            <strong>🎓 NEP Progress:</strong> Competency-based assessment adopted in 70% of classes
        </div>
        """, unsafe_allow_html=True)

def display_engagement():
    """Display Engagement Analytics"""
    st.title("📈 Real-Time Student Engagement")
    st.markdown("Track student focus and participation trends across grades, subjects, and time")
    
    # Engagement by Subject
    subjects = ['Mathematics', 'Science', 'History', 'English']
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    
    # Create sample data
    engagement_data = []
    for subject in subjects:
        for i, week in enumerate(weeks):
            base_value = np.random.randint(85, 95)
            engagement_data.append({
                'Week': week,
                'Subject': subject,
                'Engagement': base_value + np.random.randint(-5, 5)
            })
    
    df_engagement = pd.DataFrame(engagement_data)
    
    fig_line = px.line(df_engagement, x='Week', y='Engagement', color='Subject',
                       title='Engagement by Subject (Last 30 Days)',
                       markers=True)
    fig_line.update_layout(height=400)
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Engagement by Grade
    grades = ['Grade 6', 'Grade 7', 'Grade 8', 'Grade 9', 'Grade 10']
    engagement_by_grade = [88, 92, 90, 93, 87]
    
    fig_bar = go.Figure(data=[
        go.Bar(x=engagement_by_grade, y=grades, orientation='h',
               marker_color='#4f46e5')
    ])
    fig_bar.update_layout(
        title='Average Engagement by Grade',
        xaxis_title='Engagement %',
        yaxis_title='Grade',
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

def display_learning_gaps():
    """Display Learning Gaps Analysis"""
    st.title("🔍 Systemic Learning Gaps & Hotspots")
    st.markdown("Identify school-wide concepts and skills where students are struggling")
    
    # Top Misconceptions
    misconceptions = {
        "Ohm's Law": 35,
        "Newton's 2nd Law": 31,
        "Algebraic Variables": 28,
        "Past Tense": 25,
        "Cell Mitosis": 22
    }
    
    fig_misconceptions = go.Figure(data=[
        go.Bar(x=list(misconceptions.values()), 
               y=list(misconceptions.keys()),
               orientation='h',
               marker_color='#dc2626')
    ])
    fig_misconceptions.update_layout(
        title='Top 5 Misconception Hotspots',
        xaxis_title='Students with Misconception (%)',
        height=400
    )
    st.plotly_chart(fig_misconceptions, use_container_width=True)
    
    # Struggling Topics Table
    st.markdown("### 📊 Topics Requiring Attention")
    struggling_topics = pd.DataFrame({
        'Topic': ['Trigonometry', 'Chemical Equations', 'Mughal Empire', 'Figurative Language'],
        'Subject': ['Mathematics', 'Science', 'History', 'English'],
        'Struggling %': [18, 15, 12, 10],
        'Recommended Action': [
            'Schedule remedial sessions',
            'Provide visual aids and lab demos',
            'Use interactive timeline activities',
            'Implement peer reading groups'
        ]
    })
    st.dataframe(struggling_topics, use_container_width=True)
    
    # AI Insights
    if st.button("✨ Generate AI Insights", key="gaps_insights"):
        with st.spinner("Analyzing learning gaps..."):
            st.markdown("""
            <div class="insight-card">
            <strong>AI-Generated Insights:</strong><br><br>
            • <strong>Pattern Detected:</strong> Science concepts requiring abstract thinking show highest failure rates<br>
            • <strong>Recommendation:</strong> Implement visual learning tools and hands-on experiments<br>
            • <strong>Focus Area:</strong> Grade 7-8 students need additional support in transitioning from concrete to abstract concepts<br>
            • <strong>Success Strategy:</strong> Schools using interactive simulations show 23% better concept retention
            </div>
            """, unsafe_allow_html=True)

def display_pedagogy():
    """Display Pedagogical Effectiveness"""
    st.title("👩‍🏫 Pedagogical Effectiveness")
    st.markdown("Compare the impact of different teaching strategies on student performance")
    
    # Teaching Methods Comparison
    methods_data = {
        'Subject': ['Mathematics', 'Mathematics', 'Mathematics',
                   'Science', 'Science', 'Science',
                   'History', 'History', 'History'],
        'Method': ['Project-Based', 'Direct Instruction', 'Flipped Classroom'] * 3,
        'Average Score': [85, 82, 88, 92, 81, 86, 88, 90, 80]
    }
    
    df_methods = pd.DataFrame(methods_data)
    
    fig_methods = px.bar(df_methods, x='Subject', y='Average Score', 
                        color='Method', barmode='group',
                        title='Teaching Method Effectiveness by Subject')
    fig_methods.update_layout(height=400)
    st.plotly_chart(fig_methods, use_container_width=True)
    
    # Best Practices
    st.markdown("### 🏆 Identified Best Practices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-card">
            <strong>Science:</strong> Project-Based Learning shows 11% higher retention<br>
            <em>Recommendation: Expand to other subjects</em>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
            <strong>Mathematics:</strong> Flipped Classroom approach improving problem-solving skills<br>
            <em>Recommendation: Train more teachers in this method</em>
        </div>
        """, unsafe_allow_html=True)

def display_at_risk_students():
    """Display At-Risk Student Alerts"""
    st.title("⚠️ At-Risk Student Alerts")
    st.markdown("Early-warning system to identify and support students before they fall behind")
    
    # Risk Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #dc2626;">
            <div class="metric-label">High Risk</div>
            <div class="metric-value" style="color: #dc2626;">23</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #f59e0b;">
            <div class="metric-label">Medium Risk</div>
            <div class="metric-value" style="color: #f59e0b;">45</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #3b82f6;">
            <div class="metric-label">Low Risk</div>
            <div class="metric-value" style="color: #3b82f6;">67</div>
        </div>
        """, unsafe_allow_html=True)
    
    # At-Risk Students Table
    st.markdown("### 📋 Students Requiring Immediate Attention")
    
    at_risk_data = pd.DataFrame({
        'Student Name': ['Rohan Sharma', 'Priya Singh', 'Amit Kumar', 'Sneha Patel', 'Raj Verma'],
        'Grade': [8, 7, 9, 6, 8],
        'Risk Level': ['High', 'High', 'Medium', 'Low', 'Medium'],
        'Primary Concern': ['Low Engagement', 'Failing Assessments', 'Declining Scores', 
                           'Low Participation', 'Irregular Attendance'],
        'Days Since Last Activity': [5, 3, 7, 2, 4],
        'Action Status': ['Pending', 'Parent Meeting Scheduled', 'Counseling Initiated', 
                         'Monitoring', 'Teacher Intervention']
    })
    
    # Apply color coding
    def highlight_risk(val):
        if val == 'High':
            return 'background-color: #fef2f2; color: #dc2626;'
        elif val == 'Medium':
            return 'background-color: #fffbeb; color: #f59e0b;'
        else:
            return 'background-color: #eff6ff; color: #3b82f6;'
    
    styled_df = at_risk_data.style.applymap(highlight_risk, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📧 Draft Parent Emails", key="draft_emails"):
            st.info("Email templates generated for high-risk students")
    with col2:
        if st.button("📊 Generate Intervention Plan", key="intervention_plan"):
            st.info("Personalized intervention plans created")
    with col3:
        if st.button("📅 Schedule Meetings", key="schedule_meetings"):
            st.info("Meeting requests sent to respective teachers")

def display_cohorts():
    """Display Cohort Performance"""
    st.title("👨‍👩‍👧‍👦 Student Cohort Performance")
    st.markdown("Compare performance across different student groups to ensure equitable progress")
    
    # Radar Chart for Cohort Comparison
    categories = ['Math', 'Science', 'History', 'English', 'Arts']
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[92, 88, 85, 90, 95],
        theta=categories,
        fill='toself',
        name='Gifted Program',
        line_color='#059669'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[80, 82, 84, 85, 78],
        theta=categories,
        fill='toself',
        name='General Cohort',
        line_color='#2563eb'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=[72, 70, 75, 78, 74],
        theta=categories,
        fill='toself',
        name='Remedial Support',
        line_color='#d97706'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Cohort Performance Comparison Across Subjects"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Performance Gap Analysis
    st.markdown("### 📊 Performance Gap Analysis")
    
    gap_data = pd.DataFrame({
        'Metric': ['Average Score', 'Concept Mastery', 'Assignment Completion', 'Class Participation'],
        'Gifted - General Gap': [12, 15, 5, 8],
        'General - Remedial Gap': [8, 12, 10, 7]
    })
    
    fig_gap = px.bar(gap_data, x='Metric', y=['Gifted - General Gap', 'General - Remedial Gap'],
                     title='Performance Gaps Between Cohorts (%)',
                     barmode='group')
    st.plotly_chart(fig_gap, use_container_width=True)

def display_teacher_efficiency():
    """Display Teacher Efficiency Metrics"""
    st.title("🧑‍💼 Teacher Efficiency & Development")
    st.markdown("Insights into teacher workload, time saved through automation, and professional growth")
    
    # Time Saved Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for time saved
        activities = ['Grading', 'Lesson Prep', 'Reporting', 'Analysis']
        hours_saved = [5, 3, 2, 1.5]
        colors = ['#047857', '#059669', '#10b981', '#34d399']
        
        fig_time = go.Figure(data=[
            go.Bar(x=activities, y=hours_saved, marker_color=colors)
        ])
        fig_time.update_layout(
            title='Hours Saved per Week via Automation',
            yaxis_title='Hours',
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Professional Development Needs
        st.markdown("### 📚 Identified Professional Development Needs")
        
        pd_needs = [
            {'icon': '💻', 'text': 'Advanced Digital Tool Integration', 'priority': 'High'},
            {'icon': '🎨', 'text': 'Differentiated Instruction Strategies', 'priority': 'Medium'},
            {'icon': '📊', 'text': 'Data Interpretation for Classroom Action', 'priority': 'High'},
            {'icon': '🧠', 'text': 'Social-Emotional Learning Techniques', 'priority': 'Medium'}
        ]
        
        for need in pd_needs:
            color = '#dc2626' if need['priority'] == 'High' else '#f59e0b'
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {color};">
                <span style="font-size: 20px;">{need['icon']}</span>
                <span style="margin-left: 10px;">{need['text']}</span>
                <span style="float: right; color: {color}; font-weight: bold;">{need['priority']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Teacher Performance Overview
    st.markdown("---")
    st.markdown("### 👨‍🏫 Teacher Performance Overview")
    
    teacher_data = pd.DataFrame({
        'Teacher': ['Ms. Sharma', 'Mr. Patel', 'Ms. Gupta', 'Mr. Singh', 'Ms. Verma'],
        'Student Progress': [85, 78, 92, 88, 75],
        'Digital Tool Usage': [95, 70, 88, 92, 65],
        'Parent Engagement': [80, 85, 90, 75, 88],
        'Overall Rating': [4.5, 4.0, 4.8, 4.3, 3.9]
    })
    
    st.dataframe(teacher_data.style.background_gradient(cmap='RdYlGn', subset=['Student Progress', 'Digital Tool Usage', 'Parent Engagement']),
                use_container_width=True)

def display_tech_adoption():
    """Display Technology Adoption Metrics"""
    st.title("💻 Technology Adoption & Impact")
    st.markdown("Monitor tool usage and its correlation with academic outcomes")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Scatter plot for correlation
        np.random.seed(42)
        usage_hours = np.random.uniform(2, 10, 50)
        score_improvement = usage_hours * 1.2 + np.random.normal(0, 1.5, 50)
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=usage_hours,
            y=score_improvement,
            mode='markers',
            marker=dict(
                size=10,
                color=score_improvement,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'Student {i+1}' for i in range(50)],
            hovertemplate='%{text}<br>Usage: %{x:.1f} hrs<br>Improvement: %{y:.1f}%'
        ))
        
        # Add trendline
        z = np.polyfit(usage_hours, score_improvement, 1)
        p = np.poly1d(z)
        fig_scatter.add_trace(go.Scatter(
            x=np.sort(usage_hours),
            y=p(np.sort(usage_hours)),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(
            title='Correlation: Tool Usage vs Score Improvement',
            xaxis_title='Weekly Tool Usage (Hours)',
            yaxis_title='Score Improvement (%)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Tool adoption rates
        tools = ['Tech Book', 'SlateIQ', 'Vishwakar Lab']
        adoption_rates = [95, 88, 75]
        colors = ['#1d4ed8', '#4f46e5', '#7c3aed']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=tools,
            values=adoption_rates,
            hole=.5,
            marker_colors=colors
        )])
        fig_donut.update_layout(
            title='Tool Adoption Rate (%)',
            height=400
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Impact Metrics
    st.markdown("### 📈 Technology Impact Metrics")
    
    impact_metrics = pd.DataFrame({
        'Tool': ['Tech Book', 'SlateIQ', 'Vishwakar Lab'],
        'Active Users': [2205, 2037, 1688],
        'Avg. Usage (hrs/week)': [4.5, 3.2, 2.8],
        'Score Improvement': ['+12%', '+8%', '+15%'],
        'Engagement Boost': ['+18%', '+14%', '+22%']
    })
    
    st.dataframe(impact_metrics, use_container_width=True)

def display_nep_compliance():
    """Display NEP 2020 Compliance"""
    st.title("🇮🇳 NEP 2020 Compliance")
    st.markdown("Track alignment with key mandates of the National Education Policy 2020")
    
    # NEP Progress Bars
    nep_metrics = [
        {'title': 'Competency-Based Learning', 'value': 85, 'color': '#14b8a6'},
        {'title': 'Experiential Learning (Vishwakar Lab)', 'value': 70, 'color': '#6366f1'},
        {'title': 'Technology Integration', 'value': 90, 'color': '#3b82f6'},
        {'title': 'Formative Assessment Shift', 'value': 65, 'color': '#f59e0b'},
        {'title': 'Multilingual Education', 'value': 55, 'color': '#8b5cf6'},
        {'title': 'Vocational Education', 'value': 45, 'color': '#ec4899'}
    ]
    
    for metric in nep_metrics:
        st.markdown(f"**{metric['title']}**")
        st.progress(metric['value'] / 100)
        st.markdown(f"<div style='text-align: right; color: {metric['color']}; font-weight: bold; margin-top: -15px;'>{metric['value']}%</div>", 
                   unsafe_allow_html=True)
        st.markdown("")
    
    # Implementation Roadmap
    st.markdown("### 🗺️ Implementation Roadmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-card">
            <strong>✅ Completed:</strong><br>
            • Competency framework implementation<br>
            • Digital infrastructure setup<br>
            • Teacher training programs initiated
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-card">
            <strong>🚧 In Progress:</strong><br>
            • Vocational course integration<br>
            • Multilingual content development<br>
            • Parent engagement programs
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Principal Dashboard"""
    
    # Get organization code from session state
    org_code = st.session_state.get('stored_org_code', '012')
    
    # Display sidebar
    display_sidebar()
    
    # Fetch school data
    with st.spinner("Loading school data..."):
        school_data = fetch_school_data(org_code)
    
    # Display selected view
    view = st.session_state.selected_view
    
    if view == 'overview':
        display_overview(school_data)
    elif view == 'engagement':
        display_engagement()
    elif view == 'gaps':
        display_learning_gaps()
    elif view == 'pedagogy':
        display_pedagogy()
    elif view == 'atrisk':
        display_at_risk_students()
    elif view == 'cohorts':
        display_cohorts()
    elif view == 'teacher':
        display_teacher_efficiency()
    elif view == 'tech':
        display_tech_adoption()
    elif view == 'nep':
        display_nep_compliance()

if __name__ == "__main__":
    main()
