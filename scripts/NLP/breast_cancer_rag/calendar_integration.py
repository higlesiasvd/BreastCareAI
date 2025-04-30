"""
Google Calendar Integration for Breast Cancer Support App

This module provides a Streamlit-based interface for managing medical appointments
and integrating them directly with Google Calendar. It is designed to assist breast cancer
patients by helping them organize upcoming consultations and prepare personalized questions 
based on their clinical profile or general concerns.

Features:
- OAuth2 authentication with Google Calendar
- Creation of calendar events with customizable details (doctor, time, location)
- Option to generate personalized medical questions using an LLM (e.g., LLaMA)
- Predefined question templates by topic (diagnosis, treatment, post-treatment, quality of life)
- Persistent storage and reuse of questions saved during prior chatbot interactions

Intended for interactive clinical decision-support tools or patient-centered health assistant applications.
"""

import streamlit as st
import datetime
import os.path
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Define the scopes needed for Google Calendar
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Default timezone configuration
DEFAULT_TIMEZONE = "Europe/Madrid"  # Spain timezone

# Common timezones dictionary for the selector
COMMON_TIMEZONES = {
    "Spain (Madrid)": "Europe/Madrid",
    "UK (London)": "Europe/London",
    "Eastern US (New York)": "America/New_York",
    "Central US (Chicago)": "America/Chicago",
    "Pacific US (Los Angeles)": "America/Los_Angeles",
    "Japan (Tokyo)": "Asia/Tokyo",
    "Australia (Sydney)": "Australia/Sydney",
    "Brazil (São Paulo)": "America/Sao_Paulo",
    "India (New Delhi)": "Asia/Kolkata",
    "China (Beijing)": "Asia/Shanghai"
}
def initialize_google_calendar():
    """Initialize the connection with Google Calendar"""
    
    st.subheader("🗓️ Google Calendar Integration")
    
    # Check if credentials are already saved
    creds = None
    token_path = "token.pickle"
    
    # Try to load saved credentials
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            try:
                creds = pickle.load(token)
            except Exception as e:
                st.error(f"Error loading credentials: {e}")
    
    # If there are no valid credentials, do the authentication process
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                st.error(f"Error refreshing token: {e}")
                creds = None
        
        # If still no credentials, we need to authenticate
        if not creds:
            # Check if credentials file exists
            if not os.path.exists("credentials.json"):
                st.warning("credentials.json file not found")
                st.info("""
                To use Google Calendar integration, you need to:
                1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
                2. Enable the Google Calendar API
                3. Create OAuth 2.0 credentials
                4. Download the credentials.json file and place it in the same folder as this app
                """)
                return None
            
            try:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                # For Streamlit, we need a specific approach
                # Using a different port than Streamlit (8501)
                st.info("Please complete authentication in the browser window that will open.")
                creds = flow.run_local_server(port=8502)
                
                # Save credentials for next time
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            except Exception as e:
                st.error(f"Authentication error: {e}")
                return None
    
    try:
        # Build the service
        service = build('calendar', 'v3', credentials=creds)
        st.success("✅ Successfully connected to Google Calendar")
        return service
    except Exception as e:
        st.error(f"Error connecting to Google Calendar: {e}")
        return None


def add_questions_to_calendar(service, questions, appointment_datetime, doctor_name, appointment_duration=60, location="", timezone=None):
    """
    Add an event to the calendar with questions for the doctor
    
    Args:
        service: Google Calendar service
        questions: List of questions to include
        appointment_datetime: Date and time of the appointment
        doctor_name: Name of the doctor or specialist
        appointment_duration: Duration of the appointment in minutes
        location: Location of the appointment (optional)
    """
    
    if not service:
        st.error("No connection to Google Calendar")
        return False
    
    # Use default timezone if none specified
    if timezone is None:
        timezone = "Europe/Madrid"  # Default to Spain timezone
    
    
    # Format questions for the description
    questions_text = "\n\n".join([f"• {q}" for q in questions])
    
    # Create event body
    event = {
        'summary': f'Medical Appointment: Dr. {doctor_name}',
        'location': location,
        'description': f"""Questions prepared for the appointment:
        
{questions_text}

--
Generated by the Breast Cancer RAG App
        """,
        'start': {
            'dateTime': appointment_datetime.isoformat(),
            'timeZone': timezone,  # Adjust according to your timezone
        },
        'end': {
            'dateTime': (appointment_datetime + datetime.timedelta(minutes=appointment_duration)).isoformat(),
            'timeZone': timezone,  # Adjust according to your timezone
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                {'method': 'popup', 'minutes': 60},       # 1 hour before
                {'method': 'popup', 'minutes': 10},       # 10 minutes before
            ],
        },
        'colorId': '11'  # Red/pink for medical events (you can change the color)
    }
    
    try:
        # Create the event in the calendar
        event = service.events().insert(calendarId='primary', body=event).execute()
        return event.get('htmlLink')
    except Exception as e:
        st.error(f"Error creating event: {e}")
        return False


def generate_questions_for_profile(llm_model, patient_profile):
    """
    Generate customized questions using the LLM model based on the patient profile
    
    Args:
        llm_model: Name of the model to use (eg: "llama3:8b")
        patient_profile: Dictionary with patient profile information
    
    Returns:
        List of generated questions
    """
    try:
        import ollama
        
        stage = patient_profile.get("stage", "Pre-diagnosis")
        age = patient_profile.get("age", 45)
        preferences = patient_profile.get("preferences", ["Basic Information"])
        
        # Create prompt to generate questions based on profile
        prompt = f"""
        Generate 5 important and specific questions that a {age}-year-old patient in the '{stage}' phase 
        should ask their doctor in their next breast cancer consultation.
        
        The patient prefers information about: {', '.join(preferences)}.
        
        Questions should be clear, concise, and direct.
        Questions should be specifically adapted to the patient's '{stage}' phase.
        Each question should be on a separate line, without bullets or numbering.
        """
        
        # Generate response with Ollama
        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Process the generated questions
        generated_text = response['message']['content']
        generated_questions = [q.strip() for q in generated_text.split('\n') if q.strip() and '?' in q]
        
        return generated_questions[:5]  # Limit to 5 questions
        
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []


def calendar_management_ui():
    """
    User interface for managing Google Calendar integration
    and adding questions to calendar events
    """
    
    st.header("🗓️ Medical Appointment Management")
    
    # Initialize Google Calendar
    service = initialize_google_calendar()
    
    if not service:
        return
    
    # UI to create an event
    with st.form("calendar_form"):
        st.subheader("Schedule Medical Appointment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            appointment_date = st.date_input(
                "Appointment date",
                value=datetime.datetime.now().date() + datetime.timedelta(days=7)
            )
            
            appointment_time = st.time_input(
                "Appointment time",
                value=datetime.time(9, 0)
            )
            timezone_name = st.selectbox(
                "Timezone",
                options=list(COMMON_TIMEZONES.keys()),
                index=list(COMMON_TIMEZONES.keys()).index("Spain (Madrid)") if "Spain (Madrid)" in COMMON_TIMEZONES else 0
            )
            timezone = COMMON_TIMEZONES[timezone_name]
        
        with col2:
            doctor_name = st.text_input("Doctor/specialist name")
            appointment_duration = st.slider("Duration (minutes)", 15, 120, 60, step=15)
            location = st.text_input("Location (optional)", "")
        
        st.subheader("Questions for the consultation")
        
        # Option to automatically generate questions
        generate_questions = st.checkbox("Automatically generate questions based on my profile")
        
        # Predefined questions by category
        question_categories = {
            "Diagnosis": [
                "What type of breast cancer do I have exactly?",
                "What stage is my cancer?",
                "What additional tests do you recommend and why?",
                "How does this affect my prognosis?"
            ],
            "Treatment": [
                "What are all my treatment options?",
                "What are the side effects of each treatment?",
                "How will the treatment affect my daily life?",
                "How long will the treatment last?"
            ],
            "Post-treatment": [
                "What kind of follow-up will I need?",
                "How will I know if the cancer has returned?",
                "What can I do to reduce the risk of recurrence?",
                "When can I return to normal activities?"
            ],
            "Quality of life": [
                "What options do I have for pain management?",
                "Are there support groups you recommend?",
                "How can I manage fatigue during treatment?",
                "How can I maintain my emotional well-being?"
            ]
        }
        
        # UI to select question categories
        selected_categories = []
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("Diagnosis"):
                selected_categories.append("Diagnosis")
            if st.checkbox("Treatment"):
                selected_categories.append("Treatment")
                
        with col2:
            if st.checkbox("Post-treatment"):
                selected_categories.append("Post-treatment")
            if st.checkbox("Quality of life"):
                selected_categories.append("Quality of life")
        
        # Show and allow selection of predefined questions
        selected_questions = []
        
        if generate_questions and 'patient_profile' in st.session_state and 'llm_model' in st.session_state:
            # Use LLM to generate questions based on profile
            try:
                generated_questions = generate_questions_for_profile(
                    st.session_state.llm_model,
                    st.session_state.patient_profile
                )
                
                if generated_questions:
                    st.subheader("Questions generated for your specific situation")
                    for q in generated_questions:
                        if st.checkbox(q, key=f"gen_{q}"):
                            selected_questions.append(q)
                
            except Exception as e:
                st.error(f"Error generating questions: {e}")
                # In case of error, continue with predefined questions
        
        # Show predefined questions by category
        if selected_categories:
            st.subheader("Recommended questions by category")
            
            for category in selected_categories:
                st.write(f"**{category}:**")
                for question in question_categories[category]:
                    if st.checkbox(question, key=f"q_{category}_{question}"):
                        selected_questions.append(question)
        
        # Field for custom questions
        st.subheader("Custom questions")
        
        custom_questions = []
        for i in range(3):  # Allow up to 3 custom questions
            custom_q = st.text_input(f"Custom question {i+1}", key=f"custom_q_{i}")
            if custom_q:
                custom_questions.append(custom_q)
        
        # Add custom questions to selected questions
        selected_questions.extend(custom_questions)
        
        # Load questions saved from chat if they exist
        if 'saved_questions' in st.session_state and st.session_state.saved_questions:
            st.subheader("Questions saved from chat")
            for i, question in enumerate(st.session_state.saved_questions):
                if st.checkbox(question, key=f"saved_q_{i}"):
                    if question not in selected_questions:
                        selected_questions.append(question)
        
        # Button to create the event
        submit_button = st.form_submit_button("Add to Google Calendar")
    
    # Process form
    if submit_button:
        if not doctor_name:
            st.error("Please enter the doctor/specialist name")
        elif not selected_questions:
            st.error("Please select at least one question")
        else:
            # Combine date and time
            appointment_datetime = datetime.datetime.combine(
                appointment_date, 
                appointment_time
            )
            
            # Create event in calendar
            calendar_link = add_questions_to_calendar(
                service, 
                selected_questions, 
                appointment_datetime, 
                doctor_name, 
                appointment_duration,
                location,
                timezone #pass the selected timezone to the function
            )
            
            if calendar_link:
                st.success("✅ Event successfully created in Google Calendar")
                st.markdown(f"[View event in Google Calendar]({calendar_link})")
                
                # Show event summary
                st.subheader("Appointment Summary")
                st.write(f"📅 **Date and time:** {appointment_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"👨‍⚕️ **Doctor:** {doctor_name}")
                st.write(f"⏱️ **Duration:** {appointment_duration} minutes")
                if location:
                    st.write(f"📍 **Location:** {location}")
                
                st.subheader("Prepared Questions")
                for i, q in enumerate(selected_questions):
                    st.write(f"{i+1}. {q}")
                
                # Clear form
                st.button("Create another appointment", on_click=lambda: st.experimental_rerun())


# Main function for independent testing
if __name__ == "__main__":
    st.set_page_config(page_title="Google Calendar Integration Test", page_icon="🗓️", layout="wide")
    st.title("Google Calendar Integration Test")
    
    # Create a test patient profile
    if 'patient_profile' not in st.session_state:
        st.session_state.patient_profile = {
            "age": 45,
            "stage": "Pre-diagnosis",
            "preferences": ["Basic Information"]
        }
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "llama3:8b"
    
    # Run user interface
    calendar_management_ui()