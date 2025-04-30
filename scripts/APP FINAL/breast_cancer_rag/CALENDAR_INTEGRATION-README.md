# Google Calendar Integration Module for Healthcare Applications

## Introduction

The Google Calendar Integration Module provides a robust framework for managing medical appointments and consultations within healthcare applications. This module enables users to seamlessly schedule appointments, prepare consultation questions, and integrate them directly with Google Calendar, making it easier to manage healthcare-related events.

Designed with healthcare workflows in mind, the module offers specialized features for medical appointment management, including automatic question generation based on patient profiles and a comprehensive interface for preparing for consultations.

## Table of Contents

1. [Core Features](#core-features)
2. [Technical Architecture](#technical-architecture)
3. [Google OAuth Integration](#google-oauth-integration)
4. [Calendar Event Management](#calendar-event-management)
5. [Question Generation](#question-generation)
6. [User Interface Components](#user-interface-components)
7. [Timezone Support](#timezone-support)
8. [Integration with Patient Profiles](#integration-with-patient-profiles)
9. [Example Usage](#example-usage)
10. [Technical Reference](#technical-reference)

## Core Features

The Google Calendar Integration Module offers several key capabilities:

1. **OAuth Authentication**: Secure connection with Google Calendar API using OAuth 2.0
2. **Appointment Scheduling**: Create, view, and manage medical appointments
3. **Question Preparation**: Tool for preparing questions for medical consultations
4. **AI-Generated Questions**: Automatic generation of relevant questions based on patient profile
5. **Multi-timezone Support**: Support for scheduling appointments across different timezones
6. **Customizable Reminders**: Configurable email and pop-up reminders for appointments
7. **Streamlit Integration**: Ready-to-use UI components for Streamlit applications

## Technical Architecture

The module consists of three primary components:

1. **Authentication System**: Handles Google OAuth flow and token management
2. **Calendar API Wrapper**: Manages interactions with the Google Calendar API
3. **UI Components**: Provides Streamlit interface elements for appointment management

The architecture prioritizes security, user experience, and reliability, with comprehensive error handling to ensure robust performance across different usage scenarios.

## Google OAuth Integration

The module implements a complete OAuth 2.0 authentication flow for Google Calendar:

```python
def initialize_google_calendar():
    """Initialize the connection with Google Calendar"""
    
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
            # Authentication flow implementation...
            
    try:
        # Build the service
        service = build('calendar', 'v3', credentials=creds)
        st.success("✅ Successfully connected to Google Calendar")
        return service
    except Exception as e:
        st.error(f"Error connecting to Google Calendar: {e}")
        return None
```

This implementation handles:

1. Loading existing credentials from pickled storage
2. Refreshing expired tokens when possible
3. Initiating new OAuth flows when needed
4. Saving tokens for future sessions
5. Comprehensive error handling

The module uses a specific port (8502) for OAuth redirection to avoid conflicts with Streamlit's default port, ensuring a smooth authentication experience.

## Calendar Event Management

The module provides specialized functions for creating and managing medical appointments:

```python
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
        timezone: Timezone for the appointment (optional)
    """
```

Key features include:

1. Structured formatting of questions in the event description
2. Automatic reminder configuration (1 day before via email, 1 hour and 10 minutes before via popup)
3. Color-coding of events for easy identification (red/pink for medical appointments)
4. Support for different timezones
5. Comprehensive metadata including doctor name, location, and appointment duration

## Question Generation

A unique feature of this module is the ability to automatically generate relevant medical consultation questions based on the patient's profile:

```python
def generate_questions_for_profile(llm_model, patient_profile):
    """
    Generate customized questions using the LLM model based on the patient profile
    
    Args:
        llm_model: Name of the model to use (eg: "llama3:8b")
        patient_profile: Dictionary with patient profile information
    
    Returns:
        List of generated questions
    """
```

The function:

1. Uses Ollama to access local LLM models
2. Creates a context-aware prompt based on patient age, treatment stage, and information preferences
3. Processes the response to extract clearly formatted questions
4. Returns a curated list of relevant questions for the medical appointment

This provides personalized guidance for patients, helping them make the most of their consultation time based on their specific medical situation.

## User Interface Components

The module provides a complete UI for appointment management via the `calendar_management_ui()` function:

```python
def calendar_management_ui():
    """
    User interface for managing Google Calendar integration
    and adding questions to calendar events
    """
```

This function creates a comprehensive interface with:

1. Date and time selection with timezone support
2. Doctor/specialist name and appointment duration inputs
3. Location field
4. Automatic question generation based on patient profile
5. Category-based predefined questions (Diagnosis, Treatment, Post-treatment, Quality of life)
6. Support for custom questions
7. Integration with questions saved from the chat interface
8. Direct submission to Google Calendar with visual confirmation

## Timezone Support

The module includes robust timezone handling for global applicability:

```python
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
```

This allows users to:

1. Select their appropriate timezone from a human-readable list
2. Create calendar events that properly account for timezone differences
3. Avoid confusion when managing appointments across different regions

## Integration with Patient Profiles

The module seamlessly integrates with the patient profile system, accessing information stored in Streamlit's session state:

```python
if generate_questions and 'patient_profile' in st.session_state and 'llm_model' in st.session_state:
    # Use LLM to generate questions based on profile
    try:
        generated_questions = generate_questions_for_profile(
            st.session_state.llm_model,
            st.session_state.patient_profile
        )
```

This integration allows for:

1. Contextually appropriate question generation
2. Personalized appointment management
3. Consistent user experience across the application

## Example Usage

### Basic Calendar Integration

```python
import streamlit as st
from calendar_integration import initialize_google_calendar, add_questions_to_calendar
import datetime

st.title("Medical Appointment Scheduler")

# Initialize Google Calendar
service = initialize_google_calendar()

if service:
    # Create a simple appointment form
    with st.form("simple_appointment"):
        doctor_name = st.text_input("Doctor name")
        appointment_date = st.date_input("Appointment date")
        appointment_time = st.time_input("Appointment time")
        questions = [
            "What are my treatment options?",
            "What are the potential side effects?"
        ]
        
        if st.form_submit_button("Schedule Appointment"):
            # Combine date and time
            appointment_datetime = datetime.datetime.combine(
                appointment_date, 
                appointment_time
            )
            
            # Create calendar event
            link = add_questions_to_calendar(
                service, 
                questions, 
                appointment_datetime, 
                doctor_name
            )
            
            if link:
                st.success("Appointment scheduled!")
                st.markdown(f"[View in Google Calendar]({link})")
```

### Full UI Integration

```python
import streamlit as st
from calendar_integration import calendar_management_ui

st.title("Medical Appointment Manager")

# Set up patient profile in session state
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "age": 45,
        "stage": "In treatment",
        "preferences": ["Treatment options", "Side effects", "Quality of life"]
    }

# Run the complete calendar management UI
calendar_management_ui()
```

## Technical Reference

### Core Functions

```python
def initialize_google_calendar():
    """
    Initialize the connection with Google Calendar
    
    Returns:
        Google Calendar service object or None if authentication fails
    """

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
        timezone: Timezone for the appointment (optional)
        
    Returns:
        URL to the created Google Calendar event, or False if failed
    """

def generate_questions_for_profile(llm_model, patient_profile):
    """
    Generate customized questions using the LLM model based on the patient profile
    
    Args:
        llm_model: Name of the model to use (eg: "llama3:8b")
        patient_profile: Dictionary with patient profile information
    
    Returns:
        List of generated questions
    """

def calendar_management_ui():
    """
    User interface for managing Google Calendar integration
    and adding questions to calendar events
    """
```

### Dependencies

- `google-auth-oauthlib`: For OAuth 2.0 flow implementation
- `google-api-python-client`: For Google Calendar API access
- `streamlit`: For UI components
- `ollama`: For LLM-based question generation

### Required Permissions

- `https://www.googleapis.com/auth/calendar.events`: Permission to create and manage calendar events

---

This Calendar Integration Module enhances the healthcare experience by streamlining the management of medical appointments and preparing patients for productive consultations. With its focus on personalization, usability, and integration with modern AI capabilities, it represents a significant advancement in digital health management tools.
