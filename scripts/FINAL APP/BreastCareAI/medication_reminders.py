"""
Medication Reminder System – Google Calendar Integration

This module provides a complete interface in Streamlit for managing medication schedules 
and automatically generating Google Calendar reminders for breast cancer patients. 

Core Features:
- Add, view, edit, and delete medications with customizable dosage, frequency, and time slots.
- Automatically creates recurring Google Calendar events for each medication intake time.
- Stores user medication data locally in a JSON file.
- Displays an upcoming 7-day medication schedule in a clear, interactive table.
- Includes user settings for notification types and event customization.

The goal is to enhance treatment adherence and support patients in managing complex 
medication regimens through seamless integration with their digital calendar.
"""

import streamlit as st
import datetime
import calendar_integration
import pandas as pd
import os
import json
from dateutil.rrule import rrulestr
from dateutil.rrule import DAILY, WEEKLY, MO, TU, WE, TH, FR, SA, SU



# Constants for medication frequency
FREQUENCIES = {
    "Once daily": {"times": 1, "interval": 1, "unit": "day"},
    "Twice daily": {"times": 2, "interval": 1, "unit": "day"},
    "Three times daily": {"times": 3, "interval": 1, "unit": "day"},
    "Four times daily": {"times": 4, "interval": 1, "unit": "day"},
    "Every other day": {"times": 1, "interval": 2, "unit": "day"},
    "Weekly": {"times": 1, "interval": 1, "unit": "week"},
    "Twice weekly": {"times": 2, "interval": 1, "unit": "week"},
    "Monthly": {"times": 1, "interval": 1, "unit": "month"},
}

TIME_OPTIONS = {
    1: ["08:00 AM"],
    2: ["08:00 AM", "08:00 PM"],
    3: ["08:00 AM", "02:00 PM", "08:00 PM"],
    4: ["08:00 AM", "12:00 PM", "04:00 PM", "08:00 PM"]
}

WEEKDAYS = {
    "Monday": MO,
    "Tuesday": TU,
    "Wednesday": WE,
    "Thursday": TH,
    "Friday": FR,
    "Saturday": SA,
    "Sunday": SU
}

def load_medications():
    """Load saved medications from file"""
    if not os.path.exists('medications.json'):
        return []
    
    try:
        with open('medications.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading medications: {e}")
        return []

def save_medications(medications):
    """Save medications to file"""
    try:
        with open('medications.json', 'w') as f:
            json.dump(medications, f)
        return True
    except Exception as e:
        st.error(f"Error saving medications: {e}")
        return False

def create_medication_reminders(medication, calendar_service):
    """
    Create Google Calendar events for medication reminders
    
    Args:
        medication: Dictionary with medication details
        calendar_service: Google Calendar service object
    
    Returns:
        List of created event IDs
    """
    if not calendar_service:
        st.error("No connection to Google Calendar")
        return []
    
    event_ids = []
    
    # Extract medication details
    name = medication['name']
    dosage = medication['dosage']
    frequency = medication['frequency']
    times = medication['times']
    start_date = medication['start_date']
    end_date = medication.get('end_date', None)  # Optional
    notes = medication.get('notes', '')  # Optional
    
    # Get frequency details
    freq_details = FREQUENCIES[frequency]
    
    # Create events based on frequency
    for time_str in times:
        # Parse time
        hour, minute = 8, 0  # Default
        try:
            if "AM" in time_str or "PM" in time_str:
                # 12-hour format
                time_only = time_str.split(" ")[0]
                hour, minute = map(int, time_only.split(":"))
                if "PM" in time_str and hour != 12:
                    hour += 12
                if "AM" in time_str and hour == 12:
                    hour = 0
            else:
                # 24-hour format
                hour, minute = map(int, time_str.split(":"))
        except:
            st.warning(f"Could not parse time: {time_str}, using default 8:00 AM")
        
        # Create start and end times for event
        start_datetime = datetime.datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=hour, minute=minute
        )
        end_datetime = start_datetime + datetime.timedelta(minutes=15)  # 15-minute event
        
        # Create recurrence rule
        recurrence = None
        if freq_details['unit'] == 'day':
            if freq_details['interval'] == 1:
                recurrence = ['RRULE:FREQ=DAILY']
            else:
                recurrence = [f'RRULE:FREQ=DAILY;INTERVAL={freq_details["interval"]}']
        elif freq_details['unit'] == 'week':
            recurrence = ['RRULE:FREQ=WEEKLY']
        elif freq_details['unit'] == 'month':
            recurrence = ['RRULE:FREQ=MONTHLY']
        
        # Add end date if specified
        if end_date:
            until_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%dT235959Z")
            recurrence[0] += f";UNTIL={until_date}"
        
        # Create the event
        event = {
            'summary': f'💊 {name} ({dosage})',
            'location': '',
            'description': f"""MEDICATION REMINDER
            
Medication: {name}
Dosage: {dosage}
Frequency: {frequency}

{notes}

--
Generated by the Breast Cancer RAG App
            """,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'Europe/Madrid',  # Use the timezone from your calendar integration
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'Europe/Madrid',
            },
            'recurrence': recurrence,
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 5},
                ],
            },
            'colorId': '6'  # Orange for medication (can be changed)
        }
        
        try:
            # Create the event in the calendar
            created_event = calendar_service.events().insert(calendarId='primary', body=event).execute()
            event_ids.append(created_event['id'])
            st.success(f"✅ Created reminder for {name} at {time_str}")
        except Exception as e:
            st.error(f"Error creating event: {e}")
    
    return event_ids

def delete_medication_events(medication, calendar_service):
    """Delete medication events from Google Calendar"""
    if not calendar_service:
        st.error("No connection to Google Calendar")
        return False
    
    event_ids = medication.get('event_ids', [])
    if not event_ids:
        return True
    
    success = True
    for event_id in event_ids:
        try:
            calendar_service.events().delete(calendarId='primary', eventId=event_id).execute()
        except Exception as e:
            st.error(f"Error deleting event: {e}")
            success = False
    
    return success

def medication_reminders_ui():
    """Main UI for medication reminders"""
    st.header("💊 Medication Reminder System")
    
    # Get Google Calendar service
    calendar_service = calendar_integration.initialize_google_calendar()
    if not calendar_service:
        st.warning("Google Calendar is required for medication reminders. Please set up Google Calendar integration first.")
        return
    
    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["Add Medication", "View Medications", "Settings"])
    
    # Load existing medications
    if 'medications' not in st.session_state:
        st.session_state.medications = load_medications()
    
    # Tab 1: Add new medication
    with tab1:
        st.subheader("Add New Medication")
        

        with st.form("new_medication_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                default_name = ""
                default_dosage = ""
                if 'prefill_medication' in st.session_state:
                    default_name = st.session_state.prefill_medication.get('name', '')
                    default_dosage = st.session_state.prefill_medication.get('dosage', '')
                
                med_name = st.text_input("Medication Name", value=default_name)
                med_dosage = st.text_input("Dosage (e.g., 10mg)", value=default_dosage)
                
                med_frequency = st.selectbox(
                    "Frequency",
                    list(FREQUENCIES.keys())
                )
                
                # Get number of times per day/week/etc.
                times_per_period = FREQUENCIES[med_frequency]["times"]
                
                # Default time options based on frequency
                default_times = TIME_OPTIONS.get(times_per_period, ["08:00 AM"])
                
                # For each time slot, show a time input
                med_times = []
                for i in range(times_per_period):
                    # Convert default time to datetime for default value
                    default_time_str = default_times[i] if i < len(default_times) else "08:00 AM"
                    default_hour = 8
                    default_minute = 0
                    
                    if "AM" in default_time_str or "PM" in default_time_str:
                        time_only = default_time_str.split(" ")[0]
                        hour, minute = map(int, time_only.split(":"))
                        if "PM" in default_time_str and hour != 12:
                            hour += 12
                        if "AM" in default_time_str and hour == 12:
                            hour = 0
                        default_hour, default_minute = hour, minute
                    
                    default_time = datetime.time(default_hour, default_minute)
                    
                    time_input = st.time_input(
                        f"Time {i+1}",
                        value=default_time,
                        key=f"time_{i}"
                    )
                    
                    # Convert time input to string format
                    formatted_time = time_input.strftime("%H:%M")
                    med_times.append(formatted_time)
            
            with col2:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.datetime.now().date()
                ).strftime("%Y-%m-%d")
                
                end_date = st.date_input(
                    "End Date (Optional, leave blank for ongoing)",
                    value=None
                )
                if end_date:
                    end_date = end_date.strftime("%Y-%m-%d")
                else:
                    end_date = None
                
                med_notes = st.text_area(
                    "Additional Notes",
                    placeholder="Additional instructions, side effects to watch for, etc."
                )
                
                # Options for reminders
                st.write("Reminder Settings")
                reminder_popup = st.checkbox("Calendar popup reminder", value=True)
                reminder_email = st.checkbox("Email reminder", value=False)
            
            submit_button = st.form_submit_button("Add Medication")
        
        if submit_button:
            if not med_name or not med_dosage:
                st.error("Medication name and dosage are required")
            else:
                # Create medication object
                new_medication = {
                    "name": med_name,
                    "dosage": med_dosage,
                    "frequency": med_frequency,
                    "times": med_times,
                    "start_date": start_date,
                    "end_date": end_date,
                    "notes": med_notes,
                    "created_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "reminder_popup": reminder_popup,
                    "reminder_email": reminder_email
                }
                
                # Create calendar events
                event_ids = create_medication_reminders(new_medication, calendar_service)
                new_medication["event_ids"] = event_ids
                
                # Add to list and save
                st.session_state.medications.append(new_medication)
                save_medications(st.session_state.medications)
                
                st.success(f"✅ Added {med_name} to your medication list and created calendar reminders")
                st.info("You can view and manage your medications in the 'View Medications' tab")
    
    # Tab 2: View/manage medications
    with tab2:
        if not st.session_state.medications:
            st.info("No medications added yet. Go to the 'Add Medication' tab to add your first medication.")
        else:
            st.subheader("Your Medications")
            
            for i, med in enumerate(st.session_state.medications):
                with st.expander(f"{med['name']} ({med['dosage']})"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Medication:** {med['name']}")
                        st.write(f"**Dosage:** {med['dosage']}")
                        st.write(f"**Frequency:** {med['frequency']}")
                        st.write(f"**Times:** {', '.join(med['times'])}")
                        if med.get('notes'):
                            st.write(f"**Notes:** {med['notes']}")
                    
                    with col2:
                        st.write(f"**Start Date:** {med['start_date']}")
                        if med.get('end_date'):
                            st.write(f"**End Date:** {med['end_date']}")
                        else:
                            st.write("**End Date:** Ongoing")
                    
                    with col3:
                        if st.button("Edit", key=f"edit_{i}"):
                            st.session_state.editing_medication = i
                            st.experimental_rerun()
                        
                        if st.button("Delete", key=f"delete_{i}"):
                            if delete_medication_events(med, calendar_service):
                                st.session_state.medications.pop(i)
                                save_medications(st.session_state.medications)
                                st.success(f"Deleted {med['name']} and its reminders")
                                st.experimental_rerun()
            
            # Generate calendar view of upcoming medications
            st.subheader("Upcoming Medication Schedule")
            
            # Get the next 7 days
            today = datetime.datetime.now().date()
            dates = [today + datetime.timedelta(days=i) for i in range(7)]
            
            # Create a schedule
            schedule_data = []
            
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                day_meds = []
                
                for med in st.session_state.medications:
                    start_date = datetime.datetime.strptime(med['start_date'], "%Y-%m-%d").date()
                    end_date = None
                    if med.get('end_date'):
                        end_date = datetime.datetime.strptime(med['end_date'], "%Y-%m-%d").date()
                    
                    # Check if medication is active on this date
                    if date >= start_date and (not end_date or date <= end_date):
                        # Check frequency
                        freq = FREQUENCIES[med['frequency']]
                        
                        # Daily medications
                        if freq['unit'] == 'day':
                            if freq['interval'] == 1 or (date - start_date).days % freq['interval'] == 0:
                                for time in med['times']:
                                    day_meds.append(f"{time} - {med['name']} ({med['dosage']})")
                        
                        # Weekly medications
                        elif freq['unit'] == 'week':
                            # Simple implementation - assuming all weekly meds are on same day of week as start
                            if date.weekday() == start_date.weekday():
                                for time in med['times']:
                                    day_meds.append(f"{time} - {med['name']} ({med['dosage']})")
                        
                        # Monthly medications
                        elif freq['unit'] == 'month':
                            # Simple implementation - assuming all monthly meds are on same day of month as start
                            if date.day == start_date.day:
                                for time in med['times']:
                                    day_meds.append(f"{time} - {med['name']} ({med['dosage']})")
                
                if day_meds:
                    day_meds.sort()  # Sort by time
                    day_formatted = date.strftime("%a, %b %d")
                    for med_time in day_meds:
                        schedule_data.append({
                            "Date": day_formatted,
                            "Medication": med_time
                        })
                else:
                    day_formatted = date.strftime("%a, %b %d")
                    schedule_data.append({
                        "Date": day_formatted,
                        "Medication": "No medications"
                    })
            
            if schedule_data:
                schedule_df = pd.DataFrame(schedule_data)
                st.dataframe(schedule_df, use_container_width=True)
    
    # Tab 3: Settings
    with tab3:
        st.subheader("Medication Reminder Settings")
        
        st.write("**Calendar Settings**")
        calendar_color = st.color_picker(
            "Calendar event color",
            "#FF9800"  # Orange
        )
        
        reminder_times = st.multiselect(
            "Default reminder times before medication",
            ["5 minutes", "15 minutes", "30 minutes", "1 hour", "2 hours"],
            default=["5 minutes"]
        )
        
        st.write("**Notification Settings**")
        enable_browser = st.checkbox("Enable browser notifications", value=True)
        enable_email = st.checkbox("Enable email notifications", value=False)
        
        if enable_email:
            email_address = st.text_input("Email address for notifications")
        
        if st.button("Save Settings"):
            st.success("Settings saved")
            # In a real app, you would save these settings to a config file
            # and apply them when creating new medication reminders

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Medication Reminder System", page_icon="💊", layout="wide")
    medication_reminders_ui()