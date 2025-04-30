# Medication Reminder System for Healthcare Applications

## Introduction

The Medication Reminder System is a comprehensive solution for managing medication schedules and generating automated reminders for healthcare applications. Designed specifically for breast cancer patients and survivors, this module helps users track medications, set up recurring reminders, and visualize their medication schedule within a streamlined Streamlit interface.

By integrating with Google Calendar, the system creates time-based reminders with customizable notifications, ensuring patients maintain adherence to complex medication regimens. The module emphasizes usability, flexibility, and integration capabilities to support overall treatment management.

## Table of Contents

1. [Core Features](#core-features)
2. [Technical Architecture](#technical-architecture)
3. [Medication Data Model](#medication-data-model)
4. [Calendar Integration](#calendar-integration)
5. [Recurrence Rules](#recurrence-rules)
6. [Visualization Features](#visualization-features)
7. [User Interface Components](#user-interface-components)
8. [Data Persistence](#data-persistence)
9. [Example Usage](#example-usage)
10. [Technical Reference](#technical-reference)

## Core Features

The Medication Reminder System provides several key capabilities:

1. **Medication Tracking**: Record medication details including name, dosage, and notes
2. **Flexible Scheduling**: Support for various dosing frequencies (daily, weekly, monthly)
3. **Multiple Daily Doses**: Configure multiple time slots for medications taken multiple times per day
4. **Google Calendar Integration**: Automatic creation of recurring events with reminders
5. **Visual Schedule**: Weekly calendar view showing upcoming medication doses
6. **Persistent Storage**: Local JSON-based storage for medication data
7. **Customizable Reminders**: Configure popup and email reminders with adjustable timing
8. **Complete Management Interface**: Add, edit, view, and delete medications through a streamlined UI

## Technical Architecture

The module consists of several complementary components:

1. **Data Management**: Functions for loading, saving, and manipulating medication data
2. **Calendar Integration**: Components for creating and managing Google Calendar events
3. **UI Components**: Streamlit-based interface elements for medication management
4. **Scheduling Logic**: Algorithm for determining medication timing based on frequency

The architecture emphasizes modularity and flexibility, allowing the system to be easily extended or integrated with other healthcare management tools.

## Medication Data Model

The system uses a structured data model to represent medications and their scheduling parameters:

```python
{
    "name": "Tamoxifen",
    "dosage": "20mg",
    "frequency": "Once daily",
    "times": ["08:00"],
    "start_date": "2025-04-21",
    "end_date": "2025-10-21",  # Optional
    "notes": "Take with food to reduce nausea",
    "created_date": "2025-04-20",
    "reminder_popup": True,
    "reminder_email": False,
    "event_ids": ["google_calendar_event_id_1"]  # References to Google Calendar events
}
```

This comprehensive data structure captures all necessary aspects of medication management:

- Basic medication information (name, dosage)
- Scheduling parameters (frequency, times, start/end dates)
- Additional context (notes, creation date)
- Reminder preferences (popup, email)
- Calendar integration metadata (event IDs)

## Calendar Integration

The module seamlessly integrates with Google Calendar to create recurring reminder events:

```python
def create_medication_reminders(medication, calendar_service):
    """
    Create Google Calendar events for medication reminders
    
    Args:
        medication: Dictionary with medication details
        calendar_service: Google Calendar service object
        
    Returns:
        List of created event IDs
    """
```

This function handles:

1. Converting medication frequency into proper recurrence rules
2. Creating separate events for each time of day the medication should be taken
3. Setting up appropriate reminders (popup, email)
4. Formatting event titles and descriptions for clear identification
5. Tracking created events for future reference and management

Each medication is represented as a recurring event with:
- A clear title including medication name and dosage
- A detailed description with all medication details
- Color-coding for easy visual identification (orange for medications)
- Appropriately configured recurrence rules based on frequency
- Custom reminders (5 minutes before by default)

## Recurrence Rules

The system supports sophisticated recurrence patterns using the RFC 5545 standard for iCalendar:

```python
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
```

For each frequency, the system generates appropriate RRULE strings:
- Daily medications: `RRULE:FREQ=DAILY`
- Every other day: `RRULE:FREQ=DAILY;INTERVAL=2`
- Weekly medications: `RRULE:FREQ=WEEKLY`
- Monthly medications: `RRULE:FREQ=MONTHLY`

End dates are incorporated as UNTIL parameters when specified:
```python
if end_date:
    until_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%dT235959Z")
    recurrence[0] += f";UNTIL={until_date}"
```

## Visualization Features

The system includes a 7-day visual schedule that shows upcoming medications:

```python
# Generate calendar view of upcoming medications
st.subheader("Upcoming Medication Schedule")
# Get the next 7 days
today = datetime.datetime.now().date()
dates = [today + datetime.timedelta(days=i) for i in range(7)]
```

This schedule:
1. Shows medications for the next 7 days
2. Sorts medications by time within each day
3. Presents a clear, formatted display of medication times and details
4. Handles all frequency patterns appropriately 
5. Uses pandas DataFrames for structured display

Example output:
```
| Date         | Medication                      |
|--------------|--------------------------------|
| Wed, Apr 30  | 08:00 - Tamoxifen (20mg)       |
| Wed, Apr 30  | 20:00 - Anastrozole (1mg)      |
| Thu, May 1   | 08:00 - Tamoxifen (20mg)       |
| Thu, May 1   | 14:00 - Vitamin D (1000 IU)    |
```

## User Interface Components

The module provides a complete Streamlit-based user interface through the `medication_reminders_ui()` function:

```python
def medication_reminders_ui():
    """Main UI for medication reminders"""
```

The interface is organized into three tabs:

### 1. Add Medication Tab

This tab provides a form for adding new medications with fields for:
- Medication name and dosage
- Frequency selection
- Time selection (with appropriate number of time slots based on frequency)
- Start and end dates
- Additional notes
- Reminder preferences

### 2. View Medications Tab

This tab displays existing medications with features for:
- Expandable sections showing full medication details
- Edit and delete options for each medication
- A 7-day schedule view of upcoming doses

### 3. Settings Tab

This tab provides configuration options for:
- Calendar event color
- Default reminder times
- Notification preferences (browser, email)

## Data Persistence

The system implements simple but effective data persistence using JSON files:

```python
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
```

This approach:
1. Stores all medication data in a local JSON file
2. Handles loading medications at application startup
3. Saves changes whenever medications are added or deleted
4. Includes comprehensive error handling for file operations

## Example Usage

### Basic Medication Reminder Setup

```python
import streamlit as st
from medication_reminders import medication_reminders_ui

# Initialize the app
st.set_page_config(page_title="Medication Reminders", page_icon="💊")
st.title("My Medication Reminders")

# Run the medication reminders UI
medication_reminders_ui()
```

### Integration with Calendar

```python
import streamlit as st
import calendar_integration
from medication_reminders import create_medication_reminders

# Initialize calendar service
calendar_service = calendar_integration.initialize_google_calendar()

if calendar_service:
    # Create a simple medication reminder
    medication = {
        "name": "Tamoxifen",
        "dosage": "20mg",
        "frequency": "Once daily",
        "times": ["08:00"],
        "start_date": "2025-05-01",
        "notes": "Take with food"
    }
    
    # Create calendar reminders
    event_ids = create_medication_reminders(medication, calendar_service)
    
    if event_ids:
        st.success(f"Created {len(event_ids)} calendar reminder(s)")
```

### Custom Schedule View

```python
import streamlit as st
import pandas as pd
import datetime
from medication_reminders import load_medications, FREQUENCIES

# Load existing medications
medications = load_medications()

# Create schedule for next 30 days
today = datetime.datetime.now().date()
schedule = []

for i in range(30):
    date = today + datetime.timedelta(days=i)
    
    # Check each medication
    for med in medications:
        # Logic to determine if medication should be taken on this date
        # (similar to the weekly schedule logic in the module)
        # ...
        
        # Add to schedule if needed
        # schedule.append(...)
        
# Create DataFrame and display
schedule_df = pd.DataFrame(schedule)
st.dataframe(schedule_df)
```

## Technical Reference

### Core Functions

```python
def load_medications():
    """
    Load saved medications from file
    
    Returns:
        List of medication dictionaries
    """

def save_medications(medications):
    """
    Save medications to file
    
    Args:
        medications: List of medication dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """

def create_medication_reminders(medication, calendar_service):
    """
    Create Google Calendar events for medication reminders
    
    Args:
        medication: Dictionary with medication details
        calendar_service: Google Calendar service object
        
    Returns:
        List of created event IDs
    """

def delete_medication_events(medication, calendar_service):
    """
    Delete medication events from Google Calendar
    
    Args:
        medication: Dictionary with medication details and event_ids
        calendar_service: Google Calendar service object
        
    Returns:
        bool: True if successful, False otherwise
    """

def medication_reminders_ui():
    """
    Main UI for medication reminders
    """
```

### Constants

```python
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
```

### Dependencies

- `streamlit`: For UI components
- `calendar_integration`: For Google Calendar integration
- `pandas`: For data management and display
- `datetime`: For date and time handling
- `json`: For data persistence

---

This Medication Reminder System provides a powerful yet user-friendly way for breast cancer patients to manage their medication regimens. By combining calendar integration, flexible scheduling, and an intuitive interface, it helps promote medication adherence and reduces the cognitive burden of complex treatment schedules. The system can be easily integrated into broader healthcare applications or used as a standalone tool for medication management.
