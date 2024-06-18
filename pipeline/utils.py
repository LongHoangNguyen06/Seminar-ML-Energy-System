from datetime import datetime


def current_time_str():
    # Get the current time
    now = datetime.now()
    # Format the time and date as per the specified format
    return now.strftime("%H:%M %d-%m-%Y")
