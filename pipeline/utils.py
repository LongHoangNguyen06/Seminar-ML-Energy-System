from datetime import datetime


def current_time_str():
    # Get the current time
    now = datetime.now()
    # Format the time and date to include seconds in the specified format
    return now.strftime("%H:%M:%S %d-%m-%Y")
