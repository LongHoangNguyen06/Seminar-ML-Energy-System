import socket
from datetime import datetime


def current_time_str():
    # Get the current time
    now = datetime.now()
    # Get the hostname
    hostname = socket.gethostname()
    # Format the time and date to include seconds in the specified format
    current_time = now.strftime("%H:%M:%S %d-%m-%Y")
    # Combine the hostname with the current time string
    return f"{hostname}_{current_time}"
