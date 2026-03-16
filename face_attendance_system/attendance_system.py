import os
import pandas as pd
from datetime import datetime

def mark_attendance(name):
    """
    Records the attendance of a recognized person.
    Creates a new CSV file for each specific date.
    Avoids duplicate entries for the same person on the same day.
    """
    attendance_dir = 'attendance'
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    filename = os.path.join(attendance_dir, f"Attendance_{date_str}.csv")
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    if file_exists:
        # Read the existing attendance
        df = pd.read_csv(filename)
        # Check if name is already present
        if name in df['Name'].values:
            # Person already marked for today
            return False
    else:
        # Create a new DataFrame if file doesn't exist
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        
    # Append the new entry
    new_entry = pd.DataFrame([{'Name': name, 'Date': date_str, 'Time': time_str}])
    df = pd.concat([df, new_entry], ignore_index=True)
    
    # Save back to CSV
    df.to_csv(filename, index=False)
    print(f"✅ Attendance marked for {name} at {time_str}")
    return True
