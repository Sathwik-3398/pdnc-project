import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import subprocess
import os

# Set the path to the correct Python virtual environment
python_executable = "/Users/apple/pdnc project/.venv/bin/python"
if not os.path.exists(python_executable):
    python_executable = "python" # fallback

# Configure Appearance
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def register_face():
    name = simpledialog.askstring("Register Face", "Enter the Name of the Person:", parent=app)
    if name:
        messagebox.showinfo("Instructions", "The webcam will now open to scan your face.\n\nPlease look directly at the camera. It will automatically capture 30 images and close.")
        # Step 1: Collect Data
        subprocess.run([python_executable, "collect_faces.py", "--name", name, "--count", "30"])
        
        # Step 2: Automatically Train the model so it learns the new face immediately
        messagebox.showinfo("Wait", "Updating the Face Database...\n\nPlease wait a few seconds.")
        result = subprocess.run([python_executable, "train_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            messagebox.showinfo("Success", f"{name} has been successfully registered!")
        else:
            messagebox.showerror("Error", f"Failed to register face.\n{result.stderr}")

def take_attendance():
    messagebox.showinfo("Scanning", "Click OK to scan your face. \n\nPlease look directly at the camera for a few seconds.")
    # Run the single shot scanner
    result = subprocess.run([python_executable, "take_attendance.py"], capture_output=True, text=True)
    
    output_lines = [line for line in result.stdout.strip().split("\n") if line.strip() != ""]
    if not output_lines:
        messagebox.showerror("Error", "Camera failed or process was interrupted.")
        return
        
    # Process output matching the script's returns by searching all lines
    success_name = None
    status = "UNKNOWN_ERROR"
    
    for line in output_lines:
        if line.startswith("SUCCESS:"):
            status = "SUCCESS"
            success_name = line.split(":")[1]
            break
        elif line.startswith("SPOOF:"):
            status = "SPOOF"
            break
        elif line.startswith("VOICE_FAIL:"):
            status = "VOICE_FAIL"
            break
        elif line.startswith("UNKNOWN:"):
            status = "UNKNOWN"
        elif line.startswith("ERROR:"):
            status = "ERROR"
        elif line.startswith("TIMEOUT:"):
            status = "TIMEOUT"

    if status == "SUCCESS":
        messagebox.showinfo("Attendance Marked", f"Attendance Marked successfully for {success_name}!")
    elif status == "SPOOF":
        messagebox.showerror("⚠ Spoof Detected", "Liveness check FAILED.\n\nNo blink was detected within the time limit.\n\nAttendance was NOT marked.")
    elif status == "VOICE_FAIL":
        messagebox.showwarning("Voice Not Verified",
            "Eye blink verified, but 'Present Sir' was not heard.\n\nPlease speak clearly and try again.\n\nAttendance was NOT marked.")
    elif status == "UNKNOWN":
        messagebox.showerror("Not Recognized", "Face not present in the registered database.")
    elif status == "ERROR":
        messagebox.showerror("System Error", "No registered faces or database model found. Please register a face first.")
    elif status == "TIMEOUT":
        messagebox.showwarning("No Face", "No face was detected. Please try again and look directly at the camera.")
    else:
        # Fallback for unexpected output
        messagebox.showerror("Error", f"Process exited with unexpected response:\n{result.stdout.strip()}")

# Build User Interface
app = ctk.CTk()
app.title("Face Attendance System")
app.geometry("450x380")
app.resizable(False, False)

# Main Frame
frame = ctk.CTkFrame(master=app, corner_radius=15)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Title Label
title_label = ctk.CTkLabel(master=frame, 
                           text="AI Face Attendance", 
                           font=ctk.CTkFont(size=28, weight="bold"))
title_label.pack(pady=(30, 10))

subtitle_label = ctk.CTkLabel(master=frame, 
                              text="Powered by Deep Neural Networks", 
                              font=ctk.CTkFont(size=14, slant="italic"),
                              text_color="gray")
subtitle_label.pack(pady=(0, 30))

# Option 1: Take Attendance
btn_attendance = ctk.CTkButton(master=frame, 
                               text="Take Attendance", 
                               font=ctk.CTkFont(size=18, weight="bold"),
                               command=take_attendance, 
                               height=50, 
                               fg_color="#2ECC71",  # Emerald Green
                               hover_color="#27AE60",
                               corner_radius=8)
btn_attendance.pack(pady=15, padx=40, fill='x')

# Option 2: Register Face
btn_register = ctk.CTkButton(master=frame, 
                             text="Register Face ID", 
                             font=ctk.CTkFont(size=18, weight="bold"),
                             command=register_face, 
                             height=50, 
                             fg_color="#3498DB",  # Bright Blue
                             hover_color="#2980B9",
                             corner_radius=8)
btn_register.pack(pady=15, padx=40, fill='x')

# Run the app
if __name__ == "__main__":
    app.mainloop()
