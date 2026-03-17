import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import subprocess
import os
from attendance_stats import get_attendance_stats

class AdminPortal(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Admin Portal - Analytics")
        self.geometry("750x550")
        self.after(10, self.lift) # Bring to front

        # Header
        ctk.CTkLabel(self, text="Admin Dashboard", font=ctk.CTkFont(size=26, weight="bold")).pack(pady=20)
        
        # Summary
        stats = get_attendance_stats()
        total_students = len(stats)
        ctk.CTkLabel(self, text=f"Total Registered Students: {total_students}", font=ctk.CTkFont(size=14)).pack()

        # Table Header
        h_frame = ctk.CTkFrame(self, fg_color="gray20", height=40)
        h_frame.pack(fill="x", padx=30, pady=(20, 0))
        
        ctk.CTkLabel(h_frame, text="Student Name", width=220, anchor="w").pack(side="left", padx=15)
        ctk.CTkLabel(h_frame, text="Attendance Record", width=150).pack(side="left", padx=15)
        ctk.CTkLabel(h_frame, text="Percentage", width=150).pack(side="left", padx=15)

        # List
        self.scroll = ctk.CTkScrollableFrame(self, width=680, height=350)
        self.scroll.pack(padx=20, pady=10, fill="both", expand=True)

        if not stats:
            ctk.CTkLabel(self.scroll, text="No attendance logs found yet.").pack(pady=30)
        else:
            for item in stats:
                row = ctk.CTkFrame(self.scroll, fg_color="transparent")
                row.pack(fill="x", pady=2)
                
                ctk.CTkLabel(row, text=item['name'], width=220, anchor="w").pack(side="left", padx=15)
                ctk.CTkLabel(row, text=f"{item['present']} / {item['total']} Days", width=150).pack(side="left", padx=15)
                
                pct = float(item['percentage'].replace('%',''))
                color = "#2ECC71" if pct >= 75 else "#E67E22" if pct >= 50 else "#E74C3C"
                ctk.CTkLabel(row, text=item['percentage'], width=150, text_color=color, font=ctk.CTkFont(weight="bold")).pack(side="left", padx=15)

class MainLauncher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Attendance System - Dual Portal")
        self.geometry("500x450")
        ctk.set_appearance_mode("Dark")
        
        # UI
        frame = ctk.CTkFrame(self, corner_radius=20)
        frame.pack(pady=30, padx=30, fill="both", expand=True)

        ctk.CTkLabel(frame, text="Select Portal", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(40, 40))

        # Student Button
        self.btn_student = ctk.CTkButton(frame, text="Student Portal", height=60, width=300, 
                                        fg_color="#3498DB", hover_color="#2980B9",
                                        font=ctk.CTkFont(size=18, weight="bold"),
                                        command=self.open_student_portal)
        self.btn_student.pack(pady=15)

        # Admin Button
        self.btn_admin = ctk.CTkButton(frame, text="Admin Portal", height=60, width=300,
                                      fg_color="#8E44AD", hover_color="#7D3C98",
                                      font=ctk.CTkFont(size=18, weight="bold"),
                                      command=self.open_admin_portal)
        self.btn_admin.pack(pady=15)

    def open_admin_portal(self):
        # Admin can be password protected optionally
        AdminPortal(self)

    def open_student_portal(self):
        import sys
        os.system(f'"{sys.executable}" student_portal.py &') 

if __name__ == "__main__":
    app = MainLauncher()
    app.mainloop()
