import cv2
import face_recognition
import pickle
import numpy as np
from attendance_system import mark_attendance
from liveness_detection import LivenessDetector   # ← NEW: anti-spoofing module
import os
import time
from collections import defaultdict

def draw_hud(frame, status_text, color):
    """Draws a beautiful, elegant overlay HUD on the camera frame."""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1) # Top bar
    cv2.rectangle(overlay, (0, h-50), (w, h), (0, 0, 0), -1) # Bottom bar
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    
    # Title Text (shadowed)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, status_text, (32, 52), font, 1.0, (0, 0, 0), 3) # Shadow
    cv2.putText(frame, status_text, (30, 50), font, 1.0, color, 2)     # Main text
    
    # Bottom Footer
    footer = "AI Biometric Attendance System - Active"
    cv2.putText(frame, footer, (30, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def draw_targeting_box(frame, top, right, bottom, left, color=(0, 255, 255), thickness=2, corner_length=25, draw_solid=False):
    """Draws a futuristic targeting square around the face."""
    if draw_solid:
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
    else:
        # Just draw a faint white border
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
        
    # Draw thicker corners
    # Top-Left
    cv2.line(frame, (left, top), (left + corner_length, top), color, thickness*2)
    cv2.line(frame, (left, top), (left, top + corner_length), color, thickness*2)
    # Top-Right
    cv2.line(frame, (right, top), (right - corner_length, top), color, thickness*2)
    cv2.line(frame, (right, top), (right, top + corner_length), color, thickness*2)
    # Bottom-Left
    cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, thickness*2)
    cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, thickness*2)
    # Bottom-Right
    cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, thickness*2)
    cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, thickness*2)

def take_attendance():
    # ── Liveness detector (created once, reused per session) ──────────────
    liveness = LivenessDetector()

    encoding_file = os.path.join("models", "encodings.pickle")
    if not os.path.exists(encoding_file):
        print("ERROR:Model Not Found")
        return

    try:
        with open(encoding_file, "rb") as f:
            data = pickle.loads(f.read())
    except Exception as e:
        print("ERROR:Model Not Found")
        return

    if not data.get("encodings"):
        print("ERROR:No faces registered")
        return

    cap = cv2.VideoCapture(0)
    
    # --- 1. Warm-up / Readiness Timer (3 Seconds) ---
    start_time = time.time()
    countdown = 3
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        elapsed = time.time() - start_time
        remaining = int(countdown - elapsed)
        if remaining <= 0: break
            
        draw_hud(frame, f"System Initializing... {remaining}s", (0, 165, 255)) # Orange
        cv2.imshow("Face Attendance Scanner", frame)
        cv2.waitKey(1)
        
    # --- 2. Scanning Phase ---
    found_unknown = False
    consecutive_matches = defaultdict(int)
    REQUIRED_MATCHES = 3 
    
    for _ in range(120): # Bumped frames slightly to accommodate slower rendering
        ret, frame = cap.read()
        if not ret: break
            
        # Default HUD Text
        status_text = "Scanning for Authorized Personnel..."
        hud_color = (0, 255, 255) # Yellow
        
        if found_unknown:
            status_text = "WARNING: Unrecognized Face Detected"
            hud_color = (0, 0, 255) # Red
            
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if not face_locations:
            draw_hud(frame, status_text, hud_color)
            cv2.imshow("Face Attendance Scanner", frame)
            cv2.waitKey(1)
            continue
            
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        found_unknown_this_frame = True 
        
        for i, face_encoding in enumerate(face_encodings):
            # Scale face locations back up
            top, right, bottom, left = face_locations[i]
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            
            box_color = (0, 255, 255) if not found_unknown else (0, 0, 255)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                    found_unknown_this_frame = False
                    name = data["names"][best_match_index]
                    
                    consecutive_matches[name] += 1
                    
                    if consecutive_matches[name] >= REQUIRED_MATCHES:
                        # ── FACE MATCHED → Run liveness / blink check first ──────────
                        try:
                            liveness_status = liveness.run_blink_check(
                                cap,
                                draw_hud_fn   = draw_hud,
                                draw_box_fn   = draw_targeting_box,
                                face_top      = top,
                                face_right    = right,
                                face_bottom   = bottom,
                                face_left     = left,
                                recognized_name = name,
                            )
                        except Exception as e:
                            import sys
                            print(f"[LIVENESS CRITICAL] {e}", file=sys.stderr)
                            liveness_status = "ERROR"

                        if liveness_status == "LIVE":
                            # ── Both eye + voice verified → mark attendance ─────────────
                            mark_attendance(name)
                            print(f"SUCCESS:{name}")

                        elif liveness_status == "VOICE_FAIL":
                            # ── Blink ok but voice not heard ──────────────────────
                            print("VOICE_FAIL:Did not say Present Sir")

                        else:
                            # ── SPOOF or ERROR → no attendance ───────────────────
                            print("SPOOF:Blink not detected")

                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    else:
                        # Seen them, but need more frames to verify
                        status_text = f"Verifying Identity: {name.upper()}..."
                        hud_color = (0, 255, 255)
                        box_color = (0, 255, 255)
            
            # Draw the box around the face
            draw_targeting_box(frame, top, right, bottom, left, color=box_color)
                        
        if found_unknown_this_frame:
            found_unknown = True
            
        draw_hud(frame, status_text, hud_color)
        cv2.imshow("Face Attendance Scanner", frame)
        cv2.waitKey(1)
            
    # Timeout / Failure
    if found_unknown:
        print("UNKNOWN:Face not present")
    else:
        print("TIMEOUT:No face detected")
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    take_attendance()
