import cv2
import face_recognition
import pickle
import numpy as np
from attendance_system import mark_attendance
import os

def recognize_faces():
    encoding_file = os.path.join("models", "encodings.pickle")
    if not os.path.exists(encoding_file):
        print("[ERROR] No trained model found! Please run train_model.py first.")
        return

    print("[INFO] Loading encodings...")
    with open(encoding_file, "rb") as f:
        data = pickle.loads(f.read())

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
            name = "Unknown"
            confidence = "0.00%"

            # Calculate face distances to find the best match and confidence score
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = data["names"][best_match_index]
                    dist = face_distances[best_match_index]
                    # Map distance 0.0-0.6 to 100%-0%
                    conf_value = max(0, min(100, (1.0 - (dist / 0.6)) * 100))
                    confidence = f"{conf_value:.2f}%"
                    
                    # Mark attendance
                    mark_attendance(name)

            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            label = f"{name} ({confidence})"
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Recognition Attendance System', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
