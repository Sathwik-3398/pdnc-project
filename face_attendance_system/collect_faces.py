import cv2
import os

def collect_data(name, num_images=30):
    dataset_dir = 'dataset'
    person_dir = os.path.join(dataset_dir, name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    # Using Haar cascade to detect faces for cropping
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    print(f"[INFO] Collecting {num_images} faces for {name}. Please look at the camera.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        orig_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save the clean frame without bounding box markings
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, orig_frame)
            count += 1
            print(f"[INFO] Captured image {count}/{num_images}")
            
            cv2.waitKey(200) # Ensure diverse capture angles
            
            if count >= num_images:
                break

        cv2.imshow('Face Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Face collection complete for {name}.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect face dataset using webcam.")
    parser.add_argument("--name", type=str, required=True, help="Name of the person")
    parser.add_argument("--count", type=int, default=30, help="Number of images to capture")
    args = parser.parse_args()
    collect_data(args.name, args.count)
