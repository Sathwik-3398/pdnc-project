import cv2
import face_recognition
import os
import pickle

def train_model():
    dataset_dir = 'dataset'
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    known_encodings = []
    known_names = []

    print("[INFO] Quantifying faces...")

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, file)
                name = os.path.basename(root)

                image = cv2.imread(path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect x,y coordinates of the bounding boxes in the image
                boxes = face_recognition.face_locations(rgb, model='hog')

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)

                # loop over the encodings (usually 1 face per collected image)
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)

    print(f"[INFO] Serializing {len(known_encodings)} encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(os.path.join(models_dir, "encodings.pickle"), "wb") as f:
        f.write(pickle.dumps(data))
        
    print("[INFO] Training complete! Model saved to models/encodings.pickle")

if __name__ == "__main__":
    train_model()
