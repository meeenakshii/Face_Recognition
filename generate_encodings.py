import os
import face_recognition
import pickle
import numpy as np

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

known_face_encodings = []
known_face_names = []

print("[INFO] Generating face encodings...")

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[+] Encoding for {name}/{filename} added.")
            else:
                print(f"[WARNING] No face found in {name}/{filename}.")
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath}: {e}")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"[INFO] {len(known_face_encodings)} face encodings saved to {ENCODINGS_FILE}.")
