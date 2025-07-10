# recognize_faces_with_registration_countdown.py

import face_recognition
import cv2
import numpy as np
import pickle
import os
import subprocess
import uuid
import sys
import time

import tkinter as tk
from tkinter import simpledialog

def get_name_gui():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Face Registration", "Enter name for the new face:")
    root.destroy()
    return name.strip() if name else ""

# Load cached face encodings
def load_encodings():
    with open("encodings.pkl", "rb") as f:
        return pickle.load(f)

known_face_encodings, known_face_names = load_encodings()

# Setup known faces directory
known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

video_capture = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

TOLERANCE = 0.5  # Adjust if needed (lower is stricter)
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < TOLERANCE:
                    name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)

        if name == "Unknown":
            cv2.putText(frame, "Press 'r' to register", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):
        # Register unknown face
        user_name = get_name_gui()


        if user_name:
            save_path = os.path.join(known_faces_dir, user_name)
            os.makedirs(save_path, exist_ok=True)

            print("[INFO] Get ready! Countdown starting...")

            # Countdown 3-2-1 on screen
            for i in range(3, 0, -1):
                ret, frame = video_capture.read()
                if not ret:
                    continue
                countdown_frame = frame.copy()
                cv2.putText(countdown_frame, str(i), (frame.shape[1]//2 - 30, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
                cv2.imshow("Face Recognition", countdown_frame)
                cv2.waitKey(1000)  # Wait 1 second

            # "Smile" message
            ret, frame = video_capture.read()
            if ret:
                smile_frame = frame.copy()
                cv2.putText(smile_frame, "Smile!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                cv2.imshow("Face Recognition", smile_frame)
                cv2.waitKey(1000)

            print("[INFO] Capturing 5 photos now...")

            capture_count = 0
            while capture_count < 5:
                ret, capture_frame = video_capture.read()
                if not ret:
                    continue

                display_frame = capture_frame.copy()
                cv2.putText(display_frame, f"Capturing image {capture_count+1}/5", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Face Recognition", display_frame)

                small_capture = cv2.resize(capture_frame, (0, 0), fx=0.25, fy=0.25)
                rgb_capture = cv2.cvtColor(small_capture, cv2.COLOR_BGR2RGB)
                capture_face_locations = face_recognition.face_locations(rgb_capture)

                for (top, right, bottom, left) in capture_face_locations:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    face_img = capture_frame[top:bottom, left:right]
                    filename = f"{uuid.uuid4().hex}.jpg"
                    file_full_path = os.path.join(save_path, filename)
                    cv2.imwrite(file_full_path, face_img)
                    print(f"[INFO] Saved image {filename}")
                    capture_count += 1

                cv2.waitKey(500)  # short delay between captures

            print("[INFO] Finished capturing images.")
            cv2.destroyWindow("Face Recognition")

            # Re-run encoding generation
            print("[INFO] Updating encodings...")
            subprocess.run(["python", "generate_encodings.py"])
            print("[INFO] Encodings updated.")

            print("[INFO] Restarting program to load new faces...")
            video_capture.release()
            cv2.destroyAllWindows()
            time.sleep(1)

            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit()

        else:
            print("[WARNING] No name entered, skipping registration.")

video_capture.release()
cv2.destroyAllWindows()



