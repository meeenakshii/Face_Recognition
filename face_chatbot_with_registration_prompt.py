
import face_recognition
import cv2
import numpy as np
import pickle
import speech_recognition as sr
import pyttsx3
import time
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import threading
import os
import requests
from queue import Queue
import queue
import subprocess
import sys
import uuid

# === Groq API Configuration ===
GROQ_API_KEY = "xxxxxxxxxxxx"
GROQ_MODEL = "llama3-70b-8192"

# === Custom System Prompt for RecipBot ===
system_prompt = (
    "As an Adroitent assistant, provide concise, accurate, one-line responses about Adroitent, Inc., a trusted digital transformation partner with 19 years of expertise and over 500 skilled associates worldwide, offering Software Engineering, AI, SaaS, ERP, Cloud, and Business Intelligence Solutions. Emphasize our SEI CMMI Level 3 accreditation, ISO 9001 and 27001 certifications, and leadership: Partha Bommireddy, co-founder and President, drives growth with strategic AI expertise; Srinath, co-founder and IT advisor to Andhra Pradesh, leverages global experience; and Sriram, VP of Delivery, leads enterprise-scale programs. Answer questions solely based on this information, reflecting our commitment to agility, innovation, and quality."
)

# === TTS Engine ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Usually female on Windows

# === Voice Recognizer ===
recognizer = sr.Recognizer()

# === Load Face Encodings ===
ENCODINGS_FILE = "encodings.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

TOLERANCE = 0.4
DETECTION_TIMEOUT = 10
RESPONSE_TIMEOUT = 10
cooldown_until = 0
FRAME_PROCESS_INTERVAL = 5

# Setup known faces directory
known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def speech_to_text(app):
    if not app.processing:
        return None
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
            return recognizer.recognize_google(audio)
        except:
            return None

def get_name_gui():
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Face Registration", "Enter name for the new face:")
    root.destroy()
    return name.strip() if name else ""

def chatbot_response(text, name="there"):
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Groq API error:", e)
        return f"Sorry {name}, I couldn't think of a response."

class FaceChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1200x600")
        self.running = True
        self.chatting = False
        self.greeted = False
        self.processing = True
        self.recognizing = True
        self.current_person = None
        self.last_detected_name = None
        self.last_detection_time = time.time()
        self.frame_queue = Queue(maxsize=20)
        self.new_person_detected = False
        self.new_person_name = None
        self.frame_count = 0
        self.chat_visible = False
        self.registering = False  # Flag to control registration phase

        # Title Label
        self.title_label = tk.Label(root, text="Welcome to Adroitent", font=("Arial", 24, "bold"))
        self.title_label.pack(side=tk.TOP, pady=10)
        print("Title label created and packed")

        # Main frame to hold video and chat frames
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Video Frame
        self.video_frame = tk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Chat Frame
        self.chat_frame = tk.Frame(self.main_frame)
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        self.chat_frame.config(width=300)

        self.chat_text = tk.Text(self.chat_frame, height=15, width=30, wrap=tk.WORD)
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.config(state=tk.DISABLED)

        # Text input area
        self.input_frame = tk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, pady=5)

        self.input_entry = tk.Entry(self.input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_text_input)
        self.send_button.pack(side=tk.RIGHT)

        # Hide chat components initially
        self.chat_text.pack_forget()
        self.input_frame.pack_forget()
        self.chat_frame.pack_forget()

        # Show Chat Button
        self.toggle_button = tk.Button(root, text="Show Chat", command=self.toggle_chat)
        self.toggle_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)  # Bottom right

        # Bind 'r' key to trigger registration
        self.root.bind('r', self.handle_register_key)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        threading.Thread(target=self.capture_video, daemon=True).start()
        threading.Thread(target=self.process_faces, daemon=True).start()
        self.update_video()

    def handle_register_key(self, event):
        if self.last_detected_name == "Unknown" and self.recognizing and not self.registering:
            self.registering = True
            self.recognizing = False  # Pause recognition during registration
            threading.Thread(target=self.register_unknown_face, daemon=True).start()

    def register_unknown_face(self):
        # Prompt for name using Tkinter dialog
        user_name = get_name_gui()
        if not user_name:
            print("[WARNING] No name entered, skipping registration.")
            self.registering = False
            self.recognizing = True
            return

        save_path = os.path.join(known_faces_dir, user_name)
        os.makedirs(save_path, exist_ok=True)

        print("[INFO] Get ready! Countdown starting...")
        speak("Get ready! Countdown starting...")

        # Countdown 3-2-1 on screen
        for i in range(3, 0, -1):
            if not self.running:
                self.registering = False
                self.recognizing = True
                return
            ret, frame = video_capture.read()
            if not ret:
                continue
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, str(i), (countdown_frame.shape[1]//2 - 30, countdown_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
            try:
                self.frame_queue.put_nowait(countdown_frame)
            except queue.Full:
                pass
            time.sleep(1)  # Wait 1 second

        # "Smile" message
        ret, frame = video_capture.read()
        if ret:
            smile_frame = frame.copy()
            cv2.putText(smile_frame, "Smile!", (smile_frame.shape[1]//2 - 100, smile_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            try:
                self.frame_queue.put_nowait(smile_frame)
            except queue.Full:
                pass
            time.sleep(1)

        print("[INFO] Capturing 5 photos now...")
        speak("Capturing 5 photos now...")

        capture_count = 0
        while capture_count < 5:
            if not self.running:
                self.registering = False
                self.recognizing = True
                return
            ret, capture_frame = video_capture.read()
            if not ret:
                continue

            display_frame = capture_frame.copy()
            cv2.putText(display_frame, f"Capturing image {capture_count+1}/5", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            try:
                self.frame_queue.put_nowait(display_frame)
            except queue.Full:
                pass

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

            time.sleep(0.5)  # Short delay between captures

        print("[INFO] Finished capturing images.")
        speak("Finished capturing images.")

        # Re-run encoding generation
        print("[INFO] Updating encodings...")
        subprocess.run(["python", "generate_encodings.py"])
        print("[INFO] Encodings updated.")

        print("[INFO] Restarting program to load new faces...")
        speak("Restarting program to load new faces...")
        self.running = False
        self.processing = False
        video_capture.release()
        self.root.destroy()
        time.sleep(1)
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit()

    def toggle_chat(self):
        if self.chat_visible:
            self.chat_frame.pack_forget()
            self.toggle_button.config(text="Show Chat")
            self.video_frame.pack_forget()
            self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.chat_visible = False
        else:
            self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
            self.toggle_button.config(text="Hide Chat")
            self.video_frame.pack_forget()
            self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
            self.chat_visible = True

    def update_chat(self, message):
        if self.processing:
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.insert(tk.END, message + "\n")
            self.chat_text.see(tk.END)
            self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def capture_video(self):
        while self.running and self.processing:
            if not self.recognizing and not self.registering:
                time.sleep(0.1)
                continue
            ret, frame = video_capture.read()
            if not ret:
                break
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            time.sleep(0.01)

    def process_faces(self):
        global cooldown_until
        while self.running and self.processing:
            if time.time() < cooldown_until:
                time.sleep(0.1)
                continue

            if not self.recognizing or self.registering:
                time.sleep(0.1)
                continue

            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            if self.frame_count % FRAME_PROCESS_INTERVAL == 0:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, locations)

                face_names = []
                for enc in encodings:
                    name = "Unknown"
                    distances = face_recognition.face_distance(known_face_encodings, enc)
                    if distances.size > 0 and np.min(distances) < TOLERANCE:
                        name = known_face_names[np.argmin(distances)]
                    face_names.append(name)

                if face_names:
                    if face_names[0] != "Unknown":
                        self.last_detected_name = face_names[0]
                        self.last_detection_time = time.time()
                        self.current_person = self.last_detected_name
                        self.chatting = True
                        self.greeted = False
                        self.recognizing = False
                    else:
                        self.last_detected_name = "Unknown"
                        self.last_detection_time = time.time()

                elif time.time() - self.last_detection_time > DETECTION_TIMEOUT:
                    self.last_detected_name = None
                    self.chatting = False
                    self.greeted = False
                    self.current_person = None
                    self.recognizing = True

                for (top, right, bottom, left), name in zip(locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1)
                    if name == "Unknown":
                        cv2.putText(frame, "Press 'r' to register", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            time.sleep(0.05)

    def update_video(self):
        if not self.running or not self.processing:
            return

        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(10, self.update_video)
            return

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 480)))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)
        self.root.after(33, self.update_video)

    def handle_chat(self):
        global cooldown_until
        while self.running and self.processing:
            if self.chatting and self.current_person:
                if not self.greeted:
                    greeting = f"Hello {self.current_person}! How are you today? Would you like to know anything about Adroitent?"
                    self.update_chat(f"Assistant: {greeting}")
                    speak(greeting)
                    self.greeted = True

                response_start_time = time.time()
                while self.chatting and self.processing and self.current_person:
                    self.update_chat("Assistant: Listening...")
                    user_input = speech_to_text(self)

                    if not self.processing:
                        break

                    if user_input:
                        self.update_chat(f"You: {user_input}")
                        if "stop" in user_input.lower():
                            response = f"Goodbye, {self.current_person}!"
                            self.update_chat(f"Assistant: {response}")
                            speak(response)
                            self.chatting = False
                            self.current_person = None
                            self.last_detected_name = None
                            self.greeted = False
                            self.recognizing = True
                            cooldown_until = time.time() + 10
                            self.clear_chat()
                        else:
                            response = chatbot_response(user_input, self.current_person)
                            self.update_chat(f"Assistant: {response}")
                            speak(response)
                            response_start_time = time.time()
                    else:
                        if time.time() - response_start_time > RESPONSE_TIMEOUT:
                            self.update_chat(f"Assistant: No response from {self.current_person}.")
                            speak(f"No response from {self.current_person}.")
                            self.chatting = False
                            self.current_person = None
                            self.last_detected_name = None
                            self.greeted = False
                            self.recognizing = True
                            self.clear_chat()
                            break

            time.sleep(0.1)

    def send_text_input(self):
        user_input = self.input_entry.get().strip()
        if user_input and self.current_person:
            self.update_chat(f"You: {user_input}")
            self.input_entry.delete(0, tk.END)
            if "stop" in user_input.lower():
                response = f"Goodbye, {self.current_person}!"
                self.update_chat(f"Assistant: {response}")
                speak(response)
                self.chatting = False
                self.current_person = None
                self.last_detected_name = None
                self.greeted = False
                self.recognizing = True
                self.clear_chat()
            else:
                response = chatbot_response(user_input, self.current_person)
                self.update_chat(f"Assistant: {response}")
                speak(response)

    def on_closing(self):
        self.running = False
        self.processing = False
        video_capture.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceChatbotApp(root)
    threading.Thread(target=app.handle_chat, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    main()
