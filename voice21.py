import face_recognition
import cv2
import numpy as np
import pickle
import speech_recognition as sr
import pyttsx3
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
import os
import requests
from queue import Queue
import queue
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Groq API Configuration ===
GROQ_API_KEY = "xxxxxxxxxx"
GROQ_MODEL = "llama3-70b-8192"

# === Custom System Prompt ===
system_prompt = (
    "As an Adroitent assistant, provide concise, accurate, one-line responses about Adroitent, Inc., "
    "a trusted digital transformation partner with 19 years of expertise and over 500 skilled associates "
    "worldwide, offering Software Engineering, AI, SaaS, ERP, Cloud, and Business Intelligence Solutions. "
    "Emphasize our SEI CMMI Level 3 accreditation, ISO 9001 and 27001 certifications, and leadership: "
    "Partha Bommireddy, co-founder and President, drives growth with strategic AI expertise; Srinath, co-founder "
    "and IT advisor to Andhra Pradesh, leverages global experience; and Sriram, VP of Delivery, leads enterprise-scale programs. "
    "Answer questions solely based on this information, reflecting our commitment to agility, innovation, and quality."
)

# === Initialize Speech Recognizer ===
recognizer = sr.Recognizer()
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = True

# === Initialize TTS Engine ===
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if not voices:
        raise Exception("No voices available for TTS engine.")
    voice_set = False
    for voice in voices:
        try:
            engine.setProperty('voice', voice.id)
            engine.setProperty('volume', 1.0)
            engine.setProperty('rate', 150)
            voice_set = True
            logging.info(f"TTS voice set to: {voice.id}")
            break
        except Exception as e:
            logging.warning(f"Failed to set voice {voice.id}: {str(e)}")
            continue
    if not voice_set:
        raise Exception("No valid TTS voices could be set.")
except Exception as e:
    logging.error(f"Failed to initialize TTS engine: {str(e)}")
    engine = None

# === TTS Queue and Thread ===
tts_queue = Queue()
tts_thread_running = True
tts_thread = None

def tts_worker():
    while tts_thread_running:
        try:
            text, app = tts_queue.get(timeout=1.0)
            if engine:
                engine.say(text)
                engine.runAndWait()
            else:
                if app:
                    app.update_chat(f"System: Audio output unavailable. Message: {text}")
            tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"TTS worker error: {str(e)}")
            if app:
                app.update_chat(f"System: TTS error: {str(e)}. Message: {text}")

# Start TTS thread if engine is initialized
if engine:
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    logging.info("TTS thread started.")
else:
    logging.error("TTS thread not started due to engine initialization failure.")

def speak(text, app=None):
    if not engine or not tts_thread or not tts_thread.is_alive():
        logging.error("Cannot speak: TTS engine or thread not available.")
        if app:
            app.update_chat(f"System: Audio output is not working. Message: {text}")
        return
    try:
        tts_queue.put((text, app))
    except Exception as e:
        logging.error(f"Error adding to TTS queue: {str(e)}")
        if app:
            app.update_chat(f"System: Failed to queue audio: {str(e)}. Message: {text}")

# === Load Face Encodings ===
ENCODINGS_FILE = "encodings.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

# === Initialize Video Capture ===
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Error: Could not open webcam.")

TOLERANCE = 0.4  # Lowered for stricter matching
DETECTION_TIMEOUT = 10
RESPONSE_TIMEOUT = 10
cooldown_until = 0
FRAME_PROCESS_INTERVAL = 5

def speech_to_text(app=None, retries=3, timeout=7, phrase_time_limit=10, noise_adjust_duration=3.0, debug=False):
    if app and not app.processing:
        return None
    while not tts_queue.empty():
        tts_queue.get()
        tts_queue.task_done()
    time.sleep(0.5)
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=noise_adjust_duration)
        time.sleep(0.5)
        for attempt in range(retries):
            try:
                if app and attempt == 0:
                    app.update_chat("Assistant: Listening... Please speak clearly.")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                text = recognizer.recognize_google(audio).strip()
                if text:
                    if app and debug:
                        app.update_chat(f"Assistant: Heard: {text}")
                    return text
                else:
                    if app:
                        app.update_chat("Assistant: I didn’t catch that. Trying again...")
            except sr.UnknownValueError:
                if app:
                    app.update_chat("Assistant: Sorry, I didn’t understand that.")
            except sr.RequestError:
                if app:
                    app.update_chat("Assistant: I’m having trouble with speech recognition. Please ensure your internet connection is stable.")
                break
            except Exception as e:
                if app:
                    app.update_chat(f"Assistant: An error occurred during speech recognition: {str(e)}")
    return None

def prompt_keyboard_confirmation(app, name):
    confirm_msg = f"Did you say {name}? Press 'y' to confirm or 'n' to retry (timeout in 10 seconds)."
    if app:
        app.update_chat(f"Assistant: {confirm_msg}")
    speak(confirm_msg, app)

    start_time = time.time()
    while time.time() - start_time < 10:
        for event in app.root.event_generate('<KeyPress>'):
            if event.char.lower() == 'y':
                return True
            elif event.char.lower() == 'n':
                return False
        time.sleep(0.1)
    if app:
        app.update_chat("Assistant: No keyboard input received. Let’s try the name again.")
    speak("No keyboard input received. Let’s try the name again.", app)
    return False

def prompt_name(app=None):
    max_attempts = 2
    attempt = 0

    while attempt < max_attempts:
        if attempt == 0:
            msg = "I don't recognize you. Please say your name clearly."
        else:
            msg = "Let's try again. Please say your name clearly."
        if app:
            app.update_chat(f"Assistant: {msg}")
        speak(msg, app)
        time.sleep(2.0)

        name = speech_to_text(app, debug=False)
        if not name:
            attempt += 1
            continue

        max_confirm_attempts = 3
        for confirm_attempt in range(max_confirm_attempts):
            confirm_msg = f"Did you say {name}? Say 'confirm' or 'no'."
            if app:
                app.update_chat(f"Assistant: {confirm_msg}")
            speak(confirm_msg, app)
            time.sleep(3.0)

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=2.0)
                time.sleep(0.5)
                try:
                    audio_confirm = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                    confirm_text = recognizer.recognize_google(audio_confirm).lower()
                    if app:
                        app.update_chat(f"Assistant: Heard: {confirm_text}")
                    if any(word in confirm_text for word in ["confirm", "yes", "yeah", "yep"]):
                        return name
                    elif any(word in confirm_text for word in ["no", "nah", "nope", "negative", "not", "denied", "reject"]):
                        attempt += 1
                        break
                    else:
                        error_msg = "I didn’t hear 'confirm' or 'no'. Please try again."
                        if app:
                            app.update_chat(f"Assistant: {error_msg}")
                        speak(error_msg, app)
                except Exception as e:
                    error_msg = f"Error confirming name: {str(e)}. Please try again."
                    if app:
                        app.update_chat(f"Assistant: {error_msg}")
                    speak(error_msg, app)
        else:
            if prompt_keyboard_confirmation(app, name):
                return name
            attempt += 1
            retry_msg = "I couldn’t confirm your name. Let’s try again."
            if app:
                app.update_chat(f"Assistant: {retry_msg}")
            speak(retry_msg, app)

    final_msg = "No name provided. Resetting system."
    if app:
        app.update_chat(f"Assistant: {final_msg}")
    speak(final_msg, app)
    return None

def clear_face_encoding(name, app):
    global known_face_encodings, known_face_names
    if name in known_face_names:
        index = known_face_names.index(name)
        known_face_encodings.pop(index)
        known_face_names.pop(index)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        person_dir = os.path.join("known_faces", name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        app.update_chat(f"Assistant: Cleared face data for {name}.")
        speak(f"Cleared face data for {name}. Please re-register.", app)
        return True
    return False

def register_unknown_face(frame, app, unknown_index=0):
    person_name = prompt_name(app)
    if not person_name:
        app.new_person_detected = False
        app.new_person_name = None
        app.recognizing = True
        return None

    # Check if name exists and clear old data if confirmed
    if person_name in known_face_names:
        app.update_chat(f"Assistant: Name {person_name} already exists. Say 'overwrite' to replace it.")
        speak(f"Name {person_name} already exists. Say 'overwrite' to replace it.", app)
        time.sleep(2.0)
        response = speech_to_text(app, debug=False)
        if response and "overwrite" in response.lower():
            clear_face_encoding(person_name, app)
        else:
            app.update_chat("Assistant: Registration cancelled. Using a different name.")
            speak("Registration cancelled. Using a different name.", app)
            return None

    known_dir = "known_faces"
    base_person_dir = os.path.join(known_dir, person_name)
    person_dir = base_person_dir
    suffix = 1
    while os.path.exists(person_dir):
        person_dir = f"{base_person_dir}_{suffix}"
        suffix += 1
    os.makedirs(person_dir, exist_ok=True)

    captured_encodings = []
    capture_count = 0
    max_captures = 15  # Increased for better encoding quality
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts and capture_count < max_captures:
        speak(f"Hold still. Capturing {max_captures} photos for {person_name}. Attempt {attempt + 1} of {max_attempts}.", app)
        app.update_chat(f"Assistant: Capturing {max_captures} images of {person_name} (Attempt {attempt + 1}/{max_attempts})...")
        time.sleep(2.0)

        start_time = time.time()
        timeout = 20
        while capture_count < max_captures and time.time() - start_time < timeout:
            ret, image = video_capture.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            if face_encodings:
                encoding = face_encodings[0]
                distances = face_recognition.face_distance(known_face_encodings, encoding)
                if distances.size == 0 or np.min(distances) >= TOLERANCE:
                    captured_encodings.append(encoding)
                    filename = os.path.join(person_dir, f"{person_name}_{capture_count}.jpg")
                    cv2.imwrite(filename, image)
                    capture_count += 1
                    time.sleep(0.1)
                else:
                    closest_name = known_face_names[np.argmin(distances)]
                    logging.info(f"Face matched to {closest_name} with distance {np.min(distances)}")
                    app.update_chat(f"Assistant: Face matches {closest_name}. Please ensure only the new person is in the frame.")
                    speak(f"Face matches {closest_name}. Please ensure only the new person is in the frame.", app)
                    time.sleep(1)
            else:
                app.update_chat("Assistant: No face detected. Please ensure good lighting and face the camera.")
                speak("No face detected. Please ensure good lighting and face the camera.", app)
                time.sleep(0.5)

        if capture_count < max_captures:
            attempt += 1
            if attempt < max_attempts:
                app.update_chat("Assistant: Not enough faces captured. Let’s try again.")
                speak("Not enough faces captured. Let’s try again.", app)
            else:
                speak("Registration failed, please try again.", app)
                app.update_chat("Assistant: Registration failed after maximum attempts.")
                app.recognizing = True
                return None
        else:
            break

    if captured_encodings:
        average_encoding = np.mean(captured_encodings, axis=0)
        known_face_encodings.append(average_encoding)
        known_face_names.append(person_name)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        speak(f"{person_name}, you have been registered successfully.", app)
        app.update_chat(f"Assistant: {person_name} registered with {max_captures} images.")
        app.new_person_name = person_name
        app.recognizing = False
        return person_name
    else:
        speak("Registration failed. No faces captured.", app)
        app.update_chat("Assistant: Registration failed. Please try again.")
        app.recognizing = True
        return None

def chatbot_response(text, names=["there"]):
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
        return f"Sorry {format_names(names)}, I couldn't think of a response."

def format_names(names):
    if not names:
        return "there"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"

class FaceChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        # State flags
        self.running = True
        self.chatting = False
        self.greeted = False
        self.processing = True
        self.recognizing = True
        self.current_people = []
        self.detected_names = []
        self.last_detection_time = time.time()
        self.raw_frame_queue = Queue(maxsize=50)
        self.processed_frame_queue = Queue(maxsize=50)
        self.new_person_detected = False
        self.new_person_name = None
        self.frame_count = 0
        self.registration_mode = False
        self.last_frame = None
        self.unknown_indices = []

        # GUI layout
        self.title_label = tk.Label(root, text="Adroitent Assistant", font=("Helvetica", 20, "bold"))
        self.title_label.pack(pady=10)

        self.content_frame = tk.Frame(root)
        self.content_frame.pack(expand=True, fill=tk.BOTH)

        self.video_frame = tk.Frame(self.content_frame)
        self.video_frame.pack(expand=True)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.chat_frame = tk.Frame(self.content_frame)
        self.chat_text = tk.Text(self.chat_frame, height=15, wrap=tk.WORD)
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.config(state=tk.DISABLED)
        self.chat_visible = False

        self.toggle_button = tk.Button(root, text="Show Chat", command=self.toggle_chat)
        self.toggle_button.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start threads
        threading.Thread(target=self.capture_video, daemon=True).start()
        threading.Thread(target=self.process_faces, daemon=True).start()
        threading.Thread(target=self.handle_chat, daemon=True).start()

        self.update_video()

    def update_chat(self, message):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, message + "\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete(1.0, tk.END)
        self.chat_text.config(state=tk.DISABLED)
        while not tts_queue.empty():
            tts_queue.get()
            tts_queue.task_done()

    def capture_video(self):
        while self.running and self.processing:
            if not self.recognizing:
                time.sleep(0.1)
                continue
            ret, frame = video_capture.read()
            if not ret:
                break
            try:
                self.raw_frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def process_faces(self):
        global cooldown_until
        while self.running and self.processing:
            if time.time() < cooldown_until or not self.recognizing:
                time.sleep(0.1)
                continue
            try:
                frame = self.raw_frame_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            processed_frame = frame.copy()
            if self.frame_count % FRAME_PROCESS_INTERVAL == 0:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, locations)

                face_names = []
                self.unknown_indices = []
                for i, enc in enumerate(encodings):
                    name = "Unknown"
                    distances = face_recognition.face_distance(known_face_encodings, enc)
                    if distances.size > 0 and np.min(distances) < TOLERANCE:
                        name = known_face_names[np.argmin(distances)]
                        logging.info(f"Matched face to {name} with distance {np.min(distances)}")
                    else:
                        logging.info(f"Face unrecognized, distance to closest match: {np.min(distances) if distances.size > 0 else 'N/A'}")
                    face_names.append(name)
                    if name == "Unknown":
                        self.unknown_indices.append(i)

                if face_names:
                    known_people = [name for name in face_names if name != "Unknown"]
                    unknown_count = face_names.count("Unknown")

                    if known_people:
                        self.detected_names = known_people
                        self.current_people = known_people
                        self.last_detection_time = time.time()
                        self.chatting = True
                        self.greeted = False
                        self.recognizing = False
                    if unknown_count > 0:
                        self.registration_mode = True
                        self.recognizing = False
                        self.new_person_detected = True
                        threading.Thread(target=self.run_registration, args=(frame,), daemon=True).start()
                        continue
                else:
                    self.update_chat("Assistant: No face detected. Please face the camera with good lighting.")
                    time.sleep(1)

            for i, name in enumerate(self.detected_names):
                cv2.putText(processed_frame, name, (10, 30 + i * 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

            try:
                self.processed_frame_queue.put_nowait(processed_frame)
            except queue.Full:
                pass

    def run_registration(self, frame):
        for idx in self.unknown_indices:
            if not self.running or not self.processing:
                break
            self.new_person_name = register_unknown_face(frame, self, unknown_index=idx)
            if self.new_person_name and self.new_person_name not in self.current_people:
                self.current_people.append(self.new_person_name)
                self.detected_names.append(self.new_person_name)
                self.chatting = True
                self.greeted = False
        self.registration_mode = False
        self.unknown_indices = []
        self.recognizing = True

    def update_video(self):
        if not self.running or not self.processing:
            return
        frame = None
        while not self.processed_frame_queue.empty():
            try:
                frame = self.processed_frame_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self.last_frame = frame
        elif self.last_frame is None:
            self.last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.last_frame, "Waiting for video...", (200, 240), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)).resize((640, 480)))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)
        self.root.after(10, self.update_video)

    def handle_chat(self):
        global cooldown_until
        while self.running and self.processing:
            if self.chatting and self.current_people:
                if not self.greeted:
                    formatted_names = format_names(self.current_people)
                    greeting = f"Hello {formatted_names}! How are you {'all' if len(self.current_people) > 1 else ''} today? Would you like to know anything about Adroitent?"
                    self.update_chat(f"Assistant: {greeting}")
                    speak(greeting, self)
                    self.greeted = True
                    time.sleep(2)

                response_start_time = time.time()
                while self.chatting and self.processing and self.current_people:
                    user_input = speech_to_text(self, debug=False)
                    if not self.processing:
                        break
                    if user_input:
                        self.update_chat(f"You: {user_input}")
                        if "stop" in user_input.lower():
                            formatted_names = format_names(self.current_people)
                            response = f"Goodbye, {formatted_names}!"
                            self.update_chat(f"Assistant: {response}")
                            speak(response, self)
                            self.chatting = False
                            self.current_people = []
                            self.detected_names = []
                            self.greeted = False
                            self.recognizing = True
                            cooldown_until = time.time() + 10
                            self.clear_chat()
                        elif "re-register" in user_input.lower():
                            self.update_chat("Assistant: Starting re-registration. Please say your name.")
                            speak("Starting re-registration. Please say your name.", self)
                            self.chatting = False
                            self.recognizing = False
                            self.registration_mode = True
                            self.unknown_indices = [0]  # Assume one face for re-registration
                            threading.Thread(target=self.run_registration, args=(self.last_frame,), daemon=True).start()
                            break
                        else:
                            response = chatbot_response(user_input, self.current_people)
                            self.update_chat(f"Assistant: {response}")
                            speak(response, self)
                            response_start_time = time.time()
                    elif time.time() - response_start_time > RESPONSE_TIMEOUT:
                        formatted_names = format_names(self.current_people)
                        msg = f"No response from {formatted_names}."
                        self.update_chat(f"Assistant: {msg}")
                        speak(msg, self)
                        self.chatting = False
                        self.current_people = []
                        self.detected_names = []
                        self.greeted = False
                        self.recognizing = True
                        self.clear_chat()
                        break
                    time.sleep(0.5)
            time.sleep(0.1)

    def toggle_chat(self, show=None):
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()
        if show is not None:
            self.chat_visible = not show
        if self.chat_visible:
            self.video_frame.pack(expand=True)
            self.chat_visible = False
            self.toggle_button.config(text="Show Chat")
        else:
            self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
            self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            self.chat_visible = True
            self.toggle_button.config(text="Hide Chat")

    def show_chat_message(self, message):
        if not self.chat_visible:
            self.toggle_chat(show=True)
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"System: {message}\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def on_closing(self):
        global tts_thread_running
        tts_thread_running = False
        self.running = False
        self.processing = False
        while not tts_queue.empty():
            tts_queue.get()
            tts_queue.task_done()
        if tts_thread and tts_thread.is_alive():
            tts_thread.join(timeout=1.0)
        video_capture.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceChatbotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
