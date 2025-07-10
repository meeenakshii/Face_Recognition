# 👤🎤💬 Tiny Python AI Assistant

A **tiny Python assistant** that recognizes your face 👤, listens to your voice 🎤, and replies using AI 💬✨

## 🚀 Features

- ✅ **Face Recognition** via webcam (knows who you are!)
- 🎙️ **Voice Input** — speak naturally to the assistant
- 🤖 **AI-Powered Replies** using OpenAI's ChatGPT
- 🔊 **Voice Response** via Google Text-to-Speech (gTTS)
- 🖥️ Simple to run — just launch and chat in real time

## 🧠 How It Works

1. Detects and recognizes your face using OpenCV + face encoding
2. Records your voice using a microphone
3. Sends your speech as a prompt to the AI
4. Speaks back the AI’s response using gTTS

## 🛠️ Requirements

- Python 3.8+
- Webcam
- Microphone
- Internet connection (for OpenAI + gTTS)

## 📦 Installation

```bash
git clone https://github.com/yourusername/face-voice-ai-assistant.git
cd face-voice-ai-assistant
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
