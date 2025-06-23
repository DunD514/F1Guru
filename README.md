F1Guru is a voice-enabled AI assistant tailored specifically for Formula 1 fans. It combines real-time speech recognition (via Whisper), intelligent responses (via Google's Gemini API), and natural text-to-speech output (via pyttsx3 or Azure) to create an engaging and informative experience for users who want to talk about F1.

F1Guru supports interactive trivia, audio feedback, personalized preferences (favorite driver/team), and different user modes like:

Beginner Mode – Simplified explanations

Fan Mode – Rich and engaging F1 talk

Tech Nerd Mode – Deep technical breakdowns

🏗️ Features
🎙️ Voice-controlled interface using faster-whisper

💬 Generative F1 knowledge via Gemini 1.5 Flash

🔊 Real-time speech output using pyttsx3

🔁 Trivia Mode with scoring and hints

📁 User preference persistence (JSON)

🎧 Audio clips for immersive experience (e.g., engine sounds, team radio)

🛠️ Modular and extensible Python class structure

📜 Command phrases like:

"switch to beginner mode"

"set favorite driver Max Verstappen"

"quiz me"

"my driver"

"exit trivia"

🛠 Tech Stack
Python 3

Faster-Whisper – Local ASR

Google Generative AI (Gemini)

pyttsx3 – Offline TTS

pygame – For sound effects

sounddevice & scipy – Audio input/output

🚀 How It Works
User speaks a query through microphone (e.g., “What’s DRS in F1?”).

Whisper transcribes the speech to text.

Gemini generates an intelligent F1-specific response based on context and mode.

The response is spoken aloud using a TTS engine.

If in trivia mode or certain keywords are triggered, the assistant also plays fun sound effects.

📦 Setup Instructions
bash
Copy
Edit
git clone https://github.com/yourusername/F1Guru.git
cd F1Guru
pip install -r requirements.txt
python f1guru.py
💡 Make sure you have audio input/output devices configured. Also, replace the Gemini API key in the script for your own usage.

🔐 Note
For privacy and security, avoid hardcoding your Gemini API key. Use environment variables or external config files in production.
