#please replace the placeholder with your api key 
import os
import json
import random
import pygame
import pyttsx3
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
import tempfile
import google.generativeai as genai
import logging
from collections import deque

class F1GuruAssistant:
    def __init__(self):
        # === Environment Setup ===
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # === Gemini API Setup ===
        self.model = self.initialize_gemini()
        
        # === TTS Setup ===
        self.tts = self.initialize_tts()
        
        # === Audio Clips Setup ===
        self.audio_clips = self.initialize_audio_clips()
        
        # === Whisper ASR Setup ===
        self.whisper_model = self.initialize_whisper()
        
        # === User Preferences and Modes ===
        self.user_prefs = {"favorite_driver": None, "favorite_team": None}
        self.prefs_file = "user_prefs.json"
        self.load_prefs()
        
        self.current_mode = "Fan Mode"  # Default mode
        self.trivia_mode = False
        self.trivia_score = 0
        self.conversation_history = deque(maxlen=5)  # Last 5 exchanges
        
        self.trivia_questions = [
            {"q": "Who won the 2024 F1 Drivers' Championship?", "a": "Max Verstappen", "hint": "He dominated with Red Bull."},
            {"q": "What is the name of the Silverstone circuit's famous corner?", "a": "Copse", "hint": "It’s a high-speed turn in the UK."},
            {"q": "How many points does P1 get in an F1 race?", "a": "25", "hint": "It’s the top score."}
        ]

    def initialize_gemini(self):
        try:
            genai.configure(api_key="this has been removed for safety pouroses ")  # Hardcoded Gemini API key
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "You are F1Guru, an expert Formula 1 assistant with a voice interface. Provide accurate, engaging, and concise answers about Formula 1, "
                    "covering rules, teams, drivers, tracks, history, aerodynamics, race strategies, and fan culture. Tailor responses based on the user’s mode: "
                    "Beginner Mode (simple explanations), Fan Mode (detailed, engaging), or Tech Nerd Mode (technical details). For non-F1 queries, respond politely "
                    "and steer back to F1. Keep responses under 160 characters for SMS compatibility when possible, or expand for voice. If asked about your voice, "
                    "clarify you use a synthetic voice to deliver F1 insights. Example: 'Explain DRS' → 'DRS lets drivers reduce drag to overtake in zones.' "
                    "Example: 'Who is Max Verstappen?' → 'Max Verstappen, Dutch F1 driver, 3-time champion (2021-2023) with Red Bull.'"
                )
            )
            self.logger.info("Gemini API initialized successfully")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    def initialize_tts(self):
        """Initialize text-to-speech engine with best available voice"""
        def select_best_voice(engine):
            """Select a high-quality English voice, preferring Microsoft David, Zira, or Hazel."""
            voices = engine.getProperty('voices')
            preferred_voices = ['David', 'Zira', 'Hazel']
            self.logger.info("Available voices:")
            for voice in voices:
                self.logger.info(f"Voice ID: {voice.id}, Name: {voice.name}, Language: {voice.languages}")
            for voice in voices:
                for pref in preferred_voices:
                    if pref in voice.name and 'en' in ''.join(voice.languages).lower():
                        return voice.id
            for voice in voices:
                if 'en' in ''.join(voice.languages).lower():
                    return voice.id
            return voices[0].id if voices else None

        try:
            tts = pyttsx3.init()
            tts.setProperty('rate', 160)
            tts.setProperty('volume', 1.0)
            voice_id = select_best_voice(tts)
            if voice_id:
                tts.setProperty('voice', voice_id)
                self.logger.info(f"Selected voice ID: {voice_id}")
            else:
                self.logger.warning("No suitable voice found, using default voice")
            self.logger.info("TTS initialized successfully")
            return tts
        except Exception as e:
            self.logger.error(f"TTS initialization failed: {e}")
            raise

    def initialize_audio_clips(self):
        """Initialize audio clips dictionary and pygame mixer"""
        try:
            pygame.mixer.init()
            audio_clips = {
                "engine": "audio_clips/engine_v6.mp3",
                "team_radio": "audio_clips/team_radio_verstappen.mp3",
                "correct": "audio_clips/correct_answer.mp3",
                "incorrect": "audio_clips/incorrect_answer.mp3"
            }
            for clip in audio_clips.values():
                if not os.path.exists(clip):
                    self.logger.warning(f"Audio clip not found: {clip}")
            self.logger.info("Pygame audio initialized successfully")
            return audio_clips
        except Exception as e:
            self.logger.error(f"Audio clips initialization failed: {e}")

    def initialize_whisper(self):
        """Initialize Whisper speech recognition model"""
        try:
            print("🔈 Loading Whisper...")
            model = WhisperModel("base.en", compute_type="int8", device="cpu")
            self.logger.info("Whisper model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Whisper initialization failed: {e}")
            raise

    def load_prefs(self):
        """Load user preferences from file"""
        if os.path.exists(self.prefs_file):
            try:
                with open(self.prefs_file, 'r') as f:
                    self.user_prefs.update(json.load(f))
                self.logger.info("Loaded user preferences")
            except Exception as e:
                self.logger.error(f"Failed to load user preferences: {e}")

    def save_prefs(self):
        """Save user preferences to file"""
        try:
            with open(self.prefs_file, 'w') as f:
                json.dump(self.user_prefs, f)
            self.logger.info("Saved user preferences")
        except Exception as e:
            self.logger.error(f"Failed to save user preferences: {e}")

    def play_audio_clip(self, clip_key):
        """Play specified audio clip if available"""
        try:
            if clip_key in self.audio_clips and os.path.exists(self.audio_clips[clip_key]):
                pygame.mixer.music.load(self.audio_clips[clip_key])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                self.logger.info(f"Played audio clip: {clip_key}")
        except Exception as e:
            self.logger.error(f"Failed to play audio clip {clip_key}: {e}")

    def handle_command(self, user_input):
        """Handle special commands before processing with Gemini"""
        user_input_lower = user_input.lower()
        commands = {
            "f1 update": "Latest F1 news: 2025 season ongoing, next race is Silverstone, July 6. Want details?",
            "race calendar": "2025 F1 calendar includes 24 races. Next: Silverstone, July 6. Want the full schedule?",
            "driver stats": "Top drivers 2025: Verstappen leads, followed by Norris and Leclerc. Want stats for a specific driver?"
        }
        
        # Sort commands by length to match longer phrases first
        sorted_commands = sorted(commands.items(), key=lambda x: len(x[0]), reverse=True)
        for cmd, response in sorted_commands:
            if cmd in user_input_lower:
                return response
                
        if "switch to beginner mode" in user_input_lower:
            self.current_mode = "Beginner Mode"
            return "Switched to Beginner Mode. I'll keep explanations simple!"
        if "switch to fan mode" in user_input_lower:
            self.current_mode = "Fan Mode"
            return "Switched to Fan Mode. Ready for some F1 passion!"
        if "switch to tech nerd mode" in user_input_lower:
            self.current_mode = "Tech Nerd Mode"
            return "Switched to Tech Nerd Mode. Let’s dive into the tech details!"
        if "quiz me" in user_input_lower:
            self.trivia_mode = True
            self.trivia_score = 0
            return self.start_trivia()
        if "set favorite driver" in user_input_lower:
            driver = user_input_lower.replace("set favorite driver", "").strip()
            self.user_prefs["favorite_driver"] = driver.title() or self.user_prefs["favorite_driver"]
            self.save_prefs()
            return f"Favorite driver set to {self.user_prefs['favorite_driver']}!"
        if "set favorite team" in user_input_lower:
            team = user_input_lower.replace("set favorite team", "").strip()
            self.user_prefs["favorite_team"] = team.title() or self.user_prefs["favorite_team"]
            self.save_prefs()
            return f"Favorite team set to {self.user_prefs['favorite_team']}!"
        if "forget my preferences" in user_input_lower:
            self.user_prefs["favorite_driver"] = None
            self.user_prefs["favorite_team"] = None
            self.save_prefs()
            return "Preferences cleared!"
        if "my driver" in user_input_lower and self.user_prefs["favorite_driver"]:
            return f"Your driver {self.user_prefs['favorite_driver']} is racing strong! Want their latest results?"
        return None

    def start_trivia(self):
        """Start trivia mode with a random question"""
        if self.trivia_questions:
            question = random.choice(self.trivia_questions)
            self.conversation_history.append(("Quiz me", f"Question: {question['q']} (Hint: {question['hint']})"))
            return f"Question: {question['q']} (Hint: {question['hint']})"
        return "No questions available. Want to exit trivia?"

    def check_trivia_answer(self, user_input, last_question):
        """Check trivia answer and update score"""
        for q in self.trivia_questions:
            if q["q"] in last_question and q["a"].lower() in user_input.lower():
                self.trivia_score += 1
                self.play_audio_clip("correct")
                response = f"Correct! Score: {self.trivia_score}. Next question?"
                question = random.choice(self.trivia_questions)
                self.conversation_history.append((user_input, response))
                self.conversation_history.append(("Next question", f"Question: {question['q']} (Hint: {question['hint']})"))
                return response + f" Question: {question['q']} (Hint: {question['hint']})"
        self.play_audio_clip("incorrect")
        response = f"Wrong, it’s {q['a']}. Score: {self.trivia_score}. Try another?"
        question = random.choice(self.trivia_questions)
        self.conversation_history.append((user_input, response))
        self.conversation_history.append(("Next question", f"Question: {question['q']} (Hint: {question['hint']})"))
        return response + f" Question: {question['q']} (Hint: {question['hint']})"

    def listen_and_transcribe(self, duration=5, sample_rate=16000):
        """Listen to microphone and transcribe speech to text"""
        try:
            print("🎤 Listening...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav.write(f.name, sample_rate, audio)
                segments, _ = self.whisper_model.transcribe(f.name)
                for segment in segments:
                    text = segment.text.strip()
                    self.logger.info(f"Transcribed: {text}")
                    return text
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None

    def generate_response(self, user_input):
        """Generate response using Gemini API"""
        try:
            # Adjust prompt based on mode
            mode_prompt = {
                "Beginner Mode": "Explain in simple terms, suitable for someone new to F1.",
                "Fan Mode": "Provide detailed and engaging answers for an F1 fan.",
                "Tech Nerd Mode": "Include technical details like tire degradation, fuel loads, or DRS mechanics."
            }
            # Build prompt with conversation history and mode
            history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.conversation_history])
            prompt = f"{mode_prompt[self.current_mode]}\n{history_text}\nUser: {user_input}\nAssistant:"
            
            # Check for audio clip triggers
            if any(keyword in user_input.lower() for keyword in ["engine", "v6", "turbo"]):
                self.play_audio_clip("engine")
            elif any(keyword in user_input.lower() for keyword in ["radio", "team radio"]):
                self.play_audio_clip("team_radio")
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            )
            reply = response.text.strip()
            if reply.startswith("Assistant:"):
                reply = reply[len("Assistant:"):].strip()
            
            # Personalize with user preferences
            if self.user_prefs["favorite_driver"] and self.user_prefs["favorite_driver"].lower() in user_input.lower():
                reply += f" By the way, your favorite driver {self.user_prefs['favorite_driver']} is awesome!"
            
            self.conversation_history.append((user_input, reply))
            self.logger.info(f"Generated response: {reply}")
            return reply
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return "Sorry, I couldn’t process that. Try asking about F1, like 'What is DRS?'"

    def speak(self, text):
        """Convert text to speech"""
        try:
            print(f"🗣 Responding: {text}")
            self.tts.say(text)
            self.tts.runAndWait()
            self.logger.info("Speech delivered successfully")
        except Exception as e:
            self.logger.error(f"Speech failed: {e}")

    def run(self):
        """Main assistant loop"""
        print("🏎️ F1Guru Assistant Started! Ask about Formula 1, say 'Quiz me', or 'exit' to quit.")
        self.speak("Welcome to F1Guru! Ready to talk Formula 1?")
        while True:
            try:
                user_input = self.listen_and_transcribe()
                if not user_input:
                    self.speak("I didn’t hear anything. Please try again.")
                    continue

                print("👂 Heard:", user_input)
                self.logger.info(f"User input: {user_input}")

                if any(kw in user_input.lower() for kw in ["exit", "quit", "bye"]):
                    self.speak("Thanks for racing with F1Guru! Goodbye!")
                    self.logger.info("Assistant terminated by user")
                    break

                if self.trivia_mode:
                    last_question = self.conversation_history[-1][1] if self.conversation_history else ""
                    if "exit trivia" in user_input.lower():
                        self.speak(f"Exiting trivia. Final score: {self.trivia_score}. Great job!")
                        self.trivia_mode = False
                        self.trivia_score = 0
                        continue
                    reply = self.check_trivia_answer(user_input, last_question)
                else:
                    command_response = self.handle_command(user_input)
                    reply = command_response if command_response else self.generate_response(user_input)

                self.speak(reply)

            except KeyboardInterrupt:
                self.speak("F1Guru shutting down. Goodbye!")
                self.logger.info("Assistant terminated via KeyboardInterrupt")
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}")
                self.speak("Sorry, something went wrong. Ask me about F1, like 'Who won in 2024?'")
                continue

if __name__ == "__main__":
    assistant = F1GuruAssistant()
    assistant.run()