import sounddevice as sd  # For recording audio from the mic
from scipy.io.wavfile import write  # To save WAV files
import whisper  # OpenAI's Whisper model for speech-to-text
from vlm_basics1 import TextToText  # Your custom GPT-4o text class
import pyttsx3  # For text-to-speech
import os  # For file paths

def record_audio(filename='output.wav', seconds=10, sample_rate=16000):
    """
    Records audio using your default microphone and saves it to a .wav file.
    """
    print(" Recording...")
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, recording)
    print(f" Saved audio to {filename}")

def transcribe_audio(wav_path='output.wav', txt_path='output.txt'):
    """
    Transcribes speech from the .wav file using Whisper and saves the text to a .txt file.
    """
    print(" Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path)

    print("\n Whisper thinks you said:")
    print(result["text"])
    print()

    with open(txt_path, 'w') as f:
        f.write(result["text"])

    print(" Transcription complete!")
    return txt_path

def speak(text):
    """
    Converts text to speech using pyttsx3.
    """
    
    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[24].id)  # <<<< Change the index here to test different voices

    engine.setProperty('rate', 100)
    engine.setProperty('volume', 1.0)

    engine.say(text)
    engine.runAndWait()


def generate_response_from_text(txt_path, system_prompt):
    """
    Reads the transcription from a file, sends it to GPT-4o, prints and speaks the response.
    """
    print(" Generating response from OpenAI...")

    #  Read transcription directly from the text file
    with open(txt_path, 'r') as f:
        user_input = f.read().strip()

    print(" User said:", user_input)

    t2t = TextToText()
    response = t2t.text_to_text(system_filename=None, system_prompt=system_prompt, user_prompt=user_input)

    print(" GPT-4o says:\n", response)

    #  Make Quori speak the response
    speak(response)

if __name__ == "__main__":
    # Step 1: Record your voice for 10 seconds
    record_audio()

    # Step 2: Transcribe to text
    txt_file = transcribe_audio()

    # Step 3: Choose a prompt style
    # act_mode = "You're a supportive bot. Always agree with the user and build on their point, keep it under 20 words"
    act_mode = "You are a disagreement bot. Whatever the user says to you, you should disagree and explain why, keep it under 20 words."

    # Step 4: Generate and speak GPT-4o response
    generate_response_from_text(txt_file, act_mode)
