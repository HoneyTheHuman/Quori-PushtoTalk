import sounddevice as sd  # For recording audio from the mic
from scipy.io.wavfile import write  # To save WAV files
import whisper  # OpenAI's Whisper model for speech-to-text
from vlm_basics import TextToText  # Your custom GPT-4o text class

def record_audio(filename='output.wav', seconds=10, sample_rate=16000):
    """
    Records audio using your default microphone and saves it to a .wav file.
    """
    print(" Recording...")
    
    # Start recording audio for the specified number of seconds
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    
    sd.wait()  # Wait until recording is finished
    write(filename, sample_rate, recording)  # Save the audio to a .wav file
    
    print(f" Saved audio to {filename}")

def transcribe_audio(wav_path='output.wav', txt_path='output.txt'):
    """
    Transcribes speech from the .wav file using Whisper and saves the text to a .txt file.
    """
    print(" Transcribing with Whisper...")
    
    model = whisper.load_model("base")  # Load the Whisper model (can change to "tiny", "small", etc.)
    result = model.transcribe(wav_path)  # Transcribe the audio
    
    # Print what Whisper heard
    print("\n Whisper thinks you said:")
    print(result["text"])
    print()

    # Save the transcription to a file
    with open(txt_path, 'w') as f:
        f.write(result["text"])
    
    print(" Transcription complete!")
    return txt_path  # Return the filename for the next step

def generate_response_from_text(txt_path, system_prompt):
    """
    Sends the transcribed text to GPT-4o using your TextToText class and prints the AI's response.
    """
    print(" Generating response from OpenAI...")
    
    t2t = TextToText()  # Create an instance of your GPT-4o wrapper
    response = t2t.text_to_text(system_filename=txt_path, system_prompt=system_prompt)
    
    print(" GPT-4o says:\n", response)

if __name__ == "__main__":
    # Step 1: Record your voice for 5 seconds and save it to 'output.wav'
    record_audio()
    
    # Step 2: Transcribe 'output.wav' into text and save it to 'output.txt'
    txt_file = transcribe_audio()

    # Step 3: Choose a prompt style for GPT-4o
    # act_mode = "You're a supportive bot. Always agree with the user and build on their point, keep it under 20 words"
    act_mode = "You are a disagreement bot. Whatever the user says to you, you should disagree and explain why,keep it under 20 words."

    # Step 4: Send the transcription to GPT-4o and print the result
    generate_response_from_text(txt_file, act_mode)

