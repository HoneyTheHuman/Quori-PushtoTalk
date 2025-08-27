import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from vlm_basics1 import TextToText
from gtts import gTTS
import os
import time
from datetime import datetime
from evdev import InputDevice, categorize, ecodes, list_devices
from select import select
import rospy
from std_srvs.srv import Empty

# === CONFIGURATION ===
# Update these if your button devices change
event_a_path = '/dev/input/event5'
event_b_path = '/dev/input/event8'

# === LOGGING ===
turn_counter = 0
LOG_FILE = os.path.expanduser("~/KityRainbowGoddessstuff/conversation_log.txt")
COUNT_LOG_FILE = os.path.expanduser("~/KityRainbowGoddessstuff/button_press_counts.txt")
press_counts = {
    "Person A": 0,
    "Person B": 0
}

# === SETUP BUTTON DEVICES ===
button1 = InputDevice(event_a_path)
button2 = InputDevice(event_b_path)
devices = {
    button1.fd: ("Person A", button1),
    button2.fd: ("Person B", button2)
}

# === ROS SETUP ===
rospy.init_node('quori_interaction_node', anonymous=True)
rospy.wait_for_service('/default_face')
rospy.wait_for_service('/thinking_face')
rospy.wait_for_service('/talking_face')
show_default_face = rospy.ServiceProxy('/default_face', Empty)
show_thinking_face = rospy.ServiceProxy('/thinking_face', Empty)
show_talking_face = rospy.ServiceProxy('/talking_face', Empty)

# === AUDIO / TEXT FUNCTIONS ===
def record_audio(filename='output.wav', seconds=10, sample_rate=16000):
    print(" Recording...")
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, recording)
    print(f" Saved audio to {filename}")

def transcribe_audio(wav_path='output.wav', txt_path='output.txt'):
    print(" Transcribing with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path)
    print("\n Whisper thinks you said:")
    print(result["text"])
    print()
    with open(txt_path, 'w') as f:
        f.write(result["text"])
    return txt_path

def speak(text, filename='response.mp3'):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    os.system(f"mpg123 {filename} && rm {filename}")

def generate_response_from_text(txt_path, system_prompt):
    print(" Generating response from OpenAI...")
    with open(txt_path, 'r') as f:
        user_input = f.read().strip()
    print(" User said:", user_input)
    t2t = TextToText()
    response = t2t.text_to_text(system_filename=None, system_prompt=system_prompt, user_prompt=user_input)
    print(" GPT-4o says:\n", response)
    return user_input, response

def log_interaction(turn, speaker, user_input, quori_response):
    try:
        with open(LOG_FILE, "a") as log:
            log.write(f"Turn #{turn} - {speaker} at {datetime.now().strftime('%H:%M:%S')}\n")
            log.write(f"User said: {user_input}\n")
            log.write(f"Quori said: {quori_response}\n")
            log.write("-" * 40 + "\n")
        print(f" Turn #{turn} logged to {LOG_FILE}")
    except Exception as e:
        print(f" Failed to write to log: {e}")

def log_press_counts():
    try:
        with open(COUNT_LOG_FILE, "w") as count_log:
            count_log.write("🔢 Button Press Totals:\n")
            for speaker, count in press_counts.items():
                count_log.write(f"{speaker}: {count} presses\n")
        print(f" Updated press counts saved to {COUNT_LOG_FILE}")
    except Exception as e:
        print(f" Failed to write press counts: {e}")

def write_log_header():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "a") as log:
            log.write(" New Quori Logging Session - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log.write("=" * 40 + "\n")

# === MAIN LOOP ===
if __name__ == "__main__":
    print("🟢 Quori is ready.")
    print("➡️ Press USB Button A or B to start. Ctrl+C to exit.\n")
    write_log_header()
    show_default_face()

    act_mode = "You are a frustrated bot. Always begin responses with Ugh, Fine,, or Seriously,. Then give a concise answer under 30 words" "and explain why in a slightly annoyed tone."


    while True:
        r, _, _ = select(devices.keys(), [], [])
        for fd in r:
            speaker_name, device = devices[fd]
            for event in device.read():
                if event.type == ecodes.EV_KEY and event.value == 1:
                    turn_counter += 1
                    press_counts[speaker_name] += 1

                    print(f"\n🟨 {speaker_name} pressed button! Turn #{turn_counter} begins...\n")
                    print(f"🔢 {speaker_name} has pressed the button {press_counts[speaker_name]} times total.")

                    show_thinking_face()
                    record_audio()
                    txt_file = transcribe_audio()

                    show_thinking_face()
                    user_input, quori_response = generate_response_from_text(txt_file, act_mode)

                    show_talking_face()
                    speak(quori_response)

                    log_interaction(turn_counter, speaker_name, user_input, quori_response)
                    log_press_counts()

                    show_default_face()
                    time.sleep(0.5)
                    print(f"\n🟢 Quori is ready again. Press your button for the next turn.")
