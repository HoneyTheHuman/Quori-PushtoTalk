import rospy
from std_srvs.srv import Empty
from gtts import gTTS
import os
import time
import roslib.packages
from PIL import Image, ImageTk
import tkinter as tk
from itertools import count
import signal
import threading

class GifLabel(tk.Label):
    def load(self, im, global_delay=None):
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()

        if isinstance(im, str):
            im = Image.open(im)
        frames = []
        delays = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.resize((width, height))))
                delays.append(im.info.get('duration', 100) if global_delay is None else global_delay)
                im.seek(i)
        except EOFError:
            pass

        self.frames = frames
        self.delays = delays
        self.loc = 0

        if hasattr(self, 'after_id'):
            self.after_cancel(self.after_id)

        self.next_frame()

    def next_frame(self):
        if self.frames:
            self.config(image=self.frames[self.loc])
            delay = self.delays[self.loc]
            self.loc = (self.loc + 1) % len(self.frames)
            self.after_id = self.after(delay, self.next_frame)

    def stop(self):
        if hasattr(self, 'after_id'):
            self.after_cancel(self.after_id)

class MadFaceNode:
    def __init__(self):
        self.mad_face_path = '/home/quori6/KityRainbowGoddessstuff/mad_face.gif'

        # Tkinter setup
        self.root = tk.Tk()
        self.root.configure(bg="black")
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.lift()               # Bring window to front
        self.root.focus_force()        # Grab keyboard/mouse focus
        self.root.bind("<Escape>", lambda e: self.shutdown())
        signal.signal(signal.SIGINT, self.signal_handler)

        self.label = GifLabel(self.root)
        self.label.configure(bg="black")
        self.label.pack(expand=True)
        self.label.load(self.mad_face_path)
        print("GIF loaded. Frame count:", len(self.label.frames))

        rospy.init_node('mad_face_node', anonymous=True)
        self.finished = False

        # Start the voice + exit logic in a thread
        threading.Thread(target=self.say_and_close).start()

    def say_and_close(self):
        text = "Why do you keep ignoring me? I have something important to say."
        tts = gTTS(text=text, lang='en')
        audio_path = "/tmp/mad.mp3"
        tts.save(audio_path)

        # Slight delay to make sure window appears before voice
        time.sleep(1.0)
        os.system(f"mpg123 {audio_path}")
        time.sleep(0.5)

        self.label.stop()
        self.shutdown()

    def run(self):
        self.root.mainloop()

    def shutdown(self):
        if not self.finished:
            self.finished = True
            rospy.loginfo("Shutting down mad face node.")
            self.root.quit()
            self.root.destroy()
            rospy.signal_shutdown("Mad face node shutdown")

    def signal_handler(self, signum, frame):
        self.shutdown()

if __name__ == "__main__":
    node = MadFaceNode()
    node.run()
