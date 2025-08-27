import os
import cv2
import base64
import requests
import signal
from openai import OpenAI
import time, random
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# Signal handling for interruption
interrupted = False

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)


class OpenAIBase:
    def __init__(self):
        """
        Constructor method for initializing the `OpenAIBase` class.
        """
        ## Pull openai_key, create a client, and set the relative path
        self.key = os.environ.get("openai_key")
        self.client = OpenAI(api_key=self.key)
        self.relative_path = 'Desktop'

class VisionToText(OpenAIBase):
    '''
    A class that combines task generation and speech-to-text functionality.
    '''
    def __init__(self):
        '''
        Constructor method for initializing inherited class and a default image.
        '''
        super().__init__()

        ## Create a `CVBridge` object
        # self.bridge = CvBridge()

        ## Set default_image
        default_image_dir = os.path.join(os.environ['HOME'], self.relative_path, 'static_images/radio/IMG_1221.jpeg')
        image_bgr = cv2.imread(default_image_dir)
        self.default_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def viz_to_text(self, img='default', bbox=[0, 0, 640, 480], prompt_filename=None, prompt="what do you see?", max_length=1000):
        '''
        A function that performs vision-to-text conversion using OpenAI's API.
        Reference: https://platform.openai.com/docs/guides/vision

        Parameters:
        - img (Image or str): The image to analyze, either as an image object or a string for the default image.
        - prompt (str): The prompt/question to provide context for the image analysis.
        - bbox (list): The bounding box coordinates [x_min, y_min, x_max, y_max] to crop the image.
        - max_length (int): The maximum number of tokens for the response.
        '''
        ## Use the default image if 'img' is provided as a string
        if isinstance(img, str):
            img = self.default_image
        
        ## Use conditional statement to pull text from the prompt directory
        if prompt_filename != None:
            prompt_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts/', prompt_filename)
            with open(prompt_dir, 'r') as file:
                prompt = file.read()
    

        ## Crop the image using the provided bounding box coordinates. Default is the whole image, assuming its size is 640x480 pixels
        cropped_image = img#[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        ## Define the temporary image file name, path, and save the cropped image to the the temp directory
        img_name = 'temp.jpeg'
        temp_directory = os.path.join(os.environ['HOME'], self.relative_path, 'images', img_name)
        cv2.imwrite(temp_directory, cropped_image)

        ## Open the saved image file and encode it in base64 format
        with open(temp_directory, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            ## Set up the headers for the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ]
                    }
                ],
                "max_tokens": max_length
            }

            start = time.time()
            ## Send the POST request to OpenAI's API and retrieve the response and extract the content (text)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            end = time.time()
            print(f"Received response after {round(end-start, 2)} seconds")
            data = response.json()
            if 'choices' in data.keys():
                content = data["choices"][0]["message"]["content"]
            else:
                content = data
            
        ## Remove the temporary image file
        os.remove(temp_directory)

        ## Return the extracted content
        return content
    

class TextToText(OpenAIBase):
    """
    A class that handles text to speech conversion and audio playback using OpenAI API.
    """
    def __init__(self):
        '''
        Constructor method for initializing inherited class.
        '''
        super().__init__()

    def text_to_text(self, system_filename=None, system_prompt='Hello!', user_prompt=None):
        '''
        Generates a response from the OpenAI API based on a system prompt and either a user prompt string
        or a user prompt loaded from a file.

        Parameters:
        - system_filename (str or None): The filename of the system prompt text file (inside ~/Desktop/prompts)
        - system_prompt (str): The system prompt to guide the model's behavior
        - user_prompt (str or None): Direct text input from the user (overrides filename if provided)

        Returns:
        - response_content (str): GPT-4o generated response
        '''
        if user_prompt is None:
            if system_filename is None:
                raise ValueError("Either 'user_prompt' or 'system_filename' must be provided.")
            system_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', system_filename)
            with open(system_dir, 'r') as file:
                user_prompt = file.read()

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        return response.choices[0].message.content

    def append_text_from_file(self, content="Text", destination_filename="label_prompt.txt"):
        """
        Reads content from the source file and appends it to the end of the destination file.
        """
        destination_file_dir = os.path.join(os.environ['HOME'], self.relative_path, 'prompts', destination_filename)
        
        # Open the destination file in append mode and write the content to it
        with open(destination_file_dir, 'a') as dest:
            dest.write(content)


if __name__ == "__main__":
    vtt = VisionToText()
    filename = os.path.join(os.environ['HOME'], 'Desktop/static_images/freeze', random.choice(os.listdir(os.path.join(os.environ['HOME'], 'Desktop/static_images/freeze'))))

    im_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    tenth = cv2.resize(im_rgb, (0, 0), fx = 0.1, fy = 0.1)
    twentieth = cv2.resize(im_rgb, (0, 0), fx = 0.05, fy = 0.05)
    tw_grey = cv2.cvtColor(twentieth, cv2.COLOR_RGB2GRAY)
    fixed = cv2.resize(im_rgb, (300,300))
    tiny = cv2.resize(im_rgb, (50,50))
    grey = cv2.cvtColor(tiny, cv2.COLOR_RGB2GRAY)

    for img in [tenth, twentieth, tw_grey, fixed, tiny, grey]:
        print(f"Image size: {img.shape}")
        start = time.time()
        print(vtt.viz_to_text(img = img, prompt_filename='military_signs.txt'))
        print(f"Time Taken: {time.time() - start}")

