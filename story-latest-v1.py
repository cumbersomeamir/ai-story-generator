#pip install openai elevenlabs pydub moviepy

import openai
import time
import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import requests
import json
import re
from pydub import AudioSegment
from moviepy.editor import *
import shutil  # For deleting folder contents

os.getenv("OPENAI_API_KEY")
api_key = os.getenv("ELEVENLABS_API_KEY")

# Initialize the client
client = openai.OpenAI()

from openai import OpenAI
client = OpenAI()


topic = input("Enter a topic ")
num_frames = input("Enter the number of frames to generate ")

def generate_text(topic, num_frames):

    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": "You are a super creative non-fiction story writer"},
        {"role": "user", "content": "Your job is to generate "+ str(num_frames) + " single line sentences which will be used in a voiceover about the topic " + str(topic)+ ". Please give numbered list only like 1. 2. 3. and so on"}
      ]
    )
    response = completion.choices[0].message.content
    # Extract sentences from the response
    sentences = re.split(r'\d+\.\s', response)[1:]  # Split by numbered list, and ignore the first empty element
    # Clean up sentences by removing any newline characters
    cleaned_sentences = [sentence.strip().replace("\n", "") for sentence in sentences]
    
    return cleaned_sentences

    
def generate_image(text):
    # Your existing code
    resp = client.images.generate(
        model="dall-e-3",
        prompt= str(text),
        n=1,
        size="1024x1024"
    )
    
    # Extract the image URL from the response
    image_url = resp.data[0].url
    
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.png"
    
    # Specify the folder to save the image
    folder = "generated_images"
    
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Download and save the image
    image_data = requests.get(image_url).content
    with open(os.path.join(folder, unique_filename), 'wb') as image_file:
        image_file.write(image_data)
    
    print(f"Image saved as {unique_filename} in the {folder} folder")


# Defining submit text function
def submit_text(generated_text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/TlLWC5O5AUzxAg7ysFZB"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": generated_text,
        "voice_settings": {
            "stability": 0.1,
            "similarity_boost": 0
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("Success:", response.content)
    else:
        print(f"Error {response.status_code}: {response.content}")


# Hitting Get Generated Items to get the history_id
# Defining get history item id function
def get_history_item_id():
    client = ElevenLabs(
    api_key=api_key,
)
    resp = client.history.get_all(
        page_size=1,
        voice_id="TlLWC5O5AUzxAg7ysFZB",

    )

    history_item_id = resp.history[0].history_item_id
    return history_item_id

# Initialize an array to store the durations
durations = []

def create_audiofile(history_item_id):
    output_folder = "generated_audio"
    client = ElevenLabs(
        api_key=api_key,
    )
    
    # Getting the audio generator
    audio_generator = client.history.get_audio(
        history_item_id=str(history_item_id),
    )
    
    # Generate a unique ID for the filename
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}.mp3"
    
    # Full path to save the file in the trimmed_audio folder
    file_path = os.path.join(output_folder, filename)
    
    # Saving the audio data to a file
    with open(file_path, "wb") as audio_file:
        for chunk in audio_generator:
            audio_file.write(chunk)

    # Load the saved audio file to get its duration
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) / 1000  # Duration in seconds
    durations.append(duration)  # Append duration to the list

    print(f"{file_path} saved successfully, duration: {duration} seconds")
    


def create_video(durations):
    audio_folder = "generated_audio"
    image_folder = "generated_images"
    output_folder = "generated_video"

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get sorted lists of audio and image files
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    clips = []

    for audio_file, image_file, duration in zip(audio_files, image_files, durations):
        # Load the audio clip
        audio_clip = AudioFileClip(os.path.join(audio_folder, audio_file))
        
        # Debug: Print audio file path and duration
        print(f"Processing audio: {audio_file}, Duration: {duration}")

        # Load the image and create a video clip with the duration of the audio
        image_clip = ImageClip(os.path.join(image_folder, image_file)).set_duration(duration)
        
        # Debug: Print image file path
        print(f"Processing image: {image_file}")

        # Combine the image and audio
        video_clip = image_clip.set_audio(audio_clip)
        clips.append(video_clip)

    # Concatenate all clips
    final_clip = concatenate_videoclips(clips)

    # Write the result to a file
    output_path = os.path.join(output_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

    print(f"Video saved as {output_path}")

    # Cleanup after video creation
    cleanup_folders([audio_folder, image_folder])

def cleanup_folders(folders):
    for folder in folders:
        if os.path.exists(folder):
            # Delete all contents of the folder
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)  # Remove file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove folder and its contents
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Cleaned up {folder} folder")


generated_array = generate_text(topic, num_frames)

for text in generated_array:
    submit_text(text)
    history_id = get_history_item_id()
    create_audiofile(history_id)
    generate_image(text)
    
# Usage
create_video(durations)

