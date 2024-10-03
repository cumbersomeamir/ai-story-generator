from flask import Flask, request, jsonify, send_file
import openai
import os
import uuid
from elevenlabs import ElevenLabs
import requests
import json
import re
from pydub import AudioSegment
from moviepy.editor import *
import shutil  # For deleting folder contents

app = Flask(__name__)

# API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# Function to generate text
def generate_text(topic, num_frames):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a super creative non-fiction story writer"},
            {"role": "user", "content": f"Your job is to generate {num_frames} single line sentences which will be used in a voiceover about the topic {topic}. Please give numbered list only like 1. 2. 3. and so on"}
        ]
    )
    response = completion.choices[0].message.content
    sentences = re.split(r'\d+\.\s', response)[1:]  # Split by numbered list, and ignore the first empty element
    cleaned_sentences = [sentence.strip().replace("\n", "") for sentence in sentences]
    
    return cleaned_sentences

# Function to generate image
def generate_image(text):
    resp = client.images.generate(
        model="dall-e-3",
        prompt=str(text),
        n=1,
        size="1024x1024"
    )
    image_url = resp.data[0].url
    unique_filename = f"{uuid.uuid4()}.png"
    folder = "generated_images"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    image_data = requests.get(image_url).content
    with open(os.path.join(folder, unique_filename), 'wb') as image_file:
        image_file.write(image_data)
    
    print(f"Image saved as {unique_filename} in the {folder} folder")

# Function to submit text to ElevenLabs
def submit_text(generated_text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/TlLWC5O5AUzxAg7ysFZB"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_api_key
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

# Function to get history item ID from ElevenLabs
def get_history_item_id():
    client = ElevenLabs(api_key=elevenlabs_api_key)
    resp = client.history.get_all(page_size=1, voice_id="TlLWC5O5AUzxAg7ysFZB")
    history_item_id = resp.history[0].history_item_id
    return history_item_id

# Function to create audio file
def create_audiofile(history_item_id, durations):
    output_folder = "generated_audio"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = ElevenLabs(api_key=elevenlabs_api_key)
    audio_generator = client.history.get_audio(history_item_id=str(history_item_id))

    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}.mp3"
    file_path = os.path.join(output_folder, filename)

    with open(file_path, "wb") as audio_file:
        for chunk in audio_generator:
            audio_file.write(chunk)

    audio = AudioSegment.from_file(file_path)
    duration = len(audio) / 1000
    durations.append(duration)
    print(f"{file_path} saved successfully, duration: {duration} seconds")

# Function to create video from images and audio
def create_video(durations):
    audio_folder = "generated_audio"
    image_folder = "generated_images"
    output_folder = "generated_video"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    clips = []

    for audio_file, image_file, duration in zip(audio_files, image_files, durations):
        audio_clip = AudioFileClip(os.path.join(audio_folder, audio_file))
        image_clip = ImageClip(os.path.join(image_folder, image_file)).set_duration(duration)
        video_clip = image_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final_clip = concatenate_videoclips(clips)
    output_path = os.path.join(output_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

    print(f"Video saved as {output_path}")
    cleanup_folders([audio_folder, image_folder])
    return output_path

# Function to clean up folders after processing
def cleanup_folders(folders):
    for folder in folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"Cleaned up {folder} folder")

# Flask endpoint for image-story
@app.route('/image-story', methods=['POST'])
def image_story():
    data = request.get_json()
    topic = data.get('topic')
    num_frames = data.get('num_frames')

    if not topic or not num_frames:
        return jsonify({"error": "Please provide 'topic' and 'num_frames' in the request body"}), 400

    try:
        num_frames = int(num_frames)
    except ValueError:
        return jsonify({"error": "'num_frames' must be an integer"}), 400

    durations = []
    generated_array = generate_text(topic, num_frames)
    
    for text in generated_array:
        submit_text(text)
        history_id = get_history_item_id()
        create_audiofile(history_id, durations)
        generate_image(text)
    
    video_path = create_video(durations)

    return send_file(video_path, as_attachment=True, mimetype='video/mp4')

# Run the Flask app on port 7003
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7003)
