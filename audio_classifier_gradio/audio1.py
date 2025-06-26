import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import tempfile
import pyttsx3
import os

# Load your trained model
model = load_model("my_model.keras")
class_names = ["it is akshen", "apshima", "it is diana"]  # Change this list as per your labels

# Create folder for output if it doesn't exist
os.makedirs("static", exist_ok=True)

def classify_audio(audio_file_path):
    # Load audio
    y, sr = librosa.load(audio_file_path, sr=16000)
    
    # Extract features (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    # Predict
    pred = model.predict(features)
    label_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    label = class_names[label_index]
    
    # Generate response text
    response_text = f"This voice is from {label} with {int(confidence * 100)} percent confidence."
    
    # Convert text to speech and save in static folder
    output_path = "static/response_audio.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(response_text, output_path)
    engine.runAndWait()

    return output_path, response_text

# Gradio Interface
interface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=[
        gr.Audio(label="Spoken Prediction Output"),
        gr.Text(label="Prediction Text")
    ],
    title="Voice Classifier",
    description="Upload or record a short audio. The system will classify the voice and speak the result.",
    live=False
)

if __name__ == "__main__":
    interface.launch()
