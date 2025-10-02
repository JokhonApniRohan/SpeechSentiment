import torch
import torch.nn as nn
import json
import librosa
import numpy as np
from emotion_detection import EmotionNet  # import your model class

# --- Load label mapping ---
with open("labels.json", "r") as f:
    emotion_to_idx = json.load(f)

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

# --- Reload model ---
input_dim = 40  # depends on your feature size (e.g., MFCC=40)
model = EmotionNet(input_dim=input_dim, hidden_dim1=128, hidden_dim2=128, num_classes=len(emotion_to_idx))
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

# --- Extract features from custom audio ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# --- Test with your audio file ---
file_path = "angrytest.wav"  # replace with your file
features = extract_features(file_path)
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    outputs = model(features_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_idx = predicted.item()
    predicted_emotion = idx_to_emotion[predicted_idx]

print(f"Predicted Emotion: {predicted_emotion}")
