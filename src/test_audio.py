import torch
import torch.nn as nn
import librosa
import numpy as np
from emotion_detection import EmotionNetMultiTask  # your multi-task model

# --- Emotion mapping ---
EMOTION_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}

# --- Reload model ---
input_dim = 40  # must match training
num_emotions = len(EMOTION_MAP)

model = EmotionNetMultiTask(input_dim=input_dim, hidden_dim1=128, hidden_dim2=128, num_emotions=num_emotions)
model.load_state_dict(torch.load("emotion_model_multitask.pth", map_location=torch.device('cpu')))
model.eval()

# --- Feature extraction ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# --- Test single audio file ---
file_path = "Record (online-voice-recorder.com).wav"  # replace with your audio
features = extract_features(file_path)
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    emotion_out, intensity_out, gender_out = model(features_tensor)

    # Emotion prediction
    _, pred_emotion = torch.max(emotion_out, 1)
    predicted_emotion = EMOTION_MAP[pred_emotion.item()]

    # Intensity prediction
    _, pred_intensity = torch.max(intensity_out, 1)
    predicted_intensity = "normal" if pred_intensity.item() == 0 else "strong"

    # Gender prediction
    _, pred_gender = torch.max(gender_out, 1)
    predicted_gender = "male" if pred_gender.item() == 0 else "female"

print(f"Predicted Emotion: {predicted_emotion}")
print(f"Predicted Intensity: {predicted_intensity}")
print(f"Predicted Gender: {predicted_gender}")
