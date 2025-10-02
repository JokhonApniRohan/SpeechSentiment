import os
import numpy as np
from audio_processing import extract_mfcc
import json

# Emotion mapping based on RAVDESS naming convention
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}



def build_dataset(data_dir="data"):
    features = []
    labels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_id = file.split("-")[2]
                emotion = EMOTION_MAP.get(emotion_id)

                # Skip unrecognized emotions
                if emotion is None:
                    continue

                mfcc_features = extract_mfcc(file_path)
                features.append(mfcc_features)
                labels.append(emotion)

    return np.array(features), np.array(labels)
