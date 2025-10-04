# dataset_builder.py
import os
import numpy as np
from audio_processing import extract_mfcc

# Emotion mapping (RAVDESS)
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

# Intensity mapping (RAVDESS: 01 = normal, 02 = strong)
INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}


def parse_ravdess_filename(filename):
    """
    RAVDESS filename format:
    '03-01-05-02-02-01-12.wav'
      |  |  |  |  |  |  |
      |  |  |  |  |  |  └── Actor ID (odd = male, even = female)
      |  |  |  |  |  └──── Repetition
      |  |  |  |  └────── Statement
      |  |  |  └──────── Intensity (01 = normal, 02 = strong)
      |  |  └────────── Emotion
      |  └──────────── Vocal Channel
      └─────────────── Modality (speech/song)
    """
    parts = filename.split(".")[0].split("-")
    emotion_id = parts[2]
    intensity_id = parts[3]
    actor_id = int(parts[-1])

    emotion = EMOTION_MAP.get(emotion_id, "unknown")
    intensity = INTENSITY_MAP.get(intensity_id, "unknown")
    gender = "male" if actor_id % 2 != 0 else "female"

    return emotion, intensity, gender


def build_dataset(data_dir="data"):
    features = []
    emotion_labels = []
    intensity_labels = []
    gender_labels = []

    # Create mapping for intensity and gender
    intensity_to_idx = {"normal": 0, "strong": 1}
    gender_to_idx = {"male": 0, "female": 1}

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)
            emotion, intensity, gender = parse_ravdess_filename(file)

            if emotion == "unknown":
                continue

            mfcc_features = extract_mfcc(file_path)

            features.append(mfcc_features)
            emotion_labels.append(emotion)
            intensity_labels.append(intensity_to_idx[intensity])
            gender_labels.append(gender_to_idx[gender])

    # Convert emotion strings to integers
    emotions = sorted(set(emotion_labels))
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}
    emotion_labels_idx = [emotion_to_idx[e] for e in emotion_labels]

    return (
        np.array(features, dtype=np.float32),
        np.array(emotion_labels_idx, dtype=np.int64),
        np.array(intensity_labels, dtype=np.int64),
        np.array(gender_labels, dtype=np.int64),

    )
