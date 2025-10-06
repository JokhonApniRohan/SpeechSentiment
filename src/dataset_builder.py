# dataset_builder.py
import os
import numpy as np
import pandas as pd
from audio_processing import extract_egemaps  # must return a DataFrame (eGeMAPS features)

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

INTENSITY_MAP = {"01": "normal", "02": "strong"}

def parse_ravdess_filename(filename):
    parts = filename.split(".")[0].split("-")
    emotion_id = parts[2]
    intensity_id = parts[3]
    actor_id = int(parts[-1])
    emotion = emotion_id
    intensity = INTENSITY_MAP.get(intensity_id, "unknown")
    gender = "male" if actor_id % 2 != 0 else "female"
    return emotion, intensity, gender


def build_dataset(data_dir="data", output_csv="ravdess_features.csv"):
    all_data = []

    intensity_to_idx = {"normal": 0, "strong": 1}
    gender_to_idx = {"male": 0, "female": 1}

    file_count = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)
            emotion, intensity, gender = parse_ravdess_filename(file)

            if emotion == "unknown":
                continue

            # Extract features for this file
            features = extract_egemaps(file_path)
            if features is None or features.empty:
                print(f"⚠️ Skipped (no features): {file}")
                continue

            feature_dict = features.iloc[0].to_dict()
            feature_dict["file"] = file
            feature_dict["emotion"] = emotion
            feature_dict["intensity"] = intensity_to_idx[intensity]
            feature_dict["gender"] = gender_to_idx[gender]

            all_data.append(feature_dict)
            file_count += 1
            if file_count % 10 == 0:
                print(f"✅ Processed {file_count} files...")

    if not all_data:
        print("❌ No features extracted! Check your data path or feature extraction.")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved as '{output_csv}' with shape {df.shape}")

    # Prepare NumPy arrays
    feature_cols = [col for col in df.columns if col not in ["file", "emotion", "intensity", "gender"]]
    features = df[feature_cols].to_numpy(dtype=np.float32)

    emotions = sorted(df["emotion"].unique())
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}
    emotion_labels = df["emotion"].map(emotion_to_idx).to_numpy(dtype=np.int64)

    intensity_labels = df["intensity"].to_numpy(dtype=np.int64)
    gender_labels = df["gender"].to_numpy(dtype=np.int64)

    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Example emotions: {emotion_to_idx}")

    return features, emotion_labels, intensity_labels, gender_labels, emotion_to_idx


if __name__ == "__main__":
    features, emotion_labels, intensity_labels, gender_labels, emotion_to_idx = build_dataset(
        data_dir="data", 
        output_csv="ravdess_features.csv"
    )
