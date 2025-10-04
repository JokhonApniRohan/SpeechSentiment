import os
import matplotlib.pyplot as plt
from collections import Counter

# Map emotion numbers to labels (RAVDESS official mapping)
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_emotion(filename):
    # Example filename: 03-01-05-01-02-02-12.wav
    parts = filename.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]  # 3rd block is emotion
    return emotion_map.get(emotion_code, None)

def visualize_ravdess(data_folder="data"):
    emotion_counts = Counter()

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                emotion = parse_emotion(file)
                if emotion:
                    emotion_counts[emotion] += 1

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color="skyblue")
    plt.title("RAVDESS Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    visualize_ravdess("data")  # change if your dataset path differs
