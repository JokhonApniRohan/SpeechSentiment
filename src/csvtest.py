import opensmile
import pandas as pd
import joblib

# === 1️⃣ Load the trained emotion model ===
model = joblib.load("emotion_model.pkl")  # change if your filename differs

# === 2️⃣ Load training dataset columns to know which features were used ===
training_df = pd.read_csv("final_audio_feature_dataset.csv", nrows=1)
# remove target columns and file column
feature_columns = [c for c in training_df.columns if c not in ['file', 'emotion', 'intensity', 'gender']]

# === 3️⃣ Initialize OpenSMILE ===
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# === 4️⃣ Extract features from a new audio file ===
def extract_features(audio_path):
    features = smile.process_file(audio_path)
    features.reset_index(drop=True, inplace=True)

    # Keep only columns that were used in training
    missing_cols = [col for col in feature_columns if col not in features.columns]
    available_cols = [col for col in feature_columns if col in features.columns]

    # Only keep the features that exist in both
    filtered_features = features[available_cols].copy()

    # Add missing columns with 0 values to maintain the same shape
    for col in missing_cols:
        filtered_features[col] = 0

    # Reorder columns to match training order
    filtered_features = filtered_features[feature_columns]

    return filtered_features

# === 5️⃣ Predict emotion for new audio ===
def predict_emotion(audio_path):
    print(f"🔍 Extracting features from: {audio_path}")
    X_new = extract_features(audio_path)

    # Predict
    prediction = model.predict(X_new)[0]
    print(f"🎯 Predicted Emotion: {prediction}")

    return prediction

# === 6️⃣ Example usage ===
if __name__ == "__main__":
    test_audio_path = "happytest.wav"  # ⬅️ Replace with your file
    predict_emotion(test_audio_path)
