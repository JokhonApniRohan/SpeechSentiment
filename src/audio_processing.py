import librosa
import numpy as np

def preprocess_audio(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    return y, sr

def extract_mfcc(path, n_mfcc=40):
    """Extract MFCC features from an audio file."""
    y, sr = preprocess_audio(path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean