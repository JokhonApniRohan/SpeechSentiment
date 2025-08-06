import librosa

def preprocess_audio(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    return y, sr

y, sr = preprocess_audio('data/Actor_01/03-01-01-01-01-01-01.wav')
print(y.shape, sr)