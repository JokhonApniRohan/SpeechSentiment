from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch, librosa

model_name = "superb/wav2vec2-base-superb-er"  # already fine-tuned for emotion recognition

extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

speech, sr = librosa.load("Audio/surprisedtest.wav", sr=16000)
inputs = extractor(speech, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_id = torch.argmax(logits, dim=-1).item()
print("Predicted emotion:", model.config.id2label[predicted_id])
