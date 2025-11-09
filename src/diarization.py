import os
from dotenv import load_dotenv
load_dotenv()
from pyannote.audio import Pipeline

# Replace this with your Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Load the pretrained diarization pipeline from Hugging Face
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Path to your audio file (wav, 16 kHz mono preferred)
audio_file = "D:\Projects\Speech Sentiment\Audio\conversation.wav"

# Run diarization
diarization_result = pipeline(audio_file)
# print(diarization_result)


# Print diarization results: who spoke when
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    print(f"Speaker {speaker} speaks from {turn.start:.1f}s to {turn.end:.1f}s")

##hello