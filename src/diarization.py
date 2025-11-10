# diarization_module.py
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Load environment variables (for HF token)
load_dotenv()

def perform_diarization(audio_file: str):
    """
    Performs speaker diarization on the given audio file.

    Args:
        audio_file (str): Path to the input audio file.

    Returns:
        list[dict]: List of dictionaries containing speaker label,
                    start time, and end time for each speech segment.
    """
    # Get Hugging Face token
    HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

    # Load the pretrained diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=HUGGINGFACE_TOKEN
    )

    # Run diarization
    diarization_result = pipeline(audio_file)

    # Collect results
    diarized_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segment_info = {
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        }
        diarized_segments.append(segment_info)

    return diarized_segments


# Optional: Run directly for testing
if __name__ == "__main__":
    test_audio = r"D:\Projects\Speech Sentiment\Audio\conversation.wav"
    results = perform_diarization(test_audio)
    for seg in results:
        print(f"Speaker {seg['speaker']} speaks from {seg['start']}s to {seg['end']}s")
