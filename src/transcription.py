# transcription_module.py
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diarization import perform_diarization

def transcribe_diarized_audio(audio_path: str):
    """
    Transcribes the given audio file into dialogue format based on diarization.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        list[dict]: List of dictionaries with speaker and transcription text.
    """
    # Step 1: Perform diarization
    diarized_segments = perform_diarization(audio_path)

    # Step 2: Load Whisper processor and model (small model for demo)
    print("Loading Whisper model (this may take a while the first time)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

    # Step 3: Load the audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Whisper expects 16kHz audio, so resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    dialogue_output = []

    for seg in diarized_segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        # Convert to numpy and then to float32 as expected by processor
        segment_array = segment_waveform.squeeze().numpy()

        # Prepare inputs for Whisper
        inputs = processor(segment_array, sampling_rate=sample_rate, return_tensors="pt").to(device)

        # Generate token ids
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features, max_length=448)

        # Decode tokens to text
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()

        dialogue_output.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": transcription
        })

        print(f"{seg['speaker']} ({seg['start']}s - {seg['end']}s): {transcription}")

    return dialogue_output


def save_transcription(dialogue_output, output_path="transcription.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in dialogue_output:
            f.write(f"{seg['speaker']}: {seg['text']}\n")
    print(f"\n✅ Transcription saved to {output_path}")


# Test run
if __name__ == "__main__":
    audio_path = r"D:\Projects\Speech Sentiment\Audio\conversation.wav"
    transcription = transcribe_diarized_audio(audio_path)
    save_transcription(transcription, "dialogue_transcription.txt")
