import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
enhancer = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank")


enhanced= enhancer.enhance_file("Audio/angrytest.wav")
waveform = enhanced.squeeze(0)
if enhanced.ndim == 1:
    enhanced = enhanced.unsqueeze(0)
fs = 16000

torchaudio.save("Audio/clean_angrytest.wav", enhanced, fs)
