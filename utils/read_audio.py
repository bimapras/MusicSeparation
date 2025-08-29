import subprocess
import tempfile
import os
import soundfile as sf

class AudioReader:
    def __init__(self, ffmpeg_path='ffmpeg', target_sr=44100):
        self.ffmpeg_path = ffmpeg_path
        self.target_sr = target_sr

    def read(self, filepath):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        cmd = [
            self.ffmpeg_path,
            '-y',                     # Overwrite without asking
            '-i', filepath,           # Input file
            '-ar', str(self.target_sr),  # Audio sample rate
            '-ac', '2',               # Stereo output
            '-f', 'wav',              # Output format
            tmp_wav_path
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio_np, samplerate = sf.read(tmp_wav_path, always_2d=True)

        os.remove(tmp_wav_path)

        if audio_np.shape[1] != 2:
            raise ValueError(f"Expected stereo audio (2 channels), got shape {audio_np.shape}")

        return audio_np, samplerate