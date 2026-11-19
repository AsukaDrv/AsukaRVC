import os
import sys
import time
import traceback

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from programs.rvc.lib.utils import load_audio
from programs.rvc.training.preprocess.slicer import Slicer

# Parse the command line arguments
exp_dir = sys.argv[1]  # Directory to save the results in
input_root = sys.argv[2]  # Directory with the input audio files
percentage = float(sys.argv[3])  # Maximum segment length in seconds / Default = 3.0 (from n to 3 sec)
sample_rate = int(sys.argv[4])  # Sample rate the data is converted to / 32000, 40000 or 48000
normalize = sys.argv[5] == "True"  # Flag to enable/disable normalization

# Supported audio extensions
AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".aifc",
    ".mp4",
    ".mkv",
    ".webm",
}


def is_audio_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in AUDIO_EXTENSIONS


class Postfix:
    # tqdm automatically inserts ", " before the postfix — this class works
    # around that so the line reads "4/4 [segments: 922]" instead of "4/4, segments: 922"
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class PreProcess:
    def __init__(self, sample_rate, exp_dir, percentage=3.0, normalize=True):
        # Directory to store the processed audio files in
        self.gt_wavs_dir = os.path.join(exp_dir, "data", "sliced_audios")
        # 16 kHz copies of the segments: F0 and HuBERT read them directly
        # without resampling on every pass
        self.wavs16k_dir = os.path.join(exp_dir, "data", "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

        # Initialize the Slicer to slice the audio
        self.slicer = Slicer(
            sr=sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sample_rate = sample_rate  # Sample rate
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)  # High-pass filter
        self.percentage = percentage  # Segment length
        self.overlap = 0.3  # Overlap between segments
        self.tail = self.percentage + self.overlap  # Tail used for processing
        self.normalize = normalize  # Normalization flag

    def norm_write(self, tmp_audio, idx0, idx1):
        if self.normalize:
            tmp_max = np.abs(tmp_audio).max()
            if tmp_max > 0:
                tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio
                peak = np.abs(tmp_audio).max()
                if peak > 1.0:
                    tmp_audio = tmp_audio * (1.0 / peak)

        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sample_rate, tmp_audio.astype(np.float32))
        # 16 kHz twin for F0/HuBERT: resample once here instead of on every feature pass
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=16000, res_type="soxr_vhq")
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))
        return 1  # Segment written

    def pipeline_inp_dir(self, input_root):
        try:
            # Collect only audio files; everything else in the dataset folder is ignored
            names = sorted(name for name in os.listdir(input_root) if is_audio_file(os.path.join(input_root, name)))
            if not names:
                raise FileNotFoundError(
                    f"No audio files found in '{input_root}' (supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}).",
                )

            print("[1/3] - Starting the audio segmentation process...")

            total_segments = 0
            # Progress per file; the segment counter ticks in the postfix in real time.
            # ncols is pinned so tqdm does not ignore bar_format on broken terminal width detection
            with tqdm(
                total=len(names),
                desc="Segmenting audio data",
                bar_format="{desc}: {n}/{total}{postfix}",
                ncols=80,
            ) as pbar:
                last_paint = 0.0
                for idx, name in enumerate(names):
                    path = os.path.join(input_root, name)
                    try:
                        # Load the audio
                        audio = load_audio(path, self.sample_rate)
                        # Apply the high-pass filter
                        audio = signal.lfilter(self.b_high, self.a_high, audio)

                        idx1 = 0
                        # Slice the audio into segments
                        for audio in self.slicer.slice(audio):
                            i = 0
                            while True:
                                # Compute the segment start point
                                start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                                i += 1
                                # Check whether a tail of the audio is left
                                if len(audio[start:]) > self.tail * self.sample_rate:
                                    tmp_audio = audio[start : start + int(self.percentage * self.sample_rate)]
                                    total_segments += self.norm_write(tmp_audio, idx, idx1)
                                    idx1 += 1
                                    # Repaint the counter at most ~10 times per second:
                                    # faster and the terminal cannot keep up, the number after the file jumps
                                    if time.monotonic() - last_paint >= 0.1:
                                        last_paint = time.monotonic()
                                        pbar.postfix = Postfix(f" [segments: {total_segments}]")
                                        pbar.refresh()
                                else:
                                    tmp_audio = audio[start:]
                                    if len(tmp_audio) > 0:  # Do not write an empty tail
                                        total_segments += self.norm_write(tmp_audio, idx, idx1)
                                        idx1 += 1
                                    break
                    except Exception:
                        raise RuntimeError(f"{path}\t-> {traceback.format_exc()}")
                    pbar.update(1)
                    pbar.postfix = Postfix(f" [segments: {total_segments}]")
                    pbar.refresh()

            print("✓ Segmentation finished successfully!")
        except Exception:
            raise RuntimeError(f"Error! {traceback.format_exc()}")


def preprocess_trainset(input_root, sample_rate, exp_dir, percentage, normalize):
    # Initialize and run the processing
    pp = PreProcess(sample_rate, exp_dir, percentage, normalize)
    pp.pipeline_inp_dir(input_root)


if __name__ == "__main__":
    # Run the preprocessing
    preprocess_trainset(input_root, sample_rate, exp_dir, percentage, normalize)
