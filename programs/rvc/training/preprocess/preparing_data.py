import logging
import os
import sys
import traceback
import warnings
from random import shuffle

# Environment variables setup and warning suppression
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Repository root: the script can be launched from any directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
from programs.rvc.lib.embedders.fairseq import load_model
from programs.rvc.lib.predictors.f0 import RMVPEF0
from programs.rvc.lib.utils import load_audio

exp_dir = str(sys.argv[1])  # Directory with the data prepared by `preprocess.py`
f0_method = str(sys.argv[2])  # F0 extraction method / rmvpe, rmvpe+, hpa-rmvpe
sample_rate = int(sys.argv[3])  # Sample rate used to generate filelist.txt
include_mutes = int(sys.argv[4])  # Number of mute files per speaker / Default = 2


class DataPreprocessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Operating rate of F0 and HuBERT — both models require 16 kHz
        self.sample_rate = 16000
        self.hop_size = 160
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Initialize the models
        self.model_rmvpe = RMVPEF0(self.device)
        self.hubert_model = self._load_hubert_model()
        # process_files switches to the slow path when the 16 kHz cache is missing
        self.use_16k_cache = True

    def _load_hubert_model(self):
        """Load the HuBERT embedding model."""
        hubert_model_path = os.path.join(PROJECT_ROOT, "assets", "models", "embedders", "contentvec_base.pt")
        return load_model(hubert_model_path).to(self.device).eval()

    def _read_16k(self, path):
        """Mono 16 kHz audio for F0/HuBERT: read the slicing cache directly when
        available, otherwise resample on the fly."""
        if self.use_16k_cache:
            wav, sr = sf.read(path)
            assert sr == 16000
            if wav.ndim == 2:
                wav = wav.mean(-1)
            return np.ascontiguousarray(wav, dtype=np.float32)
        return load_audio(path, self.sample_rate)

    def compute_f0(self, path, f0_method):
        """Compute the F0 contour."""
        audio = self._read_16k(path)
        return self.model_rmvpe.get_f0(audio, self.f0_min, self.f0_max, f0_method)

    def coarse_f0(self, f0):
        """Quantize the F0 contour."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
        return f0_coarse

    def extract_features(self, wav_path):
        """Extract HuBERT features."""
        audio = self._read_16k(wav_path)
        source = torch.from_numpy(audio).float().view(1, -1).to(self.device)
        padding_mask = torch.zeros(source.shape, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.hubert_model.extract_features(source=source, padding_mask=padding_mask, output_layer=12)
            return logits[0].squeeze(0).float().cpu().numpy()

    def process_files(self):
        """Main file processing method."""
        cache_root = f"{exp_dir}/data/sliced_audios_16k"
        if os.path.isdir(cache_root) and any(name.endswith(".wav") for name in os.listdir(cache_root)):
            inp_root = cache_root
        else:
            # The dataset was sliced before the 16 kHz cache existed: it works, but slowly
            inp_root = f"{exp_dir}/data/sliced_audios"
            self.use_16k_cache = False
            print("⚠ sliced_audios_16k is missing — reading with resampling. Re-run slicing for speed.")
        f0_quant_path = f"{exp_dir}/data/f0_quantized"
        f0_voiced_path = f"{exp_dir}/data/f0_voiced"
        features_path = f"{exp_dir}/data/features"

        os.makedirs(f0_quant_path, exist_ok=True)
        os.makedirs(f0_voiced_path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)

        # Collect the files to process
        files = sorted(f for f in os.listdir(inp_root) if f.endswith(".wav"))
        if not files:
            self._raise_no_files_error()

        print("\n[2/3] - Starting the feature extraction process...")

        # ncols is pinned so tqdm does not ignore bar_format on broken terminal width detection
        # Process the files
        for file in tqdm(files, desc="Extracting F0 (fundamental frequency)", bar_format="{desc}: {n}/{total}", ncols=80):
            try:
                inp_path = f"{inp_root}/{file}"
                opt_path1 = f"{f0_quant_path}/{file}"
                opt_path2 = f"{f0_voiced_path}/{file}"

                if not (os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy")):
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
            except:
                raise RuntimeError(f"F0 extraction error!\nFile - {inp_path}\n{traceback.format_exc()}")

        for file in tqdm(files, desc="Extracting HuBERT semantic features", bar_format="{desc}: {n}/{total}", ncols=80):
            try:
                wav_path = f"{inp_root}/{file}"
                out_path = f"{features_path}/{file.replace('.wav', '.npy')}"

                if not os.path.exists(out_path):
                    feats = self.extract_features(wav_path)
                    if np.isnan(feats).sum() > 0:
                        raise TypeError(f"File {file} contains invalid values (NaN).")
                    np.save(out_path, feats, allow_pickle=False)
            except:
                raise RuntimeError(f"Feature extraction error!\nFile - {wav_path}\n{traceback.format_exc()}")

        print("✓ Acoustic feature extraction finished successfully!")

    def _raise_no_files_error(self):
        error_message = (
            "ERROR: No segments to process were found.\n"
            "Possible causes:\n"
            "1. The dataset is silent.\n"
            "2. The dataset is too quiet.\n"
            "3. The dataset is too short (less than 3 seconds).\n"
            "4. The dataset is too long (more than 1 hour in a single file).\n\n"
            "Try increasing the volume or the amount of data. If you have one large file, you can split it into several smaller ones."
        )
        raise FileNotFoundError(error_message)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    mute_base_path = os.path.join(PROJECT_ROOT, "programs", "rvc", "training", "mute")

    gt_wavs_dir = os.path.join(model_path, "data", "sliced_audios")
    feature_dir = os.path.join(model_path, "data", "features")
    f0_dir = os.path.join(model_path, "data", "f0_quantized")
    f0nsf_dir = os.path.join(model_path, "data", "f0_voiced")

    gt_wavs_files = set(name.split(".")[0] for name in os.listdir(gt_wavs_dir))
    feature_files = set(name.split(".")[0] for name in os.listdir(feature_dir))
    f0_files = set(name.split(".")[0] for name in os.listdir(f0_dir))
    f0nsf_files = set(name.split(".")[0] for name in os.listdir(f0nsf_dir))

    names = gt_wavs_files & feature_files & f0_files & f0nsf_files

    options = []
    for name in names:
        options.append(
            f"{os.path.join(gt_wavs_dir, name)}.wav|"
            f"{os.path.join(feature_dir, name)}.npy|"
            f"{os.path.join(f0_dir, name)}.wav.npy|"
            f"{os.path.join(f0nsf_dir, name)}.wav.npy|0",
        )

    if include_mutes > 0:
        mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        mute_feature_path = os.path.join(mute_base_path, "features", "mute.npy")
        mute_f0_path = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        for _ in range(include_mutes):
            options.append(f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|0")

    shuffle(options)

    with open(os.path.join(model_path, "data", "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(options))


if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_files()

        generate_filelist(exp_dir, sample_rate, include_mutes)
    except Exception as e:
        print(f"Critical error: {e!s}")
        print(traceback.format_exc())
        sys.exit(1)
