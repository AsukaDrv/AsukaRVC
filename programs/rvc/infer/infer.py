import os
import sys
import time
import torch
import librosa
import logging
import traceback
import numpy as np
import soundfile as sf
from pydub import AudioSegment

now_dir = os.getcwd()
sys.path.append(now_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from programs.rvc.infer.pipeline import Pipeline as VC
from programs.rvc.lib.utils import load_audio_infer, load_embedding, format_title
from programs.rvc.lib.tools.split_audio import process_audio, merge_audio
from programs.rvc.lib.algorithm.synthesizers import Synthesizer
from programs.rvc.configs.config import Config

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# Process-wide VoiceConverter instance, created lazily on first use and kept
# alive so conversion caches (HuBERT embedder, generator network) are reused
# between runs. release_pipeline() unloads it without destroying the object.
_voice_converter = None


def get_voice_converter():
    """Return the process-wide VoiceConverter, creating it on first use."""
    global _voice_converter
    if _voice_converter is None:
        _voice_converter = VoiceConverter()
    return _voice_converter


def release_pipeline():
    """Unload the loaded RVC model and free GPU / RAM memory.

    Used before launching a training run, which needs all available memory.
    Returns the file name of the model that was released, or ``None`` when no
    model was loaded. The converter instance itself is kept (with its caches
    reset) so the next conversion simply reloads what it needs.
    """
    global _voice_converter
    converter = _voice_converter
    if converter is None:
        return None
    released = (
        os.path.basename(converter.current_model_path)
        if converter.current_model_path
        else None
    )
    converter.hubert_model = None
    converter.net_g = None
    converter.vc = None
    converter.cpt = None
    converter.tgt_sr = None
    converter.current_model_path = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released


class VoiceConverter:
    """
    A class for performing voice conversion using the Retrieval-Based Voice Conversion (RVC) method.
    """

    def __init__(self):
        """
        Initializes the VoiceConverter with default configuration, and sets up models and parameters.
        """
        self.config = Config()  # Load RVC configuration
        self.hubert_model = (
            None  # Initialize the Hubert model (for embedding extraction)
        )
        self.last_embedder_model = None  # Last used embedder model
        self.tgt_sr = None  # Target sampling rate for the output audio
        self.net_g = None  # Generator network for voice conversion
        self.vc = None  # Voice conversion pipeline instance
        self.cpt = None  # Checkpoint for loading model weights
        self.version = None  # Model version
        self.n_spk = None  # Number of speakers in the model
        self.use_f0 = None  # Whether the model uses F0
        self.current_model_path = None  # Path of the currently loaded model

    def load_hubert(self, embedder_model: str, embedder_model_custom: str = None):
        """
        Loads the HuBERT model for speaker embedding extraction.
        """
        self.hubert_model = load_embedding(embedder_model, embedder_model_custom)
        self.hubert_model.to(self.config.device)
        self.hubert_model = (
            self.hubert_model.half()
            if self.config.is_half
            else self.hubert_model.float()
        )
        self.hubert_model.eval()

    @staticmethod
    def convert_audio_format(input_path, output_path, output_format):
        """
        Converts the converted WAV file to the requested export format.

        Fix: the old code compared ``output_format != "WAV"`` while callers
        pass lowercase formats ("wav"), so EVERY conversion was needlessly
        resampled to the nearest "common" sample rate (e.g. 40 kHz models got
        resampled to 44.1 kHz). The comparison is now case-insensitive and a
        same-format request is a no-op.

        MP3 is exported through pydub/ffmpeg: soundfile cannot write MP3, so
        the old code silently failed (the converted file simply never appeared).
        """
        try:
            output_format = (output_format or "").strip().lower()
            input_format = os.path.splitext(input_path)[1].lstrip(".").lower()
            if not output_format or output_format == input_format:
                # Nothing to do: keep the already written file.
                return output_path

            print(f"Converting audio to {output_format} format...")
            if output_format == "mp3":
                # MP3 only supports a fixed set of sample rates; snap to the
                # nearest supported one before the ffmpeg export.
                mp3_sample_rates = [
                    8000, 11025, 12000, 16000, 22050, 24000,
                    32000, 44100, 48000,
                ]
                audio, sample_rate = librosa.load(input_path, sr=None)
                target_sr = min(mp3_sample_rates, key=lambda x: abs(x - sample_rate))
                if target_sr != sample_rate:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
                    sample_rate = target_sr
                pcm = np.clip(audio, -1.0, 1.0)
                AudioSegment(
                    (pcm * 32767).astype(np.int16).tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1,
                ).export(output_path, format="mp3")
            else:
                # Formats natively supported by soundfile (flac, ogg, ...)
                # keep their original sample rate: FLAC allows any rate, so no
                # pointless resample is needed.
                audio, sample_rate = librosa.load(input_path, sr=None)
                sf.write(output_path, audio, sample_rate, format=output_format.upper())
            return output_path
        except Exception as error:
            print(f"An error occurred converting the audio format: {error}")
            # Return the unchanged input path instead of None: the caller
            # prints/reuses this value as the conversion result.
            return input_path

    def convert_audio(
        self,
        audio_input_path: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        embedder_model: str,
        pitch: int,
        f0_file: str,
        f0_method: str,
        index_rate: float,
        volume_envelope: int,
        protect: float,
        hop_length: int,
        split_audio: bool,
        f0_autotune: bool,
        filter_radius: int,
        embedder_model_custom: str,
        export_format: str,
        resample_sr: int = 0,
        sid: int = 0,
    ):
        """
        Performs voice conversion on the input audio.
        """
        self.get_vc(model_path, sid)

        try:
            start_time = time.time()
            print(f"Converting audio '{audio_input_path}'...")
            audio = load_audio_infer(
                audio_input_path,
                16000,
            )
            audio_max = np.abs(audio).max() / 0.95

            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model

            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )

            if self.tgt_sr != resample_sr >= 16000:
                self.tgt_sr = resample_sr

            if split_audio:
                result, new_dir_path = process_audio(audio_input_path)
                if result == "Error":
                    return "Error with Split Audio", None

                dir_path = (
                    new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                )
                # Initialize paths before the conditional: the old code left
                # ``paths`` undefined when dir_path was empty, raising NameError
                # on the loop below.
                paths = []
                if dir_path:
                    paths = [
                        os.path.join(root, name)
                        for root, _, files in os.walk(dir_path, topdown=False)
                        for name in files
                        if name.endswith(".wav") and root == dir_path
                    ]
                try:
                    for path in paths:
                        self.convert_audio(
                            audio_input_path=path,
                            audio_output_path=path,
                            model_path=model_path,
                            index_path=index_path,
                            sid=sid,
                            pitch=pitch,
                            f0_file=None,
                            f0_method=f0_method,
                            index_rate=index_rate,
                            resample_sr=resample_sr,
                            volume_envelope=volume_envelope,
                            protect=protect,
                            hop_length=hop_length,
                            split_audio=False,
                            f0_autotune=f0_autotune,
                            filter_radius=filter_radius,
                            export_format=export_format,
                            embedder_model=embedder_model,
                            embedder_model_custom=embedder_model_custom,
                        )
                except Exception as error:
                    print(f"An error occurred processing the segmented audio: {error}")
                    print(traceback.format_exc())
                    return f"Error {error}"
                print("Finished processing segmented audio, now merging audio...")
                # process_audio() names the timestamps file with format_title(),
                # which replaces spaces/special characters. The reconstruction
                # below must apply the same normalization, otherwise the file
                # cannot be found for any input name containing such characters
                # (very common: song titles and voice model names).
                merge_timestamps_file = os.path.join(
                    os.path.dirname(new_dir_path),
                    f"{format_title(os.path.basename(audio_input_path).split('.')[0])}_timestamps.txt",
                )
                self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                os.remove(merge_timestamps_file)
                sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            else:
                audio_opt = self.vc.pipeline(
                    model=self.hubert_model,
                    net_g=self.net_g,
                    sid=sid,
                    audio=audio,
                    input_audio_path=audio_input_path,
                    pitch=pitch,
                    f0_method=f0_method,
                    file_index=file_index,
                    index_rate=index_rate,
                    pitch_guidance=self.use_f0,
                    filter_radius=filter_radius,
                    tgt_sr=self.tgt_sr,
                    resample_sr=resample_sr,
                    volume_envelope=volume_envelope,
                    version=self.version,
                    protect=protect,
                    hop_length=hop_length,
                    f0_autotune=f0_autotune,
                    f0_file=f0_file,
                )

            if audio_output_path:
                sf.write(audio_output_path, audio_opt, self.tgt_sr, format="WAV")
            output_path_format = audio_output_path.replace(
                ".wav", f".{export_format.lower()}"
            )
            audio_output_path = self.convert_audio_format(
                audio_output_path, output_path_format, export_format
            )

            elapsed_time = time.time() - start_time
            print(
                f"Conversion completed at '{audio_output_path}' in {elapsed_time:.2f} seconds."
            )

        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())
            # Re-raise instead of silently returning None: the caller (core.py
            # song_cover_pipeline) previously crashed later with a misleading
            # "failed to open file" error while the real failure stayed hidden.
            raise

    def convert_audio_batch(
        self,
        audio_input_paths: str,
        audio_output_path: str,
        model_path: str,
        index_path: str,
        embedder_model: str,
        pitch: int,
        f0_file: str,
        f0_method: str,
        index_rate: float,
        volume_envelope: int,
        protect: float,
        hop_length: int,
        split_audio: bool,
        f0_autotune: bool,
        filter_radius: int,
        embedder_model_custom: str,
        export_format: str,
        resample_sr: int = 0,
        sid: int = 0,
        pid_file_path: str = None,
    ):
        """
        Performs voice conversion on a batch of input audio files.
        """
        if pid_file_path:
            pid = os.getpid()
            with open(pid_file_path, "w") as pid_file:
                pid_file.write(str(pid))
        try:
            if not self.hubert_model or embedder_model != self.last_embedder_model:
                self.load_hubert(embedder_model, embedder_model_custom)
                self.last_embedder_model = embedder_model
            self.get_vc(model_path, sid)
            file_index = (
                index_path.strip()
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip()
                .replace("trained", "added")
            )
            start_time = time.time()
            print(f"Converting audio batch '{audio_input_paths}'...")
            audio_files = [
                f
                for f in os.listdir(audio_input_paths)
                if f.endswith((".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"))
            ]
            print(f"Detected {len(audio_files)} audio files for inference.")
            for i, audio_input_path in enumerate(audio_files):
                audio_output_paths = os.path.join(
                    audio_output_path,
                    f"{os.path.splitext(os.path.basename(audio_input_path))[0]}_output.{export_format.lower()}",
                )
                if os.path.exists(audio_output_paths):
                    continue
                print(f"Converting audio '{audio_input_path}'...")
                audio_input_path = os.path.join(audio_input_paths, audio_input_path)

                audio = load_audio_infer(
                    audio_input_path,
                    16000,
                )
                audio_max = np.abs(audio).max() / 0.95

                if audio_max > 1:
                    audio /= audio_max

                if self.tgt_sr != resample_sr >= 16000:
                    self.tgt_sr = resample_sr

                if split_audio:
                    result, new_dir_path = process_audio(audio_input_path)
                    if result == "Error":
                        return "Error with Split Audio", None

                    dir_path = (
                        new_dir_path.strip().strip('"').strip("\n").strip('"').strip()
                    )
                    # Same fix as in convert_audio: never leave ``paths``
                    # undefined when the chunk folder path is empty.
                    paths = []
                    if dir_path:
                        paths = [
                            os.path.join(root, name)
                            for root, _, files in os.walk(dir_path, topdown=False)
                            for name in files
                            if name.endswith(".wav") and root == dir_path
                        ]
                    try:
                        for path in paths:
                            self.convert_audio(
                                audio_input_path=path,
                                audio_output_path=path,
                                model_path=model_path,
                                index_path=index_path,
                                sid=sid,
                                pitch=pitch,
                                f0_file=None,
                                f0_method=f0_method,
                                index_rate=index_rate,
                                resample_sr=resample_sr,
                                volume_envelope=volume_envelope,
                                protect=protect,
                                hop_length=hop_length,
                                split_audio=False,
                                f0_autotune=f0_autotune,
                                filter_radius=filter_radius,
                                export_format=export_format,
                                embedder_model=embedder_model,
                                embedder_model_custom=embedder_model_custom,
                            )
                    except Exception as error:
                        print(
                            f"An error occurred processing the segmented audio: {error}"
                        )
                        print(traceback.format_exc())
                        return f"Error {error}"
                    print("Finished processing segmented audio, now merging audio...")
                    # Match the format_title() naming used by process_audio()
                    # (see the note in convert_audio).
                    merge_timestamps_file = os.path.join(
                        os.path.dirname(new_dir_path),
                        f"{format_title(os.path.basename(audio_input_path).split('.')[0])}_timestamps.txt",
                    )
                    self.tgt_sr, audio_opt = merge_audio(merge_timestamps_file)
                    os.remove(merge_timestamps_file)
                else:
                    audio_opt = self.vc.pipeline(
                        model=self.hubert_model,
                        net_g=self.net_g,
                        sid=sid,
                        audio=audio,
                        input_audio_path=audio_input_path,
                        pitch=pitch,
                        f0_method=f0_method,
                        file_index=file_index,
                        index_rate=index_rate,
                        pitch_guidance=self.use_f0,
                        filter_radius=filter_radius,
                        tgt_sr=self.tgt_sr,
                        resample_sr=resample_sr,
                        volume_envelope=volume_envelope,
                        version=self.version,
                        protect=protect,
                        hop_length=hop_length,
                        f0_autotune=f0_autotune,
                        f0_file=f0_file,
                    )

                if audio_output_paths:
                    sf.write(audio_output_paths, audio_opt, self.tgt_sr, format="WAV")
                output_path_format = audio_output_paths.replace(
                    ".wav", f".{export_format.lower()}"
                )
                audio_output_paths = self.convert_audio_format(
                    audio_output_paths, output_path_format, export_format
                )
                print(f"Conversion completed at '{audio_output_paths}'.")
            elapsed_time = time.time() - start_time
            print(f"Batch conversion completed in {elapsed_time:.2f} seconds.")
            if pid_file_path and os.path.isfile(pid_file_path):
                os.remove(pid_file_path)
        except Exception as error:
            print(f"An error occurred during audio conversion: {error}")
            print(traceback.format_exc())

    def get_vc(self, weight_root, sid):
        """
        Loads the voice conversion model and sets up the pipeline.
        """
        if sid == "" or sid == []:
            self.cleanup_model()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.load_model(weight_root)

        if self.cpt is not None:
            self.setup_network()
            self.setup_vc_instance()

    def cleanup_model(self):
        """
        Cleans up the model and releases resources.
        """
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del self.net_g, self.cpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cpt = None

    def load_model(self, weight_root):
        """
        Loads the model weights from the specified path.
        """
        # weights_only=False is explicit: PyTorch 2.6+ defaults to
        # weights_only=True, which fails on many RVC checkpoints that store
        # non-tensor pickle objects. These .pth files are local files the user
        # deliberately selected, matching the pre-2.6 behaviour.
        self.cpt = (
            torch.load(weight_root, map_location="cpu", weights_only=False)
            if os.path.isfile(weight_root)
            else None
        )
        if self.cpt is not None:
            self.current_model_path = weight_root

    def setup_network(self):
        """
        Sets up the network configuration based on the loaded checkpoint.
        """
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
            self.use_f0 = self.cpt.get("f0", 1)

            self.version = self.cpt.get("version", "v1")
            self.text_enc_hidden_dim = 768 if self.version == "v2" else 256
            self.net_g = Synthesizer(
                *self.cpt["config"],
                use_f0=self.use_f0,
                text_enc_hidden_dim=self.text_enc_hidden_dim,
                is_half=self.config.is_half,
            )
            del self.net_g.enc_q
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = (
                self.net_g.half() if self.config.is_half else self.net_g.float()
            )

    def setup_vc_instance(self):
        """
        Sets up the voice conversion pipeline instance based on the target sampling rate and configuration.
        """
        if self.cpt is not None:
            self.vc = VC(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
