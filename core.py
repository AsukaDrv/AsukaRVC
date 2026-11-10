import spaces, sys
import torch
import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs
import time
import shutil
import yaml
import logging

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import noisereduce as nr
from audio_separator.separator import Separator

from programs.rvc.infer.infer import VoiceConverter
from programs.rvc.lib.tools.model_download import model_download_pipeline
from programs.music_separation_code.inference import proc_file

logging.getLogger("httpx").setLevel(logging.WARNING)

# CNFG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")
now_dir = os.getcwd()
sys.path.append(now_dir)
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

models_vocals = [
    {
        "name": "Mel-Roformer by KimberleyJSN",
        "path": os.path.join(now_dir, "models", "mel-vocals"),
        "model": os.path.join(now_dir, "models", "mel-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-vocals", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    },
    {
        "name": "BS-Roformer by ViperX",
        "path": os.path.join(now_dir, "models", "bs-vocals"),
        "model": os.path.join(now_dir, "models", "bs-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "bs-vocals", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    {
        "name": "MDX23C",
        "path": os.path.join(now_dir, "models", "mdx23c-vocals"),
        "model": os.path.join(now_dir, "models", "mdx23c-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-vocals", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
        "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
    },
]

karaoke_models = [
    {
        "name": "Mel-Roformer Karaoke by aufr33 and viperx",
        "path": os.path.join(now_dir, "models", "mel-kara"),
        "model": os.path.join(now_dir, "models", "mel-kara", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-kara", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    },
    {
        "name": "UVR-BVE",
        "full_name": "UVR-BVE-4B_SN-44100-1.pth",
        "arch": "vr",
    },
]

denoise_models = [
    {
        "name": "Mel-Roformer Denoise Normal by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise"),
        "model": os.path.join(now_dir, "models", "mel-denoise", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    {
        "name": "Mel-Roformer Denoise Aggressive by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise-aggr"),
        "model": os.path.join(now_dir, "models", "mel-denoise-aggr", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise-aggr", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel-denoise/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    },
    {
        "name": "UVR Denoise",
        "full_name": "UVR-DeNoise.pth",
        "arch": "vr",
    },
]

dereverb_models = [
    {
        "name": "MDX23C DeReverb by aufr33 and jarredou",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
    },
    {
        "name": "BS-Roformer Dereverb by anvuew",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml",
        "model_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt",
    },
    {
        "name": "UVR-Deecho-Dereverb",
        "full_name": "UVR-DeEcho-DeReverb.pth",
        "arch": "vr",
    },
    {
        "name": "MDX Reverb HQ by FoxJoy",
        "full_name": "Reverb_HQ_By_FoxJoy.onnx",
        "arch": "mdx",
    },
]

deecho_models = [
    {
        "name": "UVR-Deecho-Normal",
        "full_name": "UVR-De-Echo-Normal.pth",
        "arch": "vr",
    },
    {
        "name": "UVR-Deecho-Agggressive",
        "full_name": "UVR-De-Echo-Aggressive.pth",
        "arch": "vr",
    },
]


@lru_cache(maxsize=None)
def import_voice_converter():
    from programs.applio_code.rvc.infer.infer import VoiceConverter
    return VoiceConverter()


@lru_cache(maxsize=1)
def get_config():
    from programs.applio_code.rvc.configs.config import Config
    return Config()


def clean_old_folders(base_path: str, max_age_seconds: int = 10800):
    if not os.path.isdir(base_path):
        print(f"Error: {base_path} is not a valid directory.")
        return
    now = time.time()
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            last_modified = os.path.getmtime(folder_path)
            if now - last_modified > max_age_seconds:
                shutil.rmtree(folder_path)


def get_youtube_video_id(url, ignore_playlist=True):
    if "m.youtube.com" in url:
        url = url.replace("m.youtube.com", "www.youtube.com")
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None


def yt_download(link):
    if not link.strip():
        gr.Info("You need to provide a download link.")
        return None
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')
    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        if os.path.isdir(file):
            for ff in os.listdir(file):
                ext = os.path.splitext(ff)[1]
                if ext == '.pth':
                    rvc_model_filename = ff
                if ext == '.index':
                    rvc_index_filename = ff
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file
    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)
    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None
    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')
        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)
        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)
    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def get_audio_with_suffix(song_dir, suffix="_mysuffix.wav"):
    for file in os.listdir(song_dir):
        if file.endswith(suffix):
            return os.path.join(song_dir, file)
    return None


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)
    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def download_file(url, path, filename):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return
    try:
        torch.hub.download_url_to_file(url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")


def get_model_info_by_name(model_name):
    all_models = models_vocals + karaoke_models + dereverb_models + deecho_models + denoise_models
    for model in all_models:
        if model["name"] == model_name:
            return model
    return None


def search_with_word(folder, word):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    files = [f for f in os.listdir(folder) if word in f]
    if not files:
        return None
    return max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))


def search_with_two_words(folder, word1, word2):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    files = [f for f in os.listdir(folder) if word1 in f and word2 in f]
    if not files:
        return None
    return max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))


def get_last_modified_file(folder):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None
    return max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))


def check_fp16_support(device):
    i_device = int(str(device).split(":")[-1])
    gpu_name = torch.cuda.get_device_name(i_device)
    low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
    if any(gpu in gpu_name for gpu in low_end_gpus) and "V100" not in gpu_name.upper():
        print(f"Your GPU {gpu_name} not support FP16 inference. Using FP32 instead.")
        return False
    return True


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress, keep_orig, orig_song_path):
    song_output_dir = os.path.join(output_dir, song_id)
    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)
    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)
    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)
    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def get_audio_file(song_input, is_webui, input_type, progress):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None
    return keep_orig, orig_song_path


device = "cuda:0" if torch.cuda.is_available() else "cpu"
compute_half = True if torch.cuda.is_available() else False
config = get_config()
hubert_model = load_hubert("cuda", config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
print(device, "half>>", config.is_half)


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui, steps):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    compute_half = True if torch.cuda.is_available() else False
    config = get_config()
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
    global hubert_model
    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model, steps)
    del cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width=1.0):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'
    board = Pedalboard([
        HighpassFilter(),
        Compressor(ratio=4, threshold_db=-15),
        Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping, width=reverb_width)
    ])
    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    return output_path


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)


def merge_audios(vocals_path, inst_path, backing_path, output_path, main_gain, inst_gain, backing_vol, output_format):
    main_vocal_audio = AudioSegment.from_file(vocals_path) + main_gain
    instrumental_audio = AudioSegment.from_file(inst_path) + inst_gain
    backing_vocal_audio = AudioSegment.from_file(backing_path) + backing_vol
    combined_audio = main_vocal_audio.overlay(instrumental_audio.overlay(backing_vocal_audio))
    combined_audio.export(output_path, format=output_format)
    return output_path


def apply_noisereduce(audio_list, type_output="wav"):
    result = []
    for audio_path in audio_list:
        out_path = f"{os.path.splitext(audio_path)[0]}_nr.{type_output}"
        try:
            audio = AudioSegment.from_file(audio_path)
            samples = np.array(audio.get_array_of_samples())
            reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate, prop_decrease=0.6)
            reduced_audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
            reduced_audio.export(out_path, format=type_output)
            result.append(out_path)
        except Exception as e:
            print(f"Error noisereduce: {str(e)}")
            result.append(audio_path)
    return result


@spaces.GPU(duration=59)
def process_song(song_dir, song_input, mdx_model_params, song_id, is_webui, input_type, progress,
                 keep_files, pitch_change, pitch_change_all, voice_model, index_rate, filter_radius,
                 rms_mix_rate, protect, f0_method, crepe_hop_length, output_format, keep_orig, orig_song_path, steps):
    if not os.path.exists(song_dir):
        os.makedirs(song_dir)
        orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress, keep_orig, orig_song_path)
    else:
        vocals_path, main_vocals_path = None, None
        paths = get_audio_paths(song_dir)
        if any(path is None for path in paths):
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress, keep_orig, orig_song_path)
        else:
            orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path = paths

    pitch_change = pitch_change * 12 + pitch_change_all
    ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}_s{steps}.wav')
    ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{output_format}')

    if not os.path.exists(ai_vocals_path):
        display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
        voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui, steps)

    return ai_vocals_path, ai_cover_path, instrumentals_path, backup_vocals_path, vocals_path, main_vocals_path


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, reverb_width=1.0,
                        output_format='mp3', extra_denoise=False, steps=1,
                        vocal_model_name="Mel-Roformer by KimberleyJSN",
                        karaoke_model_name="Mel-Roformer Karaoke by aufr33 and viperx",
                        dereverb_model_name="MDX23C DeReverb by aufr33 and jarredou",
                        denoise_model_name="Mel-Roformer Denoise Normal by aufr33",
                        deecho=False, deecho_model_name="UVR-Deecho-Normal",
                        use_deecho=False, use_denoise=False, use_dereverb=True,
                        instruments_only=False, backing_vocals_volume=0,
                        inst_pitch_shift=0, export_format_final="mp3",
                        split_audio=False, autotune=False, embedder_model="contentvec",
                        batch_size=1, use_tta=False,
                        progress=gr.Progress()):
    """
    Extended song cover pipeline with more separation options.
    """
    if not keep_files or IS_ZERO_GPU:
        clean_old_folders(output_dir, 14400)
    if IS_ZERO_GPU:
        clean_old_folders(rvc_models_dir, 10800)

    try:
        if not song_input or not voice_model:
            raise_exception('Ensure that the song input field and voice model field is filled.', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        # Determine input type
        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                raise_exception('Invalid YouTube url.', is_webui)
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                song_id = None
                raise_exception(f'{song_input} does not exist.', is_webui)

        song_dir = os.path.join(output_dir, song_id)
        keep_orig, orig_song_path = get_audio_file(song_input, is_webui, input_type, progress)
        orig_song_path = convert_to_stereo(orig_song_path)

        start = time.time()

        # Use advanced separation if any advanced model differs from default
        use_advanced = (vocal_model_name != "Mel-Roformer by KimberleyJSN" or
                       karaoke_model_name != "Mel-Roformer Karaoke by aufr33 and viperx" or
                       use_denoise or use_deecho or use_dereverb or instruments_only)

        if use_advanced:
            # Run advanced pipeline using audio-separator style
            music_folder = os.path.splitext(os.path.basename(orig_song_path))[0]
            audio_files_dir = os.path.join(song_dir, "audio_files", music_folder)
            os.makedirs(audio_files_dir, exist_ok=True)

            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                devices = "0"
                print(f"Number of GPUs available: {n_gpu}")
                fp16 = check_fp16_support("cuda:0")
            else:
                devices = "cpu"
                fp16 = False

            # Vocals Separation
            display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
            model_info = get_model_info_by_name(vocal_model_name)
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(model_info["model_url"], model_info["path"], "model.ckpt")
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(model_info["config_url"], model_info["path"], "config.yaml")
            if not fp16 and "config" in model_info:
                with open(model_info["config"], "r") as file:
                    cfg = yaml.safe_load(file)
                cfg["training"]["use_amp"] = False
                with open(model_info["config"], "w") as file:
                    yaml.safe_dump(cfg, file)

            store_dir = os.path.join(audio_files_dir, "vocals")
            inst_dir = os.path.join(audio_files_dir, "instrumentals")
            os.makedirs(store_dir, exist_ok=True)
            os.makedirs(inst_dir, exist_ok=True)

            search_result = search_with_word(store_dir, "vocals")
            if not search_result:
                command = [
                    "python", os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                    "--model_type", model_info["type"],
                    "--config_path", model_info["config"],
                    "--start_check_point", model_info["model"],
                    "--input_file", orig_song_path,
                    "--store_dir", store_dir,
                    "--flac_file", "--pcm_type", "PCM_16",
                    "--extract_instrumental",
                ]
                if devices == "cpu":
                    command.append("--force_cpu")
                else:
                    device_ids = [str(int(d)) for d in devices.split()]
                    command.extend(["--device_ids"] + device_ids)
                subprocess.run(command)
                # Move instrumental to inst_dir
                inst_result = search_with_two_words(store_dir, music_folder, "instrumental")
                if inst_result:
                    os.rename(
                        os.path.join(store_dir, inst_result),
                        os.path.join(inst_dir, f"{music_folder}_instrumentals.flac")
                    )
            vocals_file = search_with_word(store_dir, "vocals")
            vocals_path = os.path.join(store_dir, vocals_file)
            inst_file_name = search_with_two_words(inst_dir, music_folder, "instrumentals")
            instrumentals_path = os.path.join(inst_dir, inst_file_name) if inst_file_name else None

            # Karaoke separation
            display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
            karaoke_dir = os.path.join(audio_files_dir, "karaoke")
            os.makedirs(karaoke_dir, exist_ok=True)
            karaoke_exists = search_with_word(karaoke_dir, "karaoke") is not None

            if not karaoke_exists:
                karaoke_info = get_model_info_by_name(karaoke_model_name)
                if karaoke_info["name"] == "Mel-Roformer Karaoke by aufr33 and viperx":
                    model_ckpt_path = os.path.join(karaoke_info["path"], "model.ckpt")
                    if not os.path.exists(model_ckpt_path):
                        download_file(karaoke_info["model_url"], karaoke_info["path"], "model.ckpt")
                    config_json_path = os.path.join(karaoke_info["path"], "config.yaml")
                    if not os.path.exists(config_json_path):
                        download_file(karaoke_info["config_url"], karaoke_info["path"], "config.yaml")
                    if not fp16 and "config" in karaoke_info:
                        with open(karaoke_info["config"], "r") as file:
                            cfg = yaml.safe_load(file)
                        cfg["training"]["use_amp"] = False
                        with open(karaoke_info["config"], "w") as file:
                            yaml.safe_dump(cfg, file)
                    command = [
                        "python", os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                        "--model_type", karaoke_info["type"],
                        "--config_path", karaoke_info["config"],
                        "--start_check_point", karaoke_info["model"],
                        "--input_file", vocals_path,
                        "--store_dir", karaoke_dir,
                        "--flac_file", "--pcm_type", "PCM_16",
                        "--extract_instrumental",
                    ]
                    if devices == "cpu":
                        command.append("--force_cpu")
                    else:
                        device_ids = [str(int(d)) for d in devices.split()]
                        command.extend(["--device_ids"] + device_ids)
                    subprocess.run(command)
                else:
                    separator = Separator(
                        model_file_dir=os.path.join(now_dir, "models", "karaoke"),
                        log_level=logging.WARNING,
                        normalization_threshold=1.0,
                        output_format="flac",
                        output_dir=karaoke_dir,
                        vr_params={"batch_size": batch_size, "enable_tta": use_tta},
                    )
                    separator.load_model(model_filename=karaoke_info["full_name"])
                    separator.separate(vocals_path)

            karaoke_file = search_with_word(karaoke_dir, "karaoke")
            if karaoke_file:
                main_vocals_path = os.path.join(karaoke_dir, karaoke_file)
            else:
                main_vocals_path = vocals_path
            backup_vocals_path = search_with_two_words(karaoke_dir, music_folder, "instrumental") or search_with_word(karaoke_dir, "instrumental")
            if backup_vocals_path:
                backup_vocals_path = os.path.join(karaoke_dir, backup_vocals_path)
            else:
                backup_vocals_path = instrumentals_path

            # Dereverb
            if use_dereverb and dereverb_model_name:
                display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
                dereverb_dir = os.path.join(audio_files_dir, "dereverb")
                os.makedirs(dereverb_dir, exist_ok=True)
                dereverb_exists = search_with_word(dereverb_dir, "noreverb") is not None

                if not dereverb_exists:
                    dereverb_info = get_model_info_by_name(dereverb_model_name)
                    if dereverb_info["name"] in ["BS-Roformer Dereverb by anvuew", "MDX23C DeReverb by aufr33 and jarredou"]:
                        model_ckpt_path = os.path.join(dereverb_info["path"], "model.ckpt")
                        if not os.path.exists(model_ckpt_path):
                            download_file(dereverb_info["model_url"], dereverb_info["path"], "model.ckpt")
                        config_json_path = os.path.join(dereverb_info["path"], "config.yaml")
                        if not os.path.exists(config_json_path):
                            download_file(dereverb_info["config_url"], dereverb_info["path"], "config.yaml")
                        if not fp16 and "config" in dereverb_info:
                            with open(dereverb_info["config"], "r") as file:
                                cfg = yaml.safe_load(file)
                            cfg["training"]["use_amp"] = False
                            with open(dereverb_info["config"], "w") as file:
                                yaml.safe_dump(cfg, file)
                        command = [
                            "python", os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                            "--model_type", dereverb_info["type"],
                            "--config_path", dereverb_info["config"],
                            "--start_check_point", dereverb_info["model"],
                            "--input_file", main_vocals_path,
                            "--store_dir", dereverb_dir,
                            "--flac_file", "--pcm_type", "PCM_16",
                        ]
                        if devices == "cpu":
                            command.append("--force_cpu")
                        else:
                            device_ids = [str(int(d)) for d in devices.split()]
                            command.extend(["--device_ids"] + device_ids)
                        subprocess.run(command)
                    else:
                        separator = Separator(
                            model_file_dir=os.path.join(now_dir, "models", "dereverb"),
                            log_level=logging.WARNING,
                            normalization_threshold=1.0,
                            output_format="flac",
                            output_dir=dereverb_dir,
                            output_single_stem="No Reverb",
                            vr_params={"batch_size": batch_size, "enable_tta": use_tta},
                        )
                        separator.load_model(model_filename=dereverb_info["full_name"])
                        separator.separate(main_vocals_path)
                dereverb_file = search_with_word(dereverb_dir, "noreverb") or search_with_two_words(dereverb_dir, music_folder, "No Reverb")
                if dereverb_file:
                    main_vocals_dereverb_path = os.path.join(dereverb_dir, dereverb_file)
                else:
                    main_vocals_dereverb_path = main_vocals_path
            else:
                main_vocals_dereverb_path = main_vocals_path

            # Deecho
            if use_deecho and deecho_model_name:
                display_progress('[~] Removing Echo from Vocals...', 0.35, is_webui, progress)
                deecho_dir = os.path.join(audio_files_dir, "deecho")
                os.makedirs(deecho_dir, exist_ok=True)
                deecho_exists = search_with_word(deecho_dir, "noecho") is not None
                if not deecho_exists:
                    deecho_info = get_model_info_by_name(deecho_model_name)
                    separator = Separator(
                        model_file_dir=os.path.join(now_dir, "models", "deecho"),
                        log_level=logging.WARNING,
                        normalization_threshold=1.0,
                        output_format="flac",
                        output_dir=deecho_dir,
                        output_single_stem="No Echo",
                        vr_params={"batch_size": batch_size, "enable_tta": use_tta},
                    )
                    separator.load_model(model_filename=deecho_info["full_name"])
                    separator.separate(main_vocals_dereverb_path)
                deecho_file = search_with_word(deecho_dir, "noecho") or search_with_two_words(deecho_dir, music_folder, "No Echo")
                if deecho_file:
                    main_vocals_dereverb_path = os.path.join(deecho_dir, deecho_file)

            # Denoise
            if use_denoise and denoise_model_name:
                display_progress('[~] Removing Noise from Vocals...', 0.4, is_webui, progress)
                denoise_dir = os.path.join(audio_files_dir, "denoise")
                os.makedirs(denoise_dir, exist_ok=True)
                denoise_exists = search_with_word(denoise_dir, "dry") is not None
                if not denoise_exists:
                    denoise_info = get_model_info_by_name(denoise_model_name)
                    if denoise_info["name"] in ["Mel-Roformer Denoise Normal by aufr33", "Mel-Roformer Denoise Aggressive by aufr33"]:
                        model_ckpt_path = os.path.join(denoise_info["path"], "model.ckpt")
                        if not os.path.exists(model_ckpt_path):
                            download_file(denoise_info["model_url"], denoise_info["path"], "model.ckpt")
                        config_json_path = os.path.join(denoise_info["path"], "config.yaml")
                        if not os.path.exists(config_json_path):
                            download_file(denoise_info["config_url"], denoise_info["path"], "config.yaml")
                        if not fp16 and "config" in denoise_info:
                            with open(denoise_info["config"], "r") as file:
                                cfg = yaml.safe_load(file)
                            cfg["training"]["use_amp"] = False
                            with open(denoise_info["config"], "w") as file:
                                yaml.safe_dump(cfg, file)
                        command = [
                            "python", os.path.join(now_dir, "programs", "music_separation_code", "inference.py"),
                            "--model_type", denoise_info["type"],
                            "--config_path", denoise_info["config"],
                            "--start_check_point", denoise_info["model"],
                            "--input_file", main_vocals_dereverb_path,
                            "--store_dir", denoise_dir,
                            "--flac_file", "--pcm_type", "PCM_16",
                        ]
                        if devices == "cpu":
                            command.append("--force_cpu")
                        else:
                            device_ids = [str(int(d)) for d in devices.split()]
                            command.extend(["--device_ids"] + device_ids)
                        subprocess.run(command)
                    else:
                        separator = Separator(
                            model_file_dir=os.path.join(now_dir, "models", "denoise"),
                            log_level=logging.WARNING,
                            normalization_threshold=1.0,
                            output_format="flac",
                            output_dir=denoise_dir,
                            output_single_stem="No Noise",
                            vr_params={"batch_size": batch_size, "enable_tta": use_tta},
                        )
                        separator.load_model(model_filename=denoise_info["full_name"])
                        separator.separate(main_vocals_dereverb_path)
                denoise_file = search_with_word(denoise_dir, "dry") or search_with_two_words(denoise_dir, music_folder, "No Noise")
                if denoise_file:
                    main_vocals_dereverb_path = os.path.join(denoise_dir, denoise_file)

            vocals_path_final = vocals_path
            backup_vocals_path_final = backup_vocals_path

        else:
            # Original MDX pipeline
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(
                song_input, mdx_model_params, song_id, is_webui, input_type, progress, keep_orig, orig_song_path
            )
            main_vocals_dereverb_path = main_vocals_dereverb_path
            vocals_path_final = vocals_path
            backup_vocals_path_final = backup_vocals_path

        # RVC Conversion
        pitch_change = pitch_change * 12 + pitch_change_all
        ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}_s{steps}.wav')
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{export_format_final}')

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui, steps)

        end = time.time()
        print(f"Execution time: {end - start:.4f} seconds")
        with sf.SoundFile(ai_vocals_path) as f:
            duration__ = len(f) / f.samplerate
        print(f"Audio duration: {duration__:.2f} seconds")

        # Apply audio effects
        display_progress('[~] Applying audio effects to Vocals...', 0.8, is_webui, progress)
        nr_path = ai_vocals_path
        if extra_denoise:
            ai_vocals_path = apply_noisereduce([ai_vocals_path])[0]

        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width)

        ins_path = get_audio_with_suffix(song_dir, "_Voiceless.wav")
        if not ins_path:
            instrumentals_path, _ = run_mdx(
                mdx_model_params,
                os.path.join(output_dir, song_id),
                os.path.join(mdxnet_models_dir, "UVR-MDX-NET-Inst_HQ_4.onnx"),
                instrumentals_path,
                exclude_inversion=True,
                suffix="Voiceless",
                denoise=False,
                keep_orig=True,
                base_device=("" if IS_ZERO_GPU else "cuda")
            )

        if pitch_change_all != 0 or inst_pitch_shift != 0:
            display_progress('[~] Applying overall pitch change', 0.85, is_webui, progress)
            if pitch_change_all != 0:
                instrumentals_path = pitch_shift(instrumentals_path, pitch_change_all)
                backup_vocals_path_final = pitch_shift(backup_vocals_path_final, pitch_change_all)
            if inst_pitch_shift != 0:
                instrumentals_path = pitch_shift(instrumentals_path, inst_pitch_shift)

        # Combine
        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
        combine_audio(
            [ai_vocals_mixed_path, backup_vocals_path_final, instrumentals_path],
            ai_cover_path, main_gain, backup_gain, inst_gain, export_format_final
        )

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path_final, main_vocals_path, ai_vocals_mixed_path, ins_path, nr_path]
            if pitch_change_all != 0:
                intermediate_files += [instrumentals_path, backup_vocals_path_final]
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return ai_cover_path

    except Exception as e:
        raise_exception(str(e), is_webui)


def download_model(link):
    model_download_pipeline(link)
    return "Model downloaded with success"


def download_music(link):
    os.makedirs(os.path.join(now_dir, "audio_files", "original_files"), exist_ok=True)
    command = [
        "yt-dlp", "-x",
        "--output", os.path.join(now_dir, "audio_files", "original_files", "%(title)s.%(ext)s"),
        link,
    ]
    subprocess.run(command)
    return "Music downloaded with success"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Asuka RVC V3.', add_help=True)
    parser.add_argument('-i', '--song-input', type=str, required=True, help='Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)')
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction, help='Whether to keep all intermediate audio files generated in the song_output/id directory, e.g. Isolated Vocals/Instrumentals')
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5, help='A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset')
    parser.add_argument('-fr', '--filter-radius', type=int, default=3, help='A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.')
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25, help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe', help='Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128, help='If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds.')
    parser.add_argument('-pro', '--protect', type=float, default=0.33, help='A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts.')
    parser.add_argument('-mv', '--main-vol', type=int, default=0, help='Volume change for AI main vocals in decibels.')
    parser.add_argument('-bv', '--backup-vol', type=int, default=0, help='Volume change for backup vocals in decibels.')
    parser.add_argument('-iv', '--inst-vol', type=int, default=0, help='Volume change for instrumentals in decibels.')
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0, help='Change the pitch/key of vocals and instrumentals.')
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15, help='Reverb room size between 0 and 1.')
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2, help='Reverb wet level between 0 and 1.')
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8, help='Reverb dry level between 0 and 1.')
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7, help='Reverb damping between 0 and 1.')
    parser.add_argument('-rwidth', '--reverb-width', type=float, default=1.0, help='Reverb width between 0 and 1.')
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3', help='Output format of audio file. mp3 for smaller file size, wav for best quality.')
    # Extended options
    parser.add_argument('-vmodel', '--vocal-model', type=str, default='Mel-Roformer by KimberleyJSN', help='Vocal separation model name.')
    parser.add_argument('-kmodel', '--karaoke-model', type=str, default='Mel-Roformer Karaoke by aufr33 and viperx', help='Karaoke separation model name.')
    parser.add_argument('-drmodel', '--dereverb-model', type=str, default='MDX23C DeReverb by aufr33 and jarredou', help='Dereverb model name.')
    parser.add_argument('-dnmodel', '--denoise-model', type=str, default='Mel-Roformer Denoise Normal by aufr33', help='Denoise model name.')
    parser.add_argument('-demodel', '--deecho-model', type=str, default='UVR-Deecho-Normal', help='Deecho model name.')
    parser.add_argument('-deecho', '--use-deecho', action='store_true', help='Enable deecho.')
    parser.add_argument('-denoise', '--use-denoise', action='store_true', help='Enable denoise.')
    parser.add_argument('-dereverb', '--use-dereverb', action='store_true', default=True, help='Enable dereverb.')
    parser.add_argument('-inst-only', '--instruments-only', action='store_true', help='Output instruments only.')
    parser.add_argument('-bvv', '--backing-vocals-volume', type=int, default=0, help='Backing vocals volume in dB.')
    parser.add_argument('-ips', '--inst-pitch-shift', type=int, default=0, help='Instrumental pitch shift in semitones.')
    parser.add_argument('-ef', '--export-format-final', type=str, default='mp3', help='Final export format.')
    parser.add_argument('-split', '--split-audio', action='store_true', help='Split audio for RVC.')
    parser.add_argument('-autotune', '--autotune', action='store_true', help='Enable autotune for RVC.')
    parser.add_argument('-emb', '--embedder-model', type=str, default='contentvec', help='Embedder model for RVC.')
    parser.add_argument('-bs', '--batch-size', type=int, default=1, help='Batch size for separation.')
    parser.add_argument('-tta', '--use-tta', action='store_true', help='Enable test-time augmentation.')
    parser.add_argument('-steps', '--steps', type=int, default=1, help='RVC inference steps.')

    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist.')

    cover_path = song_cover_pipeline(
        args.song_input, rvc_dirname, args.pitch_change, args.keep_files,
        main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
        index_rate=args.index_rate, filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
        crepe_hop_length=args.crepe_hop_length, protect=args.protect,
        pitch_change_all=args.pitch_change_all,
        reverb_rm_size=args.reverb_size, reverb_wet=args.reverb_wetness,
        reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
        reverb_width=args.reverb_width,
        output_format=args.output_format,
        vocal_model_name=args.vocal_model,
        karaoke_model_name=args.karaoke_model,
        dereverb_model_name=args.dereverb_model,
        denoise_model_name=args.denoise_model,
        deecho_model_name=args.deecho_model,
        use_deecho=args.use_deecho,
        use_denoise=args.use_denoise,
        use_dereverb=args.use_dereverb,
        instruments_only=args.instruments_only,
        backing_vocals_volume=args.backing_vocals_volume,
        inst_pitch_shift=args.inst_pitch_shift,
        export_format_final=args.export_format_final,
        split_audio=args.split_audio,
        autotune=args.autotune,
        embedder_model=args.embedder_model,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        steps=args.steps,
    )
    print(f'[+] Cover generated at {cover_path}')
