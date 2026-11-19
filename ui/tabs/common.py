"""Shared helpers and constants used by every WebUI tab."""

import json
import os
import shutil
import urllib.request
import zipfile

import gradio as gr

BASE_DIR = os.getcwd()
IS_ZERO_GPU = os.getenv("SPACES_ZERO_GPU")

rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

_public_models_cache = None

AUDIO_EXTENSIONS = ['.mp3', '.m4a', '.flac', '.wav', '.aac', '.ogg',
                    '.wma', '.alac', '.aiff', '.opus', 'amr']


def get_current_models(models_dir):
    """List installed voice models, excluding base/helper files."""
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    """Refresh callback for the voice models dropdown."""
    return gr.update(choices=get_current_models(rvc_models_dir))


def load_public_models_json():
    """Load and cache public_models.json from the rvc_models directory."""
    global _public_models_cache
    if _public_models_cache is None:
        with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
            _public_models_cache = json.load(infile)
    return _public_models_cache


def load_public_models():
    """Fill the public models table with models that are not installed yet."""
    public_models = load_public_models_json()
    voice_models = get_current_models(rvc_models_dir)
    models_table = []
    for model in public_models['voice_models']:
        if model['name'] not in voice_models:
            models_table.append([model['name'], model['description'], model['credit'],
                                 model['url'], ', '.join(model['tags'])])
    tags = list(public_models['tags'].keys())
    return gr.update(value=models_table), gr.update(choices=tags)


def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def download_online_model(url, dir_name, progress=gr.Progress()):
    """Download a voice model from a HuggingFace/Pixeldrain URL or a list of file URLs."""
    try:
        progress(0, desc=f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        if "," in url:
            urls = [u.strip() for u in url.split(",") if u.strip()]
            os.makedirs(extraction_folder, exist_ok=True)
            for u in urls:
                u = u.replace("?download=true", "")
                file_name = u.split('/')[-1]
                file_path = os.path.join(extraction_folder, file_name)
                if not os.path.exists(file_path):
                    urllib.request.urlretrieve(u, file_path)
        else:
            urllib.request.urlretrieve(url, zip_name)
            progress(0.5, desc='[~] Extracting zip...')
            extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully downloaded!'
    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    """Extract an uploaded zip with a locally trained RVC model."""
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')
        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] {dir_name} Model successfully uploaded!'
    except Exception as e:
        raise gr.Error(str(e))


def filter_models(tags, query):
    """Filter the public models table by tags and/or a free text query."""
    public_models = load_public_models_json()
    models_table = []
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                attrs = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in attrs:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    else:
        for model in public_models['voice_models']:
            attrs = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in attrs:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])
    return gr.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    """Autofill the download link and model name fields from a selected table row."""
    return gr.update(value=pub_models.loc[event.index[0], 'URL']), gr.update(value=pub_models.loc[event.index[0], 'Model Name'])


def swap_visibility():
    """Toggle between the song link input and the file upload input."""
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    return gr.update(visible=(pitch_detection_algo == 'crepe'))


def show_backing_vocals_rvc_inputs(enable):
    return gr.update(visible=enable)


def get_separation_model_names():
    """Return the dropdown names of every separation model group."""
    from core import models_vocals, karaoke_models, dereverb_models, denoise_models, deecho_models
    return {
        'vocals': [m['name'] for m in models_vocals],
        'karaoke': [m['name'] for m in karaoke_models],
        'dereverb': [m['name'] for m in dereverb_models],
        'denoise': [m['name'] for m in denoise_models],
        'deecho': [m['name'] for m in deecho_models],
    }
