"""REST API for the custom HTML/CSS/JS WebUI (served from ``ui/webui``).

The router is mounted under ``/api/v1`` on the same Gradio-managed FastAPI
server that hosts the classic Gradio Blocks UI, so everything (custom UI,
REST API, audio files) is reachable through the local URL and through the
public ``*.gradio.live`` share URL.

Design notes
------------
* Heavy dependencies (torch via ``programs.core``) are imported lazily inside
  handlers/worker threads so the server can start without the full ML stack.
* Pipeline functions are called with ``is_webui=0`` so they raise plain
  exceptions (``gr.Error`` instances are still caught by the job manager).
* File downloads are restricted to the song output and upload directories.
"""

import mimetypes
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config.logging_setup import get_logger
from programs.webui.jobs import BusyError, manager
from ui.tabs.common import (
    AUDIO_EXTENSIONS,
    ensure_rvc_models_dir,
    extract_zip,
    get_current_models,
    load_public_models_json,
    output_dir,
    rvc_models_dir,
)

logger = get_logger("webui_api")

router = APIRouter(prefix="/api/v1")

# Repository root (programs/webui/api.py -> programs -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
WEBUI_DIR = REPO_ROOT / "ui" / "webui"

# Directory where audio uploaded through the web UI is stored.
uploads_dir = os.path.join(os.getcwd(), "assets", "audios", "uploads")

# Only files inside these roots can be served back to the browser.
_ALLOWED_FILE_ROOTS = [
    os.path.realpath(output_dir),
    os.path.realpath(uploads_dir),
]

IS_ZERO_GPU = bool(os.getenv("SPACES_ZERO_GPU"))


def _backend_ready() -> bool:
    """True when the ML stack (torch) is importable in this environment."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _read_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "upload")
    return re.sub(r"[^\w\-.()\[\] ]", "_", name).strip() or "upload"


def _result_path_to_url(path: str, download: bool = False) -> str:
    from urllib.parse import quote

    url = f"/api/v1/files?path={quote(path)}"
    if download:
        url += "&download=1"
    return url


# --------------------------------------------------------------------- health


@router.get("/health")
def health():
    job = manager.active_public()
    return {
        "status": "ok",
        "app": "AsukaRVC",
        "version": _read_version(),
        "zero_gpu": IS_ZERO_GPU,
        "backend_ready": _backend_ready(),
        "active_job": job,
    }


@router.get("/options")
def options():
    """Dropdown choices + defaults for the frontend forms."""
    data = {
        "voice_models": _list_voice_models(),
        "f0_methods": ["rmvpe", "fcpe", "crepe"],
        "output_formats": ["mp3", "wav"],
        "final_formats": ["mp3", "wav", "flac"],
        "embedders": ["contentvec", "chinese-hubert-base", "japanese-hubert-base", "korean-hubert-base"],
        "train": {
            "sample_rates": [32000, 40000, 48000],
            "vocoders": ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
            "optimizers": ["AdamW", "AdaBelief", "PolOpt"],
            "f0_methods": ["rmvpe", "rmvpe+", "hpa-rmvpe"],
        },
        "separation_ready": True,
    }
    try:
        # Lazy import: pulls programs.core (torch, audio_separator, ...).
        from ui.tabs.common import get_separation_model_names

        data["separation"] = get_separation_model_names()
    except Exception as error:  # noqa: BLE001 — degrade gracefully without the ML stack
        data["separation"] = {
            "vocals": [], "karaoke": [], "dereverb": [], "denoise": [], "deecho": [],
        }
        data["separation_ready"] = False
        data["separation_error"] = f"{type(error).__name__}: {error}"
    return data


# --------------------------------------------------------------------- models


def _list_voice_models() -> list[dict]:
    ensure_rvc_models_dir()
    models = []
    for name in get_current_models(rvc_models_dir):
        model_dir = os.path.join(rvc_models_dir, name)
        if not os.path.isdir(model_dir):
            continue
        pth, indices = None, []
        for root, _dirs, files in os.walk(model_dir):
            for file_name in files:
                rel = os.path.relpath(os.path.join(root, file_name), model_dir)
                if file_name.endswith(".pth"):
                    pth = rel
                elif file_name.endswith(".index"):
                    indices.append(rel)
        models.append({"name": name, "pth": pth, "indices": indices})
    return models


@router.get("/models")
def list_models():
    return {"models": _list_voice_models()}


@router.get("/public-models")
def public_models(query: str = "", tags: str = ""):
    """Public models table with the same filtering as the Gradio downloader tab."""
    data = load_public_models_json()
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    text_query = (query or "").strip().lower()
    models = []
    for model in data["voice_models"]:
        if tag_list and not all(tag in model["tags"] for tag in tag_list):
            continue
        if text_query:
            attrs = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if text_query not in attrs:
                continue
        models.append(
            {
                "name": model["name"],
                "description": model["description"],
                "credit": model["credit"],
                "url": model["url"],
                "tags": model["tags"],
            }
        )
    return {"tags": list(data["tags"].keys()), "models": models}


class ModelDownloadRequest(BaseModel):
    url: str
    name: str


def _download_voice_model(url: str, dir_name: str) -> str:
    """Voice-model download with the same semantics as the Gradio downloader
    tab (HuggingFace / Pixeldrain / multi-file links), without Gradio objects."""
    url = (url or "").strip()
    dir_name = (dir_name or "").strip()
    if not url:
        raise ValueError("Download link is required.")
    if not dir_name:
        raise ValueError("Model name is required.")

    zip_name = url.split("/")[-1]
    extraction_folder = os.path.join(rvc_models_dir, dir_name)
    if os.path.exists(extraction_folder):
        raise ValueError(
            f"Voice model directory {dir_name} already exists! "
            "Choose a different name for your voice model."
        )

    if "pixeldrain.com" in url:
        url = f"https://pixeldrain.com/api/file/{zip_name}"

    logger.info("[~] Downloading voice model '%s'...", dir_name)
    if "," in url:
        urls = [u.strip() for u in url.split(",") if u.strip()]
        os.makedirs(extraction_folder, exist_ok=True)
        for single_url in urls:
            single_url = single_url.replace("?download=true", "")
            file_name = single_url.split("/")[-1]
            file_path = os.path.join(extraction_folder, file_name)
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(single_url, file_path)
    else:
        zip_path = os.path.join(os.getcwd(), zip_name)
        urllib.request.urlretrieve(url, zip_path)
        logger.info("[~] Extracting zip...")
        extract_zip(extraction_folder, zip_path)
    logger.info("[+] %s model successfully downloaded!", dir_name)
    return f"[+] {dir_name} model successfully downloaded!"


@router.post("/jobs/model-download")
def start_model_download(req: ModelDownloadRequest):
    try:
        job = manager.start_callable(
            "model-download",
            f"Download model '{req.name}'",
            _download_voice_model,
            req.url,
            req.name,
        )
    except BusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"job_id": job.id}


@router.post("/jobs/model-upload")
async def start_model_upload(file: UploadFile = File(...), name: str = Form(...)):
    dir_name = (name or "").strip()
    if not dir_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="The model archive must be a .zip file.")

    os.makedirs(uploads_dir, exist_ok=True)
    zip_path = os.path.join(uploads_dir, f"model_{int(time.time())}_{_sanitize_filename(file.filename)}")
    with open(zip_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    extraction_folder = os.path.join(rvc_models_dir, dir_name)
    if os.path.exists(extraction_folder):
        os.remove(zip_path)
        raise HTTPException(status_code=400, detail=f"Voice model directory {dir_name} already exists!")

    try:
        job = manager.start_callable(
            "model-upload",
            f"Upload model '{dir_name}'",
            extract_zip,
            extraction_folder,
            zip_path,
        )
    except BusyError as error:
        os.remove(zip_path)
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"job_id": job.id}


# --------------------------------------------------------------------- upload


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Store an audio file and return its server path (used as pipeline input)."""
    original_name = file.filename or "audio"
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type '{extension}'. Allowed: {', '.join(AUDIO_EXTENSIONS)}",
        )
    os.makedirs(uploads_dir, exist_ok=True)
    dest = os.path.join(uploads_dir, f"{int(time.time() * 1000)}_{_sanitize_filename(original_name)}")
    with open(dest, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    return {"path": dest, "name": os.path.basename(dest), "url": _result_path_to_url(dest)}


# ------------------------------------------------------------------ inference


class InferenceRequest(BaseModel):
    """AI cover pipeline parameters — defaults mirror the Gradio Inference tab."""

    song_input: str  # YouTube/URL link or local audio path
    voice_model: str
    pitch_change: int = 0          # octaves, AI vocals only
    pitch_change_all: int = 0      # semitones, whole mix
    keep_files: bool = True
    main_gain: int = 0
    backup_gain: int = 0
    inst_gain: int = 0
    backing_vocals_volume: int = 0
    index_rate: float = 0.5
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    f0_method: str = "rmvpe"
    crepe_hop_length: int = 128
    protect: float = 0.33
    reverb_rm_size: float = 0.15
    reverb_wet: float = 0.2
    reverb_dry: float = 0.8
    reverb_damping: float = 0.7
    reverb_width: float = 1.0
    output_format: str = "mp3"
    extra_denoise: bool = True
    vocal_model_name: str = "Mel-Roformer by KimberleyJSN"
    karaoke_model_name: str = "Mel-Roformer Karaoke by aufr33 and viperx"
    dereverb_model_name: str = "MDX23C DeReverb by aufr33 and jarredou"
    denoise_model_name: str = "Mel-Roformer Denoise Normal by aufr33"
    deecho_model_name: str = "UVR-Deecho-Normal"
    use_deecho: bool = False
    use_denoise: bool = False
    use_dereverb: bool = True
    instruments_only: bool = False
    inst_pitch_shift: int = 0
    export_format_final: str = "mp3"
    split_audio: bool = False
    autotune: bool = False
    embedder_model: str = "contentvec"
    infer_backing_vocals: bool = False
    infer_backing_vocals_model: str | None = None
    infer_backing_vocals_index: str = ""
    pitch_back: int = 0
    filter_radius_back: int = 3
    index_rate_back: float = 0.5
    rms_mix_rate_back: float = 0.25
    protect_back: float = 0.33
    pitch_extract_back: str = "rmvpe+"
    hop_length_back: int = 128
    export_format_rvc_back: str = "wav"
    split_audio_back: bool = False
    autotune_back: bool = False
    embedder_model_back: str = "contentvec"
    hf_upload: bool = False
    hf_repo_id: str = ""
    hf_token: str = ""
    batch_size: int = 1
    use_tta: bool = False


@router.post("/jobs/inference")
def start_inference(req: InferenceRequest):
    if not req.song_input.strip():
        raise HTTPException(status_code=400, detail="Song input is required (URL or audio file).")
    if not req.voice_model.strip():
        raise HTTPException(status_code=400, detail="Select a voice model first.")
    known_models = {model["name"] for model in _list_voice_models()}
    if req.voice_model not in known_models:
        raise HTTPException(status_code=400, detail=f"Voice model '{req.voice_model}' is not installed.")

    if IS_ZERO_GPU:
        req.keep_files = False  # ZeroGPU sessions keep no intermediate files (same as the Gradio UI)

    def _run_pipeline():
        # Lazy import: pulls programs.core (torch, audio_separator, ...).
        from programs.core import song_cover_pipeline

        return song_cover_pipeline(
            song_input=req.song_input.strip(),
            voice_model=req.voice_model,
            pitch_change=req.pitch_change,
            keep_files=req.keep_files,
            is_webui=0,  # plain exceptions + logger-based progress for the job log
            main_gain=req.main_gain,
            backup_gain=req.backup_gain,
            inst_gain=req.inst_gain,
            index_rate=req.index_rate,
            filter_radius=req.filter_radius,
            rms_mix_rate=req.rms_mix_rate,
            f0_method=req.f0_method,
            crepe_hop_length=req.crepe_hop_length,
            protect=req.protect,
            pitch_change_all=req.pitch_change_all,
            reverb_rm_size=req.reverb_rm_size,
            reverb_wet=req.reverb_wet,
            reverb_dry=req.reverb_dry,
            reverb_damping=req.reverb_damping,
            reverb_width=req.reverb_width,
            output_format=req.output_format,
            extra_denoise=req.extra_denoise,
            vocal_model_name=req.vocal_model_name,
            karaoke_model_name=req.karaoke_model_name,
            dereverb_model_name=req.dereverb_model_name,
            denoise_model_name=req.denoise_model_name,
            deecho_model_name=req.deecho_model_name,
            use_deecho=req.use_deecho,
            use_denoise=req.use_denoise,
            use_dereverb=req.use_dereverb,
            instruments_only=req.instruments_only,
            backing_vocals_volume=req.backing_vocals_volume,
            inst_pitch_shift=req.inst_pitch_shift,
            export_format_final=req.export_format_final,
            split_audio=req.split_audio,
            autotune=req.autotune,
            embedder_model=req.embedder_model,
            infer_backing_vocals=req.infer_backing_vocals,
            infer_backing_vocals_model=req.infer_backing_vocals_model or None,
            infer_backing_vocals_index=req.infer_backing_vocals_index,
            pitch_back=req.pitch_back,
            filter_radius_back=req.filter_radius_back,
            index_rate_back=req.index_rate_back,
            rms_mix_rate_back=req.rms_mix_rate_back,
            protect_back=req.protect_back,
            pitch_extract_back=req.pitch_extract_back,
            hop_length_back=req.hop_length_back,
            export_format_rvc_back=req.export_format_rvc_back,
            split_audio_back=req.split_audio_back,
            autotune_back=req.autotune_back,
            embedder_model_back=req.embedder_model_back,
            hf_upload=req.hf_upload,
            hf_repo_id=req.hf_repo_id,
            hf_token=req.hf_token,
            batch_size=req.batch_size,
            use_tta=req.use_tta,
        )

    try:
        job = manager.start_callable("inference", f"AI cover — {req.voice_model}", _run_pipeline)
    except BusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"job_id": job.id}


# ------------------------------------------------------------------- training


class TrainSliceRequest(BaseModel):
    model_name: str
    dataset_folder: str = ""
    sample_rate: int = 48000
    segment_len: float = 3.0
    normalize: bool = True


class TrainExtractRequest(BaseModel):
    model_name: str
    sample_rate: int = 48000
    f0_method: str = "rmvpe"
    include_mutes: int = 2


class TrainStartRequest(BaseModel):
    model_name: str
    sample_rate: int = 48000
    total_epoch: int = 300
    save_every_epoch: int = 25
    batch_size: int = 8
    vocoder: str = "HiFi-GAN"
    optimizer: str = "AdamW"
    gpus: str = "0"
    pretrain: str = ""
    pretrain_g: str = ""
    pretrain_d: str = ""
    save_to_zip: bool = False
    save_half: bool = True


class TrainIndexRequest(BaseModel):
    model_name: str


def _train_module():
    """The Gradio Training tab builders: reused so the web UI runs the exact
    same validated pipeline commands (same subprocess launcher, same
    pretrain/config checks). Raises ``gr.Error`` on invalid input, which the
    job manager converts into a job error message."""
    from ui.tabs.train import train as train_module

    return train_module


def _train_state(model_name: str) -> dict:
    """Artifact availability for the three data-preparation steps."""
    train_module = _train_module()
    name = (model_name or "").strip()
    if not name:
        return {"name": "", "sliced": False, "features": False, "index": False, "filelist": False}
    exp_dir = train_module._exp_dir(name)
    data_dir = os.path.join(exp_dir, "data")

    def _has_files(path: str) -> bool:
        try:
            return os.path.isdir(path) and bool(os.listdir(path))
        except OSError:
            return False

    return {
        "name": name,
        "sliced": _has_files(os.path.join(data_dir, "sliced_audios")),
        "features": _has_files(os.path.join(data_dir, "features")),
        "index": os.path.isfile(os.path.join(exp_dir, f"{name}.index")),
        "filelist": os.path.isfile(os.path.join(data_dir, "filelist.txt")),
    }


@router.get("/train/status")
def train_status(model: str = Query("", alias="model")):
    return _train_state(model)


@router.get("/train/pretrains")
def train_pretrains(vocoder: str = "HiFi-GAN", sample_rate: int = 48000):
    from assets.pretrains import default_pretrain, pretrain_choices

    return {
        "choices": pretrain_choices(vocoder, sample_rate),
        "default": default_pretrain(vocoder, sample_rate),
    }


def _drain_into_log(generator, job) -> None:
    """Consume a ``ui.utils.jobs.run_job``-style generator, streaming every
    accumulated log snapshot into the job so the browser can poll it."""
    for text in generator:
        manager.set_log(job, text)


@router.post("/jobs/train-slice")
def train_slice(req: TrainSliceRequest):
    train_module = _train_module()
    folder = (req.dataset_folder or "").strip()
    _validate_train_sample_rate(req.sample_rate)

    def _run(job):
        files = None
        if not folder:
            # Fall back to files uploaded earlier into the experiment dataset dir.
            dataset_dir = os.path.join(train_module._exp_dir(req.model_name), "dataset")
            if os.path.isdir(dataset_dir):
                files = [
                    os.path.join(dataset_dir, name)
                    for name in sorted(os.listdir(dataset_dir))
                    if name.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".webm", ".mp4"))
                ]
        _drain_into_log(
            train_module._slice_dataset(
                req.model_name, files, folder or None, req.sample_rate, req.segment_len, req.normalize
            ),
            job,
        )

    return {"job_id": _start_train_job("train-slice", f"Dataset slicing '{req.model_name}'", _run)}


@router.post("/jobs/train-extract")
def train_extract(req: TrainExtractRequest):
    train_module = _train_module()
    _validate_train_sample_rate(req.sample_rate)

    def _run(job):
        _drain_into_log(
            train_module._extract_features(req.model_name, req.sample_rate, req.f0_method, req.include_mutes),
            job,
        )

    return {"job_id": _start_train_job("train-extract", f"Feature extraction '{req.model_name}'", _run)}


@router.post("/jobs/train-index")
def train_index(req: TrainIndexRequest):
    train_module = _train_module()

    def _run(job):
        _drain_into_log(train_module._train_index(req.model_name), job)

    return {"job_id": _start_train_job("train-index", f"Index '{req.model_name}'", _run)}


def _validate_train_sample_rate(sample_rate) -> int:
    """Reject invalid sample rates with a clear message instead of the
    confusing "Pretrain set '' has no files for 0 Hz" error."""
    rate = int(sample_rate or 0)
    if rate not in (32000, 40000, 48000):
        raise HTTPException(
            status_code=400,
            detail="Sample rate must be one of: 32000, 40000, 48000 "
            f"(got {rate or 'none'}). The page may still be loading — try again in a moment.",
        )
    return rate


@router.post("/jobs/train-start")
def train_start(req: TrainStartRequest):
    train_module = _train_module()
    req.sample_rate = _validate_train_sample_rate(req.sample_rate)
    if not (req.pretrain or "").strip():
        # A web client that submits before the dropdowns loaded sends an empty
        # pretrain — treat it like the Gradio tab's "No pretrain" choice.
        from assets.pretrains import NO_PRETRAIN

        req.pretrain = NO_PRETRAIN

    def _run(job):
        _drain_into_log(
            train_module._train_model(
                req.model_name,
                req.sample_rate,
                req.total_epoch,
                req.save_every_epoch,
                req.batch_size,
                req.vocoder,
                req.optimizer,
                req.gpus,
                req.pretrain,
                req.pretrain_g,
                req.pretrain_d,
                req.save_to_zip,
                req.save_half,
            ),
            job,
        )

    return {"job_id": _start_train_job("train-start", f"Training '{req.model_name}'", _run)}


def _start_train_job(kind: str, title: str, runner) -> str:
    """Register a training job; the runner receives the Job instance and
    streams the reused Gradio-tab generators into the job log for polling."""
    try:
        job = manager.start_callable(kind, title, runner, pass_job=True)
    except BusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return job.id


@router.post("/train/dataset-upload")
async def train_dataset_upload(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
):
    """Save dataset audio files into the experiment dataset directory."""
    train_module = _train_module()
    model_name = (name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    dataset_dir = os.path.join(train_module._exp_dir(model_name), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    saved = 0
    for upload in files:
        file_name = _sanitize_filename(upload.filename or "")
        if not file_name.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".webm", ".mp4")):
            continue
        with open(os.path.join(dataset_dir, file_name), "wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        saved += 1
    if not saved:
        raise HTTPException(status_code=400, detail="There is no audio among the attached files.")
    return {"saved": saved, "dataset_dir": dataset_dir, "state": _train_state(model_name)}


@router.post("/jobs/stop")
def stop_job():
    return {"message": manager.stop()}


# ----------------------------------------------------------------------- jobs


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    payload = job.public()
    if payload["status"] == "finished" and payload.get("result", {}).get("output"):
        output = payload["result"]["output"]
        if isinstance(output, str) and os.path.isfile(output):
            payload["result"]["url"] = _result_path_to_url(output)
            payload["result"]["download_url"] = _result_path_to_url(output, download=True)
    return payload


# ---------------------------------------------------------------------- files


@router.get("/files")
def get_file(path: str = Query(...), download: bool = False):
    """Serve pipeline outputs / uploaded files; paths outside the allowed
    roots are rejected to prevent arbitrary file reads."""
    resolved = os.path.realpath(path)
    if not any(
        resolved == root or resolved.startswith(root + os.sep) for root in _ALLOWED_FILE_ROOTS
    ):
        raise HTTPException(status_code=403, detail="Path is outside the allowed directories.")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
    disposition = "attachment" if download else "inline"
    return FileResponse(
        resolved,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{os.path.basename(resolved)}"'},
    )
