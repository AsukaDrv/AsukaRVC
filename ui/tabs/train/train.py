"""“Training” tab: sections as in the official RVC WebUI.

Dataset slicing → Features + index → Training.
Sections are launched as separate processes with a shared live log.
"""

import hashlib
import json
import os
import re
import shutil
from urllib.parse import urlsplit

import gradio as gr
import requests

from assets.env_paths import require_training_logs_dir, training_logs_dir
from assets.model_installer import PRETRAINS_DIR, download_with_progress, ensure_pretrains
from assets.notebook_check import colab_check, kaggle_check
from assets.pretrains import NO_PRETRAIN, default_pretrain, pretrain_choices
from ui.utils.jobs import command, request_stop, run_job
from ui.utils.tensorboard import DEFAULT_TENSORBOARD_PORT, launch_tensorboard

SAMPLE_RATES = [32000, 40000, 48000]
DEFAULT_VOCODER = "HiFi-GAN"
VOCODERS = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
OPTIMIZERS = ["AdamW", "AdaBelief", "PolOpt"]
# F0 methods supported by the feature-extraction backend
# (programs/rvc/lib/predictors/f0.py). The old list offered "crepe"/"fcpe",
# which the backend rejects with ValueError, so extraction always crashed
# when one of them was selected.
F0_METHODS = ["rmvpe", "rmvpe+", "hpa-rmvpe"]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".webm", ".mp4")


def _section_hint(code1: str, code2: str):
    """Centered hint under the training section header."""
    return gr.HTML(
        "<div style='font-size:13px;opacity:0.85;text-align:center;padding:0 4px 8px;'>"
        f"If the model folder contains <code>{code1}</code>, training will resume from it. "
        f"Metrics: <code>{code2}</code>.</div>",
    )


def _exp_dir(model_name: str) -> str:
    return os.path.join(training_logs_dir(), model_name.strip())


def _require_logs_dir() -> str:
    """Root of experiments for actions; in Colab without Drive — a clear error."""
    try:
        return require_training_logs_dir()
    except RuntimeError as error:
        raise gr.Error(str(error)) from error


def _require_name(model_name: str) -> str:
    if not model_name or not model_name.strip():
        raise gr.Error("Specify the model name.")
    return model_name.strip()


def _has_files(path: str) -> bool:
    try:
        return os.path.isdir(path) and bool(os.listdir(path))
    except OSError:
        return False


def _step_states(model_name):
    """Availability of buttons 1/2/3 based on the model's actual artifacts.

    Completed steps are locked against re-running; the index is available
    after feature extraction while the file `logs/<name>/<name>.index` doesn't exist.
    """
    if not model_name or not model_name.strip():
        return (
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    name = model_name.strip()
    exp_dir = _exp_dir(name)
    slices_done = _has_files(os.path.join(exp_dir, "data", "sliced_audios"))
    features_done = _has_files(os.path.join(exp_dir, "data", "features"))
    index_done = os.path.isfile(os.path.join(exp_dir, f"{name}.index"))
    return (
        gr.update(interactive=not slices_done),
        gr.update(interactive=slices_done and not features_done),
        gr.update(interactive=features_done and not index_done),
    )


def _refresh_pretrains(vocoder, sample_rate, current):
    """List of pretrains for the selected vocoder and sample rate; a foreign choice is reset.

    Sets are tied to the vocoder, so when the vocoder changes only compatible sets
    remain in the list, and for vocoders without built-in pretrains —
    a single “No pretrain” option.
    """
    choices = pretrain_choices(vocoder, sample_rate)
    value = current if current in choices else default_pretrain(vocoder, sample_rate)
    return gr.update(choices=choices, value=value)


def _open_board(port, base_url):
    """Launches TensorBoard and returns HTML for embedding."""
    logs_dir = _require_logs_dir()
    if colab_check() or kaggle_check():
        # In notebooks the browser can't see the runtime's localhost —
        # charts are opened with the %tensorboard magic in a cell.
        return (
            "<p>In Colab/Kaggle the embedded view is unavailable.</p>"
            "<p>Paste this into a separate notebook cell:</p>"
            f"<pre>%load_ext tensorboard\n%tensorboard --logdir {logs_dir}</pre>"
        )
    url = launch_tensorboard(logs_dir, int(port or DEFAULT_TENSORBOARD_PORT))
    if url.startswith("Error"):
        return f"<p style='color:#e5534b'>{url}</p>"

    # Behind a reverse proxy the address from launch() is unreachable from the browser —
    # in that case the explicitly specified base address is substituted.
    if base_url and base_url.strip():
        query = url.split("?", 1)[1] if "?" in url else ""
        url = f"{base_url.strip().rstrip('/')}/?{query}" if query else base_url.strip().rstrip("/")
    return f'<iframe src="{url}" width="100%" height="780" frameborder="0" title="TensorBoard"></iframe>'


def _input_root(model_name: str, files, folder: str) -> str:
    """Input folder for slicing: the specified folder or uploaded files."""
    if folder and folder.strip():
        if not os.path.isdir(folder.strip()):
            raise gr.Error(f"Folder not found: {folder.strip()}")
        return folder.strip()
    if files:
        dataset_dir = os.path.join(_exp_dir(model_name), "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        copied = 0
        for item in files:
            path = getattr(item, "name", None) or (item if isinstance(item, str) else None)
            if path and os.path.isfile(path) and path.lower().endswith(AUDIO_EXTENSIONS):
                shutil.copyfile(path, os.path.join(dataset_dir, os.path.basename(path)))
                copied += 1
        if not copied:
            raise gr.Error("There is no audio among the attached files.")
        return dataset_dir
    raise gr.Error("Specify a dataset folder or attach audio files.")


def _slice_dataset(model_name, files, folder, sample_rate, segment_len, normalize):
    model_name = _require_name(model_name)
    _require_logs_dir()
    input_root = _input_root(model_name, files, folder)
    yield from run_job(
        f"Dataset slicing “{model_name}”",
        command(
            "programs.rvc.training.preprocess.preprocess",
            _exp_dir(model_name),
            input_root,
            float(segment_len),
            int(sample_rate),
            "True" if normalize else "False",
        ),
    )


def _extract_features(model_name, sample_rate, f0_method, include_mutes):
    model_name = _require_name(model_name)
    _require_logs_dir()
    sliced = os.path.join(_exp_dir(model_name), "data", "sliced_audios")
    if not os.path.isdir(sliced) or not os.listdir(sliced):
        raise gr.Error("No sliced segments — run slicing first.")
    yield from run_job(
        f"Feature extraction “{model_name}”",
        command(
            "programs.rvc.training.preprocess.preparing_data",
            _exp_dir(model_name),
            f0_method,
            int(sample_rate),
            int(include_mutes),
        ),
    )


def _train_index(model_name):
    model_name = _require_name(model_name)
    _require_logs_dir()
    features = os.path.join(_exp_dir(model_name), "data", "features")
    if not os.path.isdir(features) or not os.listdir(features):
        raise gr.Error("No features — extract them first.")
    yield from run_job(
        f"Index “{model_name}”",
        command("programs.rvc.training.preprocess.extract_index", _exp_dir(model_name), "Auto"),
    )


def _check_config(model_name: str, exp_dir: str, vocoder: str, sample_rate: int) -> None:
    """Checks the vocoder and sample rate against the model's already created configuration.

    `data/config.json` is written once at the first training launch, so
    changing the vocoder or sample rate of an existing model would otherwise happen silently:
    `rvc.training.train` takes the architecture and sample rate from exactly this file.
    """
    config_path = os.path.join(exp_dir, "data", "config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise gr.Error(f"Failed to read {config_path}: {error}") from error

    saved_vocoder = config.get("model", {}).get("vocoder", DEFAULT_VOCODER)
    saved_rate = int(config.get("data", {}).get("sample_rate", sample_rate))
    if saved_vocoder == vocoder and saved_rate == sample_rate:
        return
    raise gr.Error(
        f"Model “{model_name}” is already configured for vocoder “{saved_vocoder}” and {saved_rate} Hz. "
        "Changing the vocoder and sample rate is only possible for a new model: delete "
        f"{config_path} or use a different name.",
    )


def _url_pretrain_path(kind: str, url: str) -> str:
    """Local name for a pretrain from a URL: file name + link hash.

    The hash in the name distinguishes files with the same name from different links,
    so a pretrain downloaded once is reused without re-downloading.
    """
    name = os.path.basename(urlsplit(url).path)
    if not name.lower().endswith(".pth"):
        name = f"pretrain_{kind}.pth"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    safe_stem = re.sub(r"[^\w\-.]", "_", name[: -len(".pth")])
    return os.path.join(PRETRAINS_DIR, f"custom_{safe_stem}_{digest}_{kind}.pth")


def _prepare_manual_pretrain(kind: str, value: str):
    """Custom pretrain: a local path as is, a URL — is downloaded.

    Generator: yields log lines, returns the local path to the file.
    """
    if value.startswith(("http://", "https://")):
        local_path = _url_pretrain_path(kind, value)
        if os.path.isfile(local_path):
            yield f"  ✓ {os.path.basename(local_path)} already downloaded"
            return local_path
        yield f"⬇ Downloading pretrain {kind}…"
        try:
            yield from download_with_progress(value, local_path)
        except (requests.RequestException, ValueError) as error:
            raise gr.Error(f"Failed to download pretrain {kind}: {error}") from error
        return local_path
    if not os.path.isfile(value):
        raise gr.Error(f"Pretrain not found: {value} — specify an existing file or a direct link to it.")
    return value


def _resolve_pretrains(pretrain, pretrain_g, pretrain_d, vocoder, sample_rate, exp_dir):
    """Paths of G/D pretrains: custom files or links, a built-in set, or nothing.

    Generator: yields log lines, returns (path G, path D).
    Custom files take priority over the built-in set. When resuming
    training from a checkpoint, pretrains aren't needed — downloading is skipped.
    """
    manual_g = (pretrain_g or "").strip()
    manual_d = (pretrain_d or "").strip()
    if manual_g or manual_d:
        if not (manual_g and manual_d):
            raise gr.Error("Specify both custom pretrains (G and D) or neither.")
        resolved_g = yield from _prepare_manual_pretrain("G", manual_g)
        resolved_d = yield from _prepare_manual_pretrain("D", manual_d)
        yield f"• Custom pretrains: {os.path.basename(resolved_g)}, {os.path.basename(resolved_d)}"
        return resolved_g, resolved_d
    if os.path.isfile(os.path.join(exp_dir, "checkpoint.pth")):
        yield "• Found checkpoint.pth — training will resume, pretrains not needed."
        return None, None
    if not (pretrain or "").strip() or pretrain == NO_PRETRAIN:
        # Empty pretrain (e.g. a web client submitting before the options
        # loaded) behaves exactly like the explicit "No pretrain" choice.
        yield "• No pretrain: more epochs and data will be required."
        return None, None
    sample_rate = int(sample_rate or 0)
    if sample_rate not in SAMPLE_RATES:
        raise gr.Error(
            f"Sample rate {sample_rate or '(none)'} Hz is not supported — pick one of: "
            + ", ".join(str(rate) for rate in SAMPLE_RATES)
            + "."
        )
    try:
        return (yield from ensure_pretrains(vocoder, pretrain, sample_rate))
    except ValueError as error:
        raise gr.Error(str(error)) from error


def _drain(generator):
    """Runs the generator to completion while showing the accumulated log.

    Returns (lines, return value of the generator).
    """
    lines = []
    try:
        while True:
            lines.append(next(generator))
            yield "\n".join(lines)
    except StopIteration as done:
        return lines, done.value


def _train_model(
    model_name,
    sample_rate,
    total_epoch,
    save_every_epoch,
    batch_size,
    vocoder,
    optimizer,
    gpus,
    pretrain,
    pretrain_g,
    pretrain_d,
    save_to_zip,
    save_half,
):
    model_name = _require_name(model_name)
    logs_dir = _require_logs_dir()
    exp_dir = _exp_dir(model_name)
    if not os.path.isfile(os.path.join(exp_dir, "data", "filelist.txt")):
        raise gr.Error("No filelist.txt — run slicing and feature extraction first.")
    _check_config(model_name, exp_dir, vocoder, int(sample_rate))
    cmd = command(
        "programs.rvc.training.train",
        "--experiment_dir",
        logs_dir,
        "--model_name",
        model_name,
        "--total_epoch",
        int(total_epoch),
        "--save_every_epoch",
        int(save_every_epoch),
        "--batch_size",
        int(batch_size),
        "--sample_rate",
        int(sample_rate),
        "--vocoder",
        vocoder,
        "--optimizer",
        optimizer,
        "--gpus",
        (gpus or "0").strip(),
        "--save_to_zip",
        "true" if save_to_zip else "false",
        "--save_half",
        "true" if save_half else "false",
    )

    notes, (resolved_g, resolved_d) = yield from _drain(
        _resolve_pretrains(pretrain, pretrain_g, pretrain_d, vocoder, int(sample_rate), exp_dir),
    )
    if resolved_g and resolved_d:
        cmd += ["--pretrain_g", resolved_g, "--pretrain_d", resolved_d]
    yield from run_job(f"Training “{model_name}”", cmd, prefix="\n".join(notes))


def training_tab():
    model_name = gr.Textbox(label="Model name", placeholder="MyVoice")

    with gr.Accordion("Data preparation", open=False), gr.Group(), gr.Row(equal_height=True):
        with gr.Column(scale=1):
            dataset_folder = gr.Textbox(label="Dataset folder", placeholder="/path/to/audio")
            dataset_files = gr.File(label="…or upload files", file_count="multiple", height=260)
        with gr.Column(scale=1):
            with gr.Row(equal_height=True):
                sample_rate = gr.Dropdown(SAMPLE_RATES, value=48000, label="Sample rate (Hz)")
                f0_method = gr.Dropdown(F0_METHODS, value="rmvpe", label="F0 method")
                normalize = gr.Checkbox(value=True, label="Normalization")
            with gr.Row(equal_height=True):
                segment_len = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=3.0, label="Segment (sec)")
                include_mutes = gr.Slider(minimum=0, maximum=10, step=1, value=2, label="Mute files")
            with gr.Row(equal_height=True):
                slice_btn = gr.Button("1. Slice", variant="primary")
                extract_btn = gr.Button("2. Extract", variant="primary", interactive=False)
                index_btn = gr.Button("3. Build index", interactive=False)

    with gr.Group():
        _section_hint("checkpoint.pth", "tensorboard --logdir logs")
        with gr.Row(equal_height=True):
            total_epoch = gr.Slider(minimum=1, maximum=10000, step=1, value=300, label="Total epochs")
            save_every_epoch = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Save every N")
            batch_size = gr.Slider(minimum=1, maximum=128, step=1, value=8, label="Batch")
        with gr.Accordion("Advanced", open=False):
            with gr.Row(equal_height=True):
                vocoder = gr.Dropdown(VOCODERS, value=DEFAULT_VOCODER, label="Vocoder")
                optimizer = gr.Dropdown(OPTIMIZERS, value="AdamW", label="Optimizer")
                gpus = gr.Textbox("0", label="GPU")
            with gr.Row(equal_height=True):
                pretrain = gr.Dropdown(
                    pretrain_choices(DEFAULT_VOCODER, 48000),
                    value=default_pretrain(DEFAULT_VOCODER, 48000),
                    label="Pretrain",
                    scale=1,
                )
                pretrain_g = gr.Textbox(label="Custom pretrain G", placeholder="Path or URL to .pth — instead of the built-in one", scale=2)
                pretrain_d = gr.Textbox(label="Custom pretrain D", placeholder="Path or URL to .pth — instead of the built-in one", scale=2)
            with gr.Row(equal_height=True):
                save_to_zip = gr.Checkbox(False, label="Build ZIP at the end")
                save_half = gr.Checkbox(True, label="float16 weights")
        with gr.Row(equal_height=True):
            train_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop process", variant="stop")
        stop_status = gr.Markdown("")

    with gr.Group():
        log = gr.Textbox(label="Log", lines=12, max_lines=12)
        board_btn = gr.Button("📊 TensorBoard", variant="secondary")
        with gr.Accordion("TensorBoard parameters", open=False), gr.Row(equal_height=True):
            board_port = gr.Number(value=DEFAULT_TENSORBOARD_PORT, label="Port", precision=0)
            board_base_url = gr.Textbox(
                label="Base address (behind a proxy)",
                placeholder="http://127.0.0.1:6006/tensorboard",
                info="Leave empty when running on your own computer.",
            )
        board_frame = gr.HTML("")

    step_buttons = [slice_btn, extract_btn, index_btn]
    model_name.change(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    # The pretrain list depends on both the vocoder and the sample rate.
    for trigger in (vocoder, sample_rate):
        trigger.change(_refresh_pretrains, inputs=[vocoder, sample_rate, pretrain], outputs=pretrain, api_name=False)
    slice_btn.click(
        _slice_dataset,
        inputs=[model_name, dataset_files, dataset_folder, sample_rate, segment_len, normalize],
        outputs=log,
    ).then(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    extract_btn.click(
        _extract_features,
        inputs=[model_name, sample_rate, f0_method, include_mutes],
        outputs=log,
    ).then(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    train_btn.click(
        _train_model,
        inputs=[
            model_name,
            sample_rate,
            total_epoch,
            save_every_epoch,
            batch_size,
            vocoder,
            optimizer,
            gpus,
            pretrain,
            pretrain_g,
            pretrain_d,
            save_to_zip,
            save_half,
        ],
        outputs=log,
    )
    index_btn.click(_train_index, inputs=model_name, outputs=log).then(
        _step_states,
        inputs=model_name,
        outputs=step_buttons,
        api_name=False,
    )
    board_btn.click(_open_board, inputs=[board_port, board_base_url], outputs=board_frame)
    stop_btn.click(request_stop, outputs=stop_status, queue=False, api_name=False)
