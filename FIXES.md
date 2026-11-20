# AsukaRVC — Bug Fixes and Code Revisions

This document lists every bug fixed and revision made in this working copy.
All code and comments are written in English. Verification: `39/39` unit checks
plus an end-to-end split/merge smoke test pass.

## New feature: two UI modes (`--ui gradio | web`)

The app now ships two interchangeable frontends on one code base:

1. **`--ui gradio` (default)** — the classic Gradio Blocks UI (`ui/`), unchanged.
2. **`--ui web`** — a fully custom HTML/CSS/JS frontend hosted by the
   Gradio-managed server, so `--share` still exposes everything through one
   public `*.gradio.live` URL.

Web-mode wiring (verified against the Gradio 6.x server internals and with a
live curl/browser smoke test):

- `ui/app.py` gains the `--ui` flag and dispatches to
  `programs/webui/server.py::launch_webui()`.
- The server is bootstrapped with `Blocks.launch(prevent_thread_lock=True)`
  (share tunnel included), then the custom API router and static files are
  attached to the same running FastAPI app: `/webui/` serves
  `ui/webui/` (vanilla HTML/CSS/JS, no build step), `/api/v1/…` serves the
  REST API, and a redirect route inserted at position 0 sends `/` to the
  custom UI. The Gradio tunnel proxies every path, so the custom UI and API
  work through the public share URL too.
- `programs/webui/jobs.py` runs long tasks (inference, model
  download/upload, training steps) in background worker threads — one active
  job at a time (HTTP 409 otherwise) — with live log streaming, elapsed-time
  status and stop support (subprocess jobs die via `ui.utils.jobs`; in-process
  inference is flagged and marked stopped on completion).
- `programs/webui/api.py` reuses the exact Gradio-tab logic: inference calls
  `song_cover_pipeline` (with `is_webui=0` so plain exceptions reach the job
  log), training endpoints consume the same `ui.tabs.train.train` generators
  (`_slice_dataset`, `_extract_features`, `_train_index`, `_train_model`) so
  the web UI runs the identical validated commands; heavy imports stay lazy
  so the server starts without the ML stack (endpoints degrade gracefully and
  `/health` reports `backend_ready`).
- File serving (`GET /api/v1/files`) is restricted to the song output and
  upload directories (path traversal rejected with 403).
- Frontend (`ui/webui/`): inference form mirroring every Gradio option,
  model browser (installed / public index with tag filters / URL & zip
  install), full training section with step-state chips, job banner with
  log drawer, toasts. No frameworks or CDNs required.

Bugs fixed along the way in this round:

- `ui/tabs/upload/upload_tab.py` was missing from the repository while
  `tabs_main.py` still imported it — `ui/app.py` crashed at startup with
  `ModuleNotFoundError`. The tab is restored (zip upload -> `rvc_models`),
  with a path normalizer so it works on Gradio >= 4 (plain path string)
  and older versions (`.name` attribute).
- `programs/core.py` imported the Hugging Face `spaces` package at module
  level although it is never used anywhere in the code base — every plain
  local install without that package crashed with `ModuleNotFoundError`
  before the UI could even start. The dead import is removed.
- `ui/app.py` now anchors the process to the repository root at startup
  (pipeline modules resolve `rvc_models` / `song_output` / `logs` from the
  CWD) and imports the Hugging Face `spaces` helper defensively, so plain
  local installs without that package can still run.
- `ui/app.py` handles the Gradio 6 API change that moved the Blocks `theme`
  from the constructor into `launch()`: the theme is passed to the right
  place based on the installed Gradio major version, so the same code runs
  on Gradio 4.x / 5.x / 6.x without warnings.
- `ui/tabs/inference/inference_tab.py` imported `programs.core` (torch and
  the whole separation stack) at module level; the import is now lazy inside
  the builder, matching the strategy already used by `ui/utils/jobs.py`.

## Critical bugs

### 1. `programs/rvc/lib/tools/model_download.py` — module crashed the whole app at import time
- The storage root was located with `find_folder_parent(os.getcwd(), "logs")`,
  which walked the whole tree looking for ANY folder named `logs`.
  - On a fresh clone no `logs` folder exists -> returned `None` ->
    `os.path.join(None, "zips")` -> `TypeError` while importing
    `programs/core.py` -> the WebUI never started.
  - On clones that do contain `.git/`, the lookup matched `.git/logs` and
    returned `.git` as the storage root, writing downloads into `.git/zips`.
  - Fix: paths are now anchored to the repository root (resolved from
    `__file__`, five directory levels up) and created on demand.
- Downloaded models were extracted into `logs/<name>` — the *training*
  experiments root — where the Inference tab never saw them (the app lists
  models from `rvc_models/`). Fix: models are now installed into
  `rvc_models/<name>`.
- All `os.chdir()` calls were removed; every operation now uses absolute
  paths. `os.chdir()` mutates the working directory of the entire process
  (all Gradio worker threads) and several branches leaked the changed CWD.
- Hugging Face `/tree/main` links were resolved but never downloaded
  (`wget.download()` sat inside the wrong `if` branch). Fix: the normalized
  URL is always fetched.
- The `/blob/` + `/resolve/` branch crashed with `KeyError` when the server
  sent no `Content-Disposition` header, and with `AttributeError` when the
  header had no `filename=` entry. Fix: `_filename_from_headers()` helper with
  URL-basename fallback.

### 2. `programs/rvc/infer/infer.py` — every conversion was needlessly resampled
- `convert_audio_format()` compared `output_format != "WAV"` (uppercase) while
  callers pass lowercase `"wav"`, so the branch always ran and resampled the
  audio to the nearest "common" sample rate (e.g. 40 kHz models became
  44.1 kHz). Fix: case-insensitive comparison; same-format requests are a no-op.
- MP3 export used `soundfile`, which cannot write MP3 — the export silently
  failed and the backing-vocals `.mp3` never appeared. Fix: MP3 is exported
  through pydub/ffmpeg (snapping to the nearest MP3-supported sample rate).
- On conversion errors the function returned `None`; now it returns the
  unchanged input path.
- `convert_audio()` / `convert_audio_batch()` swallowed all exceptions and
  returned `None`, so the cover pipeline crashed later with a misleading
  "failed to open file" error while the real failure stayed hidden.
  Fix: re-raise after logging.
- `paths` was left undefined when the split-audio chunk folder was empty
  -> `NameError` on the conversion loop. Fix: initialised to `[]`.
- The merge-timestamps path was reconstructed from the raw file name, but
  `process_audio()` stores it under `format_title()` (spaces/special chars
  replaced). Any song or voice-model name containing spaces therefore crashed
  the Split Audio feature with `FileNotFoundError`. Fix: the reconstruction
  applies `format_title()` too.
- `convert_audio_batch()` crashed on `open(None)` when `pid_file_path` was not
  provided. Fix: guarded.
- `load_model()` now passes `weights_only=False` to `torch.load` explicitly
  (see item 6).

### 3. `programs/core.py` — stem lookup and pipeline fixes
- `get_youtube_video_id()`: for `/watch/<id>` URLs the split index returned the
  literal string `watch` instead of the video id (`split('/')[1]` ->
  `split('/')[2]`).
- `search_with_word()` / `search_with_two_words()` were case-sensitive exact
  substring searches. `audio-separator` writes stems as
  `song_(No Echo)_UVR-De-Echo-Normal.flac` / `(Karaoke)` / `(Instrumental)`,
  so the converted UVR karaoke / deecho / denoise / dereverb results were
  NEVER found and silently ignored. Fix: `_normalize_filename()` makes the
  lookups case-insensitive and separator-tolerant (verified against the
  actual `audio-separator` 0.18 naming convention).
- Intermediate-file cleanup removed pitch-shifted files only when
  `pitch_change_all != 0`; instrumental-only shifts (`inst_pitch_shift`)
  leaked the shifted copy on disk. Fix: covered by an `elif`.

### 4. `programs/rvc/infer/pipeline.py` — F0 predictor caching
- A fresh `RMVPE0Predictor` was constructed on every `get_f0()` call, re-reading
  the checkpoint and rebuilding the network for every audio segment (very
  costly in Split Audio mode). Fix: the predictor is created once and cached on
  the pipeline instance.
- Style clean-up: `pitch != None` -> `pitch is not None`.

### 5. `ui/tabs/train/train.py` — F0 method list did not match the backend
- The Training tab offered `crepe` / `fcpe`, but the feature-extraction backend
  (`programs/rvc/lib/predictors/f0.py`) only supports `rmvpe`, `rmvpe+` and
  `hpa-rmvpe` — selecting the other two always crashed step 2 with
  `ValueError`. Fix: the dropdown now offers exactly the supported methods.

### 6. PyTorch 2.6+ compatibility (`torch.load` / `weights_only`)
- `programs/rvc/infer/infer.py` (`load_model`),
  `programs/rvc/lib/predictors/RMVPE.py`, `programs/rvc/lib/predictors/FCPE.py`:
  `torch.load` now passes `weights_only=False` explicitly. PyTorch 2.6 changed
  the default to `weights_only=True`, which rejects many RVC / predictor
  checkpoints (requirements.txt has no upper torch bound, so users hit this on
  install). These are local files the user deliberately selected; behaviour
  matches pre-2.6 PyTorch. `train.py` already had a load fallback and is
  unchanged.

### 7. `programs/music_separation_code/utils.py` — MSST separation fixes
- `windowingArray` was mutated in place (`window = windowingArray; window[:fade_size] = 1`),
  so the fade-in/fade-out tweaks persisted into every later batch and TTA pass,
  corrupting the crossfades. Fix: operate on a copy.
- Replaced the deprecated `torch.cuda.amp.autocast` with
  `torch.amp.autocast("cuda", ...)`, additionally disabling AMP when CUDA is
  unavailable (requesting CUDA autocast on a CPU-only machine raises
  `RuntimeError`).

## Minor revisions

- `programs/rvc/training/mel_processing.py`: replaced the mutable list default
  argument (`n_mels=[...]`) with a tuple.
- `programs/rvc/training/train.py`: guarded the metrics block when an epoch has
  zero batches (`last_batch is None` -> skip instead of crash).
- `ui/tabs/inference/inference_tab.py`: the Clear button reset the backing
  pitch algorithm to `rmvpe` while the UI default is `rmvpe+`; aligned.
- `programs/rvc/lib/tools/model_download.py`: removed the unused `six` import
  and the redundant `sys.path.append`; `programs/rvc/infer/infer.py`: removed
  the unused `scipy.io.wavfile` import.

## Verification

- `python -m py_compile` passes on every Python file in the repository.
- 39 unit checks covering the fixed logic (YouTube id parsing, normalized stem
  search, timestamps naming, module import paths, header parsing, F0 method
  lists, format-conversion semantics) all pass.
- End-to-end smoke test of `process_audio()` -> `merge_audio()` with a file
  name containing spaces and parentheses passes (this exact case used to crash
  the Split Audio feature).
- Live smoke test, `--ui gradio` mode: the Blocks app with all five tabs
  (Inference, Download model, Upload model, Train) assembles and serves
  `HTTP 200` (tested with the ML stack stubbed, since the code only needs the
  real libraries when a conversion actually runs).
- Live end-to-end test, `--ui web` mode: server boots without the ML stack,
  `POST /api/v1/jobs/model-download` installs a synthetic model zip through a
  background job (log streamed, status polled to `finished`), the new model
  appears in `GET /api/v1/models`, unknown models are rejected with `400`,
  and `GET /api/v1/files` rejects paths outside the allowed roots with `403`.
