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

- Training via the web UI could crash with the confusing
  `Pretrain set '' has no files for 0 Hz` error when a job was submitted
  before the page finished loading its options (empty selects sent
  `pretrain=""` / `sample_rate=0`). Fixed on all layers: the frontend train
  buttons now refuse to submit with an unloaded sample rate, the REST API
  rejects invalid rates with a readable HTTP 400 and maps an empty pretrain
  to "No pretrain", and the shared training logic treats an empty pretrain
  exactly like the explicit "No pretrain" choice.
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

## New features: duet covers (weights.gg style) and Google Drive save in Colab

### Duet cover — two AI voices on one cover

A new "Duet Cover" option lets a second RVC voice model sing part of the song,
the way weights.gg duets worked. The song is converted once per voice model,
then both takes are woven into one duet vocal track that replaces the main
vocals downstream (effects, backing mix, final export); the output is named
`<song> (<Model A> x <Model B> Duet).<ext>`.

- `programs/duet_cover.py` (new, torch-free): `DUET_MODES` and
  `build_duet_vocals()` with three styles — `alternate` (voices swap every N
  seconds), `sequential` (voice A sings the first X %, voice B the rest) and
  `overlap` (both voices sing together). Every segment boundary gets a short
  equal-power crossfade (80 ms, clamped for very short segments) so the voice
  switch does not click; takes are trimmed to the shorter length because RVC
  conversion can shift duration by a few samples. Kept free of torch so the
  web API validates duet requests before the ML stack is loaded.
- `programs/core.py`: new `duet_enabled / duet_model / duet_mode / duet_split /
  duet_interval` parameters on `song_cover_pipeline`. The second voice is
  converted with the same RVC options and cached per-model (a repeated run
  with different duet settings reuses the two conversions). The merged duet
  track is cached by mode+split/interval and cleaned up as an intermediate
  file. Validation: the second model must be set, installed and different
  from the main model; modes are restricted to `DUET_MODES`. The CLI gained
  matching flags (`-duet`, `-duet-model`, `-duet-mode`, `-duet-split`,
  `-duet-interval`).
- Gradio UI (`ui/tabs/inference/inference_tab.py`): "Duet Cover — two AI
  voices" accordion with the enable checkbox, second-model dropdown, style
  dropdown and split/interval sliders; wired into Generate and Clear.
- Web UI: `InferenceRequest` gained the duet fields with full validation
  (unknown model, same-model duet, bogus mode, out-of-range split/interval ->
  readable HTTP 400), the job title shows both voices, `/options` advertises
  `duet_modes`, and the frontend (`ui/webui/index.html` + `app.js`) has a duet
  accordion whose second-model select stays in sync with the model list.

### Google Drive save — Colab only

Generated covers live on the ephemeral Colab VM disk; the new Drive feature
copies them into the user's Google Drive so they survive the runtime.

- `programs/colab_drive.py` (new): `in_colab()` detection (`google.colab`
  import, `/content` fallback), `ensure_drive_mounted()` (auto-mounts on
  first save) and `save_file_to_drive()` which copies a result into
  `<mount>/MyDrive/<folder>/` with ` (1)`, ` (2)` … collision suffixes and
  traversal/absolute-folder validation.
- `programs/core.py`: `save_drive` / `drive_folder` parameters on
  `song_cover_pipeline` save the final cover right after the mix; failures
  are logged (and surfaced as a Gradio warning) without losing the local
  cover. CLI flags: `-drive`, `-drive-folder`.
- Gradio UI: a "Google Drive (Colab)" accordion appears only when the app
  runs inside Colab (`in_colab()`), with an auto-save checkbox and a folder
  field.
- Web UI: `/options` advertises `colab` + `drive_folder`, `InferenceRequest`
  accepts `save_drive` / `drive_folder` (rejected with HTTP 400 outside
  Colab), and a new `POST /api/v1/jobs/save-drive` endpoint copies an already
  generated cover as a background job. For security it only accepts files
  inside the song output directory (HTTP 403 otherwise). The frontend shows a
  "Save to Google Drive" button next to the result and the auto-save accordion
  only when the server reports a Colab runtime.

## Web UI: task log moved below the page content

The custom WebUI rendered the task log as a `position: fixed` right-side drawer
that slid over the forms (it covered the settings cards on narrow screens and
colabs). It is now a regular panel in the document flow, placed after the tab
content and before the footer:

- `ui/webui/index.html`: the `#log-drawer` block moved below `</main>`.
- `ui/webui/assets/styles.css`: the drawer lost its fixed positioning and z-index;
  it is a centred 1012px terminal panel (same width as the job banner and the
  cards) with a 360px scrolling log area. The Train-tab log card gets the same
  360px bound so long training logs scroll inside the card instead of stretching
  the page.
- `ui/webui/assets/app.js`: opening the log via the "Log" button in the job
  banner scrolls the panel into view; jobs that auto-open the log on start no
  longer shift the page the user is looking at (nothing floats above the UI at
  any point).

## Duet rework: phrase-aligned voice switching (researched)

The first duet implementation swapped voices on wall-clock marks (every N
seconds / at a raw percent of the duration), which cuts sung lines mid-word.
Real duet covers from the weights.gg era switch voices per vocal phrase.
weights.gg itself shut down on 31 March 2026 and its domain is excluded from
the Wayback Machine, so the mechanism was verified against the surviving
platform of that era — Jammable, which documents its duet feature openly in
its official guide. The rules that matter:

* the vocals are converted once per voice model and woven together (our
  pipeline already matched this);
* "auto pick" detects where the singing changes and assigns each detected
  phrase to the next voice in turn — never fixed time windows;
* parts shorter than ~2.5 seconds are merged into a neighbouring part;
* each voice can sit at its own pitch (a semitone offset per voice).

What changed in this repo:

- `programs/duet_cover.py`: new `detect_vocal_phrases()` — a numpy RMS
  envelope (50 ms hops) with hysteresis thresholds (-38 / -45 dBFS), short-gap
  merging (a <350 ms breath does not split a phrase) and a minimum part
  length (2.5 s, the same rule the web duet editors use); the returned parts
  cover the whole take (gaps are split evenly between neighbours) so no
  audio is dropped. New default mode `auto` assigns the detected phrases to
  the voices in turn (A, B, A, …); when a take has no usable phrase
  structure (continuous singing) it falls back to timed swaps so both voices
  still get a turn. `alternate` and `sequential` keep their sliders but every
  swap now snaps to the nearest phrase boundary (alternate: max 5 s drift),
  so voice changes land in the pauses between lines instead of mid-word.
  `overlap` is unchanged, and the crossfade can no longer exceed a quarter of
  the shortest segment.
- `programs/core.py`: new `duet_pitch` parameter — a semitone offset applied
  only to the second voice's conversion (-12..+12; e.g. +12 to put a female
  model over a male original), carried in the cache filename so takes are
  never reused across offsets. `duet_mode` defaults to `auto`; CLI gained
  `-duet-pitch` and the `-duet-mode` help reflects the new styles.
- REST API: `duet_mode` defaults to `auto` and a new `duet_pitch` field is
  validated (HTTP 400 outside -12..12).
- Both UIs: `auto` is the default duet style and a second-voice pitch slider
  was added (Gradio slider + web slider), with updated hints.
- Tests: 15 new checks (phrase detection on gapped audio, mid-gap boundaries,
  full-timeline coverage, short-phrase merging, per-phrase round-robin in
  `auto`, sequential/alternate snapping, the auto fallback on continuous
  singing, `duet_pitch` validation) — the duet + drive suite now passes
  56/56; the 39-check fix suite and the Gradio/web e2e suites still pass.

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

### 5. `assets/model_installer.py` — every pretrain download URL was a 404
- `_remote_url()` built `BASE_URL/<set>/<file>` while the paths in
  `assets/pretrains.py` are documented as relative to
  `pretrained/v2/<rate>/` — the `32k`/`40k`/`48k` segment was never inserted,
  so every pretrain download (all 84 entries of the table) hit
  `.../pretrained/v2/<Set>/<file>` and HuggingFace answered `404`.
- Fix: the rate label is now part of the URL
  (`BASE_URL/<rate>/<set dir>/<file>`), verified with an automated check that
  HEAD-requests every URL generated from the table: 84/84 answer `HTTP 200`.

### 6. `ui/tabs/train/train.py` — F0 method list did not match the backend
- The Training tab offered `crepe` / `fcpe`, but the feature-extraction backend
  (`programs/rvc/lib/predictors/f0.py`) only supports `rmvpe`, `rmvpe+` and
  `hpa-rmvpe` — selecting the other two always crashed step 2 with
  `ValueError`. Fix: the dropdown now offers exactly the supported methods.

### 7. PyTorch 2.6+ compatibility (`torch.load` / `weights_only`)
- `programs/rvc/infer/infer.py` (`load_model`),
  `programs/rvc/lib/predictors/RMVPE.py`, `programs/rvc/lib/predictors/FCPE.py`:
  `torch.load` now passes `weights_only=False` explicitly. PyTorch 2.6 changed
  the default to `weights_only=True`, which rejects many RVC / predictor
  checkpoints (requirements.txt has no upper torch bound, so users hit this on
  install). These are local files the user deliberately selected; behaviour
  matches pre-2.6 PyTorch. `train.py` already had a load fallback and is
  unchanged.

### 8. `programs/music_separation_code/utils.py` — MSST separation fixes
- `windowingArray` was mutated in place (`window = windowingArray; window[:fade_size] = 1`),
  so the fade-in/fade-out tweaks persisted into every later batch and TTA pass,
  corrupting the crossfades. Fix: operate on a copy.
- The first revision of the copy used numpy-style `.copy()`, but `_getWindowingArray()`
  builds the window with `torch.linspace` / `torch.ones`, i.e. it returns a **torch
  Tensor** — and tensors have no `.copy()` method. Every separation run therefore
  crashed at the first batch with
  `AttributeError: 'Tensor' object has no attribute 'copy'. Did you mean: 'copy_'?`
  (right after the `Processing audio chunks` progress bar appeared). Fix: the
  window is cloned with `Tensor.clone()`, which copies the values without
  sharing storage and keeps the array a tensor so it multiplies with the
  torch result/counter buffers below.
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
- Duet + Drive feature test (`scripts/test_duet_drive_features.py`, 56 checks):
  the duet weaving is verified with real audio (a loud and a quiet take are
  woven in auto / alternate / sequential / overlap mode and the RMS per segment
  proves which voice sings where), phrase detection and phrase-aligned swaps
  are covered with gapped audio, input-length trimming and parameter
  validation are included, the Drive helper is exercised against a fake drive
  root (copy, collision suffixes, folder validation, missing files), Colab
  detection is checked both ways, and the web API layer is driven through
  FastAPI's TestClient: duet 400 validations, Drive-only guards (400 / 403)
  and a full `save-drive` job lifecycle with a stubbed copy helper.
