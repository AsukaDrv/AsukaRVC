/* AsukaRVC custom WebUI — vanilla JS frontend.
 *
 * Talks to the REST API mounted at /api/v1 by programs/webui/api.py.
 * Long tasks run as background jobs; this script starts them and polls
 * GET /jobs/{id} until they finish, streaming the log into the drawer.
 */

"use strict";

const API = "/api/v1";

/* ------------------------------------------------------------------ utils */

const $ = (id) => document.getElementById(id);

function toast(message, type = "info", ms = 5000) {
  const box = document.createElement("div");
  box.className = `toast ${type}`;
  box.textContent = message;
  $("toasts").appendChild(box);
  setTimeout(() => box.remove(), ms);
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API}${path}`, options);
  } catch (networkError) {
    throw new Error("Network error — is the server still running?");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (_) { /* keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

function bindSlider(id, fmt = (v) => v) {
  const input = $(id);
  const label = $(`${id}-val`);
  const update = () => { label.textContent = fmt(input.value); };
  input.addEventListener("input", update);
  update();
}

/* ------------------------------------------------------------------- tabs */

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".panel").forEach((panel) => { panel.hidden = panel.id !== `tab-${button.dataset.tab}`; });
    if (button.dataset.tab === "models") loadModelsTab();
    if (button.dataset.tab === "train") refreshTrainState();
  });
});

/* ------------------------------------------------------------- job engine */

let currentJobId = null;
let pollTimer = null;
let pollStart = 0;

function setBanner(job) {
  const banner = $("job-banner");
  if (!job) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.classList.toggle("done", job.status === "finished");
  banner.classList.toggle("error", job.status === "error" || job.status === "stopped");
  $("job-title").textContent = job.title;
  const seconds = Math.max(0, Math.round((Date.now() / 1000 - (pollStart || job.created_at))));
  const statusText = {
    pending: "queued…",
    running: `running · ${seconds}s`,
    finished: "finished",
    stopped: "stopped",
    error: "failed",
  }[job.status] || job.status;
  $("job-status").textContent = job.status === "running" ? statusText : `${statusText} · ${seconds}s`;
}

function setLogText(text) {
  $("log-output").textContent = text || "(empty)";
  const pre = $("log-output");
  pre.scrollTop = pre.scrollHeight;
  $("train-log").textContent = text || "(empty)";
  $("train-log").scrollTop = $("train-log").scrollHeight;
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  currentJobId = null;
}

async function watchJob(jobId, onFinished) {
  stopPolling();
  currentJobId = jobId;
  pollStart = Date.now() / 1000;
  $("log-drawer").hidden = false;

  const poll = async () => {
    let job;
    try {
      job = await api(`/jobs/${jobId}`);
    } catch (error) {
      console.warn(error);
      return; // transient error — keep polling
    }
    setBanner(job);
    setLogText(job.log);
    if (job.status === "finished" || job.status === "stopped" || job.status === "error") {
      stopPolling();
      setBanner(job);
      if (job.status === "finished") {
        if (onFinished) onFinished(job);
      } else if (job.status === "error") {
        toast(job.error || "The task failed — open the log for details.", "error", 9000);
      } else {
        toast("Task stopped.", "info");
      }
      refreshTrainState();
    }
  };
  await poll();
  pollTimer = setInterval(poll, 1200);
}

// The task log is a panel below the page content (never an overlay).
// When the user opens it explicitly, bring it into view; when a job
// auto-opens it on start, the page is left where the user was.
$("job-log-toggle").addEventListener("click", () => {
  const drawer = $("log-drawer");
  drawer.hidden = !drawer.hidden;
  if (!drawer.hidden) drawer.scrollIntoView({ behavior: "smooth", block: "nearest" });
});
$("log-close").addEventListener("click", () => { $("log-drawer").hidden = true; });
$("job-stop").addEventListener("click", async () => {
  try {
    const response = await api("/jobs/stop", { method: "POST" });
    toast(response.message, "info");
  } catch (error) {
    toast(error.message, "error");
  }
});

async function startJob(path, body, onFinished) {
  try {
    const response = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("inference-progress").hidden = path !== "/jobs/inference";
    await watchJob(response.job_id, (job) => {
      $("inference-progress").hidden = true;
      if (onFinished) onFinished(job);
    });
  } catch (error) {
    toast(error.message, "error", 8000);
  }
}

/* ------------------------------------------------------------------ health */

async function loadHealth() {
  try {
    const health = await api("/health");
    $("brand-version").textContent = `v${health.version}`;
    const badges = $("topbar-badges");
    badges.innerHTML = "";
    const addBadge = (text, cls = "") => {
      const el = document.createElement("span");
      el.className = `badge ${cls}`;
      el.textContent = text;
      badges.appendChild(el);
    };
    if (health.zero_gpu) addBadge("ZeroGPU", "warn");
    else addBadge(health.backend_ready ? "GPU backend ready" : "Backend not loaded (no torch)", health.backend_ready ? "ok" : "warn");
    addBadge("custom WebUI", "accent");
    if (health.active_job) {
      pollStart = health.active_job.created_at;
      watchJob(health.active_job.id);
    }
  } catch (error) {
    toast(`Health check failed: ${error.message}`, "error");
  }
}

/* ----------------------------------------------------------------- options */

const state = { options: null };

async function loadOptions() {
  state.options = await api("/options");
  const options = state.options;

  fillSelect("voice-model", options.voice_models.map((m) => m.name));
  fillSelect("backing-model", options.voice_models.map((m) => m.name));
  fillSelect("duet-model", options.voice_models.map((m) => m.name));
  fillSelect("f0-method", options.f0_methods);
  fillSelect("embedder-model", options.embedders);
  fillSelect("embedder-model-back", options.embedders);
  fillSelect("output-format", options.output_formats);
  fillSelect("export-format-final", options.final_formats);
  fillSelect("export-format-rvc-back", ["wav", "flac", "mp3"]);

  // Google Drive saving only exists inside a Colab runtime.
  const inColab = !!options.colab;
  $("drive-accordion").hidden = !inColab;
  $("result-drive-btn").hidden = !inColab;
  if (inColab && !value("drive-folder")) {
    $("drive-folder").value = options.drive_folder || "AsukaRVC";
  }
  const separation = options.separation || {};
  fillSelect("vocal-model-name", separation.vocals || []);
  fillSelect("karaoke-model-name", separation.karaoke || []);
  fillSelect("dereverb-model-name", separation.dereverb || []);
  fillSelect("denoise-model-name", separation.denoise || []);
  fillSelect("deecho-model-name", separation.deecho || []);
  $("separation-warn").hidden = !!options.separation_ready;

  fillSelect("train-sample-rate", options.train.sample_rates);
  fillSelect("train-f0-method", options.train.f0_methods);
  fillSelect("train-vocoder", options.train.vocoders);
  fillSelect("train-optimizer", options.train.optimizers);
  refreshPretrains();

  $("models-hint").textContent = options.voice_models.length
    ? `${options.voice_models.length} model(s) installed.`
    : "No models installed yet — grab one in the Models tab.";
}

function fillSelect(id, choices, defaultFirst = true) {
  const select = $(id);
  if (!select) return;
  select.innerHTML = "";
  choices.forEach((choice) => {
    // Choices may be plain strings, numbers (sample rates) or {name} objects.
    const label = typeof choice === "object" ? choice.name : String(choice);
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    select.appendChild(option);
  });
  if (defaultFirst && choices.length) {
    select.value = typeof choices[0] === "object" ? choices[0].name : String(choices[0]);
  }
}

async function refreshVoiceModels() {
  const data = await api("/models");
  state.options.voice_models = data.models;
  fillSelect("voice-model", data.models.map((m) => m.name));
  fillSelect("backing-model", data.models.map((m) => m.name));
  fillSelect("duet-model", data.models.map((m) => m.name));
  $("models-hint").textContent = data.models.length
    ? `${data.models.length} model(s) installed.`
    : "No models installed yet — grab one in the Models tab.";
  toast("Model list refreshed.", "success", 2500);
}

$("refresh-models").addEventListener("click", () => refreshVoiceModels().catch((e) => toast(e.message, "error")));
$("refresh-models-2").addEventListener("click", () => refreshVoiceModels().catch((e) => toast(e.message, "error")));

/* ----------------------------------------------------------------- source */

let uploadedAudioPath = null;

document.querySelectorAll("#source-mode .seg-btn").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#source-mode .seg-btn").forEach((b) => b.classList.toggle("active", b === button));
    const isFile = button.dataset.mode === "file";
    $("source-file-block").hidden = !isFile;
    $("source-url-block").hidden = isFile;
  });
});

function currentSongInput() {
  const isFile = document.querySelector("#source-mode .seg-btn.active").dataset.mode === "file";
  return isFile ? uploadedAudioPath : $("song-url").value.trim();
}

async function uploadAudioFile(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await api("/upload", { method: "POST", body: form });
  uploadedAudioPath = response.path;
  const zone = $("dropzone-file");
  zone.hidden = false;
  zone.textContent = `✓ ${response.name} — ready`;
  toast("Audio uploaded.", "success", 2500);
}

const dropzone = $("dropzone");
dropzone.addEventListener("click", () => $("audio-file-input").click());
dropzone.addEventListener("keydown", (event) => { if (event.key === "Enter") $("audio-file-input").click(); });
$("audio-file-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) uploadAudioFile(file).catch((error) => toast(error.message, "error"));
});
["dragover", "dragenter"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) uploadAudioFile(file).catch((error) => toast(error.message, "error"));
});

/* --------------------------------------------------------------- inference */

bindSlider("pitch-change");
bindSlider("pitch-change-all");
bindSlider("index-rate", (v) => Number(v).toFixed(2));
bindSlider("filter-radius");
bindSlider("rms-mix-rate", (v) => Number(v).toFixed(2));
bindSlider("protect", (v) => Number(v).toFixed(2));
bindSlider("crepe-hop-length");
bindSlider("pitch-back");
bindSlider("filter-radius-back");
bindSlider("index-rate-back", (v) => Number(v).toFixed(2));
bindSlider("rms-mix-rate-back", (v) => Number(v).toFixed(2));
bindSlider("protect-back", (v) => Number(v).toFixed(2));
bindSlider("hop-length-back");
bindSlider("main-gain");
bindSlider("backup-gain");
bindSlider("inst-gain");
bindSlider("backing-vocals-volume");
bindSlider("reverb-rm-size", (v) => Number(v).toFixed(2));
bindSlider("reverb-wet", (v) => Number(v).toFixed(2));
bindSlider("reverb-dry", (v) => Number(v).toFixed(2));
bindSlider("reverb-damping", (v) => Number(v).toFixed(2));
bindSlider("reverb-width", (v) => Number(v).toFixed(2));
bindSlider("inst-pitch-shift");
bindSlider("duet-split");
bindSlider("duet-interval");
bindSlider("segment-len", (v) => Number(v).toFixed(1));
bindSlider("include-mutes");
bindSlider("total-epoch");
bindSlider("save-every-epoch");
bindSlider("train-batch-size");

$("f0-method").addEventListener("change", () => { $("crepe-hop-block").hidden = $("f0-method").value !== "crepe"; });
$("infer-backing-vocals").addEventListener("change", () => { $("backing-block").hidden = !$("infer-backing-vocals").checked; });
$("hf-upload").addEventListener("change", () => { $("hf-block").hidden = !$("hf-upload").checked; });
$("duet-enabled").addEventListener("change", () => { $("duet-block").hidden = !$("duet-enabled").checked; });

const checked = (id) => $(id).checked;
const value = (id) => $(id).value;

function buildInferenceBody() {
  const songInput = currentSongInput();
  if (!songInput) throw new Error("Provide a song file or URL first.");
  if (!value("voice-model")) throw new Error("Select a voice model first.");
  const duetEnabled = checked("duet-enabled");
  if (duetEnabled && !value("duet-model")) throw new Error("Select the second duet voice model, or disable the duet option.");
  if (duetEnabled && value("duet-model") === value("voice-model")) throw new Error("The duet second voice must be different from the main voice model.");
  return {
    song_input: songInput,
    voice_model: value("voice-model"),
    pitch_change: Number(value("pitch-change")),
    pitch_change_all: Number(value("pitch-change-all")),
    keep_files: checked("keep-files"),
    main_gain: Number(value("main-gain")),
    backup_gain: Number(value("backup-gain")),
    inst_gain: Number(value("inst-gain")),
    backing_vocals_volume: Number(value("backing-vocals-volume")),
    index_rate: Number(value("index-rate")),
    filter_radius: Number(value("filter-radius")),
    rms_mix_rate: Number(value("rms-mix-rate")),
    f0_method: value("f0-method"),
    crepe_hop_length: Number(value("crepe-hop-length")),
    protect: Number(value("protect")),
    reverb_rm_size: Number(value("reverb-rm-size")),
    reverb_wet: Number(value("reverb-wet")),
    reverb_dry: Number(value("reverb-dry")),
    reverb_damping: Number(value("reverb-damping")),
    reverb_width: Number(value("reverb-width")),
    output_format: value("output-format"),
    extra_denoise: checked("extra-denoise"),
    vocal_model_name: value("vocal-model-name"),
    karaoke_model_name: value("karaoke-model-name"),
    dereverb_model_name: value("dereverb-model-name"),
    denoise_model_name: value("denoise-model-name"),
    deecho_model_name: value("deecho-model-name"),
    use_deecho: checked("use-deecho"),
    use_denoise: checked("use-denoise"),
    use_dereverb: checked("use-dereverb"),
    instruments_only: checked("instruments-only"),
    inst_pitch_shift: Number(value("inst-pitch-shift")),
    export_format_final: value("export-format-final"),
    split_audio: checked("split-audio"),
    autotune: checked("autotune"),
    embedder_model: value("embedder-model"),
    infer_backing_vocals: checked("infer-backing-vocals"),
    infer_backing_vocals_model: checked("infer-backing-vocals") ? value("backing-model") : null,
    infer_backing_vocals_index: value("backing-index"),
    pitch_back: Number(value("pitch-back")),
    filter_radius_back: Number(value("filter-radius-back")),
    index_rate_back: Number(value("index-rate-back")),
    rms_mix_rate_back: Number(value("rms-mix-rate-back")),
    protect_back: Number(value("protect-back")),
    pitch_extract_back: value("pitch-extract-back"),
    hop_length_back: Number(value("hop-length-back")),
    export_format_rvc_back: value("export-format-rvc-back"),
    split_audio_back: checked("split-audio-back"),
    autotune_back: checked("autotune-back"),
    embedder_model_back: value("embedder-model-back"),
    hf_upload: checked("hf-upload"),
    hf_repo_id: value("hf-repo-id"),
    hf_token: value("hf-token"),
    duet_enabled: duetEnabled,
    duet_model: duetEnabled ? value("duet-model") : null,
    duet_mode: value("duet-mode"),
    duet_split: Number(value("duet-split")),
    duet_interval: Number(value("duet-interval")),
    save_drive: checked("save-drive"),
    drive_folder: value("drive-folder") || "AsukaRVC",
  };
}

let lastCoverPath = null;

$("generate-btn").addEventListener("click", () => {
  $("result-block").hidden = true;
  let body;
  try {
    body = buildInferenceBody();
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  startJob("/jobs/inference", body, (job) => {
    const output = job.result && job.result.output;
    if (output && job.result.url) {
      lastCoverPath = output;
      $("result-audio").src = job.result.url;
      $("result-download").href = job.result.download_url || job.result.url;
      $("result-download").setAttribute("download", output.split("/").pop());
      $("result-path").textContent = output;
      $("result-block").hidden = false;
      toast("AI cover generated!", "success");
    } else {
      toast("Task finished, but no output file was reported — check the log.", "error");
    }
  });
});

/* Copy the last generated cover into the user's Google Drive (Colab only —
 * the button stays hidden everywhere else). */
$("result-drive-btn").addEventListener("click", () => {
  if (!lastCoverPath) { toast("Generate a cover first.", "error"); return; }
  startJob("/jobs/save-drive", { path: lastCoverPath, folder: value("drive-folder") || "AsukaRVC" }, () => {
    toast("Cover saved to Google Drive!", "success");
  });
});

/* ------------------------------------------------------------- models tab */

async function renderInstalledModels() {
  const data = await api("/models");
  const grid = $("installed-models");
  grid.innerHTML = "";
  if (!data.models.length) {
    grid.innerHTML = '<p class="hint">No models installed — download one below.</p>';
    return;
  }
  data.models.forEach((model) => {
    const card = document.createElement("div");
    card.className = "model-card";
    const name = document.createElement("strong");
    name.textContent = model.name;
    const badges = document.createElement("div");
    badges.className = "model-badges";
    badges.innerHTML = model.pth
      ? `<span>PTH ✓</span>${model.indices.length ? `<span>INDEX ×${model.indices.length}</span>` : '<span class="missing">no index</span>'}`
      : '<span class="missing">no .pth — unusable</span>';
    card.appendChild(name);
    card.appendChild(badges);
    grid.appendChild(card);
  });
}

function loadModelsTab() {
  renderInstalledModels().catch((e) => toast(e.message, "error"));
  loadPublicModels().catch((e) => toast(e.message, "error"));
}

/* public index */

const pubState = { tags: new Set(), query: "" };

function renderPublicModels(models) {
  const tbody = $("pub-tbody");
  tbody.innerHTML = "";
  if (!models.length) {
    const row = tbody.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 4;
    cell.textContent = "No models match the current filters.";
    return;
  }
  models.forEach((model) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = model.name;
    row.insertCell().textContent = model.description || "";
    row.insertCell().textContent = model.credit || "";
    const tagsCell = row.insertCell();
    model.tags.forEach((tag) => {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = tag;
      tagsCell.appendChild(span);
    });
    row.addEventListener("click", () => {
      $("dl-url").value = model.url;
      $("dl-name").value = model.name;
      document.querySelector("#tab-models .card:nth-of-type(2)").scrollIntoView({ behavior: "smooth" });
      toast("Download form filled — press “Download model”.", "info", 3500);
    });
  });
}

async function queryPublicModels() {
  const params = new URLSearchParams({ query: pubState.query, tags: [...pubState.tags].join(",") });
  const data = await api(`/public-models?${params}`);
  const chips = $("pub-tags");
  chips.innerHTML = "";
  data.tags.forEach((tag) => {
    const chip = document.createElement("span");
    chip.className = `chip ${pubState.tags.has(tag) ? "active" : ""}`;
    chip.textContent = tag;
    chip.addEventListener("click", () => {
      if (pubState.tags.has(tag)) pubState.tags.delete(tag);
      else pubState.tags.add(tag);
      queryPublicModels().catch((e) => toast(e.message, "error"));
    });
    chips.appendChild(chip);
  });
  renderPublicModels(data.models);
}

function loadPublicModels() { return queryPublicModels(); }

let pubSearchTimer = null;
$("pub-search").addEventListener("input", (event) => {
  clearTimeout(pubSearchTimer);
  pubSearchTimer = setTimeout(() => {
    pubState.query = event.target.value;
    queryPublicModels().catch((e) => toast(e.message, "error"));
  }, 350);
});

/* download / upload jobs */

$("dl-btn").addEventListener("click", () => {
  const url = value("dl-url").trim();
  const name = value("dl-name").trim();
  if (!url || !name) { toast("Both the download link and the model name are required.", "error"); return; }
  startJob("/jobs/model-download", { url, name }, () => {
    toast(`Model "${name}" downloaded!`, "success");
    renderInstalledModels().catch(() => {});
    refreshVoiceModels().catch(() => {});
  });
});

$("upload-model-btn").addEventListener("click", async () => {
  const zipInput = $("upload-zip");
  const name = value("upload-name").trim();
  if (!zipInput.files.length) { toast("Choose a .zip archive first.", "error"); return; }
  if (!name) { toast("Give the model a name.", "error"); return; }
  const form = new FormData();
  form.append("file", zipInput.files[0]);
  form.append("name", name);
  try {
    const response = await api("/jobs/model-upload", { method: "POST", body: form });
    $("log-drawer").hidden = false;
    await watchJob(response.job_id, () => {
      toast(`Model "${name}" uploaded!`, "success");
      renderInstalledModels().catch(() => {});
      refreshVoiceModels().catch(() => {});
    });
  } catch (error) {
    toast(error.message, "error");
  }
});

/* -------------------------------------------------------------- train tab */

function renderTrainState(stateInfo) {
  const chips = $("train-state-chips");
  chips.innerHTML = "";
  const addChip = (label, on) => {
    const chip = document.createElement("span");
    chip.className = `chip ${on ? "state-on" : "state-off"}`;
    chip.textContent = `${on ? "✓" : "○"} ${label}`;
    chips.appendChild(chip);
  };
  addChip("dataset sliced", stateInfo.sliced);
  addChip("features extracted", stateInfo.features);
  addChip("index built", stateInfo.index);
  addChip("ready to train", stateInfo.filelist);
  $("slice-btn").disabled = stateInfo.sliced;
  $("extract-btn").disabled = !stateInfo.sliced || stateInfo.features;
  $("index-btn").disabled = !stateInfo.features || stateInfo.index;
}

let trainStateTimer = null;
$("train-model-name").addEventListener("input", () => {
  clearTimeout(trainStateTimer);
  trainStateTimer = setTimeout(refreshTrainState, 400);
});

async function refreshTrainState() {
  const name = value("train-model-name").trim();
  if (!name) {
    renderTrainState({ sliced: false, features: false, index: false, filelist: false });
    return;
  }
  try {
    renderTrainState(await api(`/train/status?model=${encodeURIComponent(name)}`));
  } catch (_) { /* server hiccup — ignore */ }
}

async function refreshPretrains() {
  const vocoder = value("train-vocoder");
  const rate = value("train-sample-rate");
  // The selects are still empty while /options is loading — nothing to do.
  if (!vocoder || !rate) return;
  try {
    const data = await api(`/train/pretrains?vocoder=${encodeURIComponent(vocoder)}&sample_rate=${encodeURIComponent(rate)}`);
    fillSelect("train-pretrain", data.choices);
    $("train-pretrain").value = data.default;
  } catch (error) {
    toast(`Failed to load the pretrain list: ${error.message}`, "error");
  }
}
$("train-vocoder").addEventListener("change", refreshPretrains);
$("train-sample-rate").addEventListener("change", refreshPretrains);

$("dataset-files").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  const name = value("train-model-name").trim();
  if (!files.length) return;
  if (!name) { toast("Set the model name before uploading the dataset.", "error"); event.target.value = ""; return; }
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("name", name);
  try {
    const response = await api("/train/dataset-upload", { method: "POST", body: form });
    toast(`${response.saved} file(s) uploaded to the dataset.`, "success");
    renderTrainState(response.state);
  } catch (error) {
    toast(error.message, "error");
  }
});

function trainName() {
  const name = value("train-model-name").trim();
  if (!name) { toast("Set the model name first.", "error"); return null; }
  return name;
}

/* Name + sample rate for every training action. The rate select stays empty
 * until /options finishes loading, and Number("") is 0 — catch that here so
 * the job never starts with a broken 0 Hz / empty-pretrain request. */
function trainNameAndRate() {
  const name = trainName();
  if (!name) return null;
  const rate = Number(value("train-sample-rate"));
  if (!rate) {
    toast("UI options are still loading — sample rate is not ready yet. Try again in a few seconds.", "error");
    return null;
  }
  return { name, rate };
}

$( "slice-btn").addEventListener("click", () => {
  const common = trainNameAndRate();
  if (!common) return;
  startJob("/jobs/train-slice", {
    model_name: common.name,
    dataset_folder: value("dataset-folder"),
    sample_rate: common.rate,
    segment_len: Number(value("segment-len")),
    normalize: checked("train-normalize"),
  });
});

$( "extract-btn").addEventListener("click", () => {
  const common = trainNameAndRate();
  if (!common) return;
  startJob("/jobs/train-extract", {
    model_name: common.name,
    sample_rate: common.rate,
    f0_method: value("train-f0-method"),
    include_mutes: Number(value("include-mutes")),
  });
});

$("index-btn").addEventListener("click", () => {
  const name = trainName();
  if (!name) return;
  startJob("/jobs/train-index", { model_name: name });
});

$("train-btn").addEventListener("click", () => {
  const common = trainNameAndRate();
  if (!common) return;
  startJob("/jobs/train-start", {
    model_name: common.name,
    sample_rate: common.rate,
    total_epoch: Number(value("total-epoch")),
    save_every_epoch: Number(value("save-every-epoch")),
    batch_size: Number(value("train-batch-size")),
    vocoder: value("train-vocoder"),
    optimizer: value("train-optimizer"),
    gpus: value("train-gpus"),
    pretrain: value("train-pretrain") || "No pretrain",
    pretrain_g: value("train-pretrain-g"),
    pretrain_d: value("train-pretrain-d"),
    save_to_zip: checked("save-to-zip"),
    save_half: checked("save-half"),
  });
});

$("train-stop-btn").addEventListener("click", async () => {
  try {
    const response = await api("/jobs/stop", { method: "POST" });
    toast(response.message, "info");
  } catch (error) {
    toast(error.message, "error");
  }
});

/* ------------------------------------------------------------------- boot */

(async function boot() {
  loadHealth();
  try {
    await loadOptions();
  } catch (error) {
    toast(`Failed to load UI options: ${error.message}`, "error", 9000);
  }
})();
