# AsukaRVC  

## Overview  
**AsukaRVC** is a WebUI designed to create **AI-powered song covers** with customizable voice models, pitch adjustments, and audio mixing options.  
Built with ❤️ for music enthusiasts and creators, it allows you to transform vocals, generate new covers, and manage models easily.

| Description | Link |  
| ----------- | ---- |  
| 📙 Colab Notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BF667/notebook/blob/main/colab_notebook.ipynb) |  
| 🎉 Repository | [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=flat-square&logo=github)](https://github.com/AsukaDrv/AsukaRVC) |  

## Features

### Inference (AI cover generation)
- Music separation with MSST (Roformer / MDX23C) and VR/MDX (UVR) models: vocals,
  karaoke, dereverb, deecho and denoise chains.
- RVC voice conversion of the main vocals, with pitch, index rate, filter radius,
  RMS mix, protect rate, autotune, split audio and selectable F0 algorithms
  (`rmvpe`, `fcpe`, `crepe`).
- **Backing Vocals RVC**: convert the backup vocals with a second RVC model
  (own pitch, index, filter settings and export format) and mix it into the
  final cover with an independent volume control.
- Audio effects (reverb, compressor, high-pass), overall pitch shift,
  instrumental-only export and Hugging Face upload.

### Model management
- Download voice models from HuggingFace / Pixeldrain URLs or the public model index.
- Upload locally trained RVC v2 models as zip archives.

### Training
- Full training pipeline in the **Training** tab: dataset slicing → F0 +
  HuBERT feature extraction (RMVPE / RMVPE+ / HPA-RMVPF hybrid modes) → FAISS
  index building → model training with a live log and a stop button.
- Configurable sample rate (32k/40k/48k), vocoder (HiFi-GAN, MRF HiFi-GAN,
  RefineGAN), optimizer (AdamW, AdaBelief, PolOpt), custom or built-in
  pretrains, checkpoint resume and TensorBoard embedding.

### Running
```bash
pip install -r requirements.txt   # apt packages: aria2, ffmpeg, sox (packages.txt)
python ui/app.py                  # add --share for a public Gradio link
```

#### Two UI modes
Pick the interface with `--ui` (default: `gradio`):

```bash
# 1. Classic Gradio Blocks UI
python ui/app.py --ui gradio

# 2. Custom HTML/CSS/JS UI served by the same Gradio server
python ui/app.py --ui web --share
```

The `web` mode keeps the Gradio-managed server (so `--share` still produces a
public `*.gradio.live` URL that proxies every route) and mounts:

| Route        | Content                                                     |
| ------------ | ----------------------------------------------------------- |
| `/webui/`    | Custom frontend (vanilla HTML/CSS/JS, no build step)         |
| `/api/v1/…`  | REST API (`programs/webui/api.py`): models, inference jobs, model download/upload, training steps, file serving |
| `/`          | Redirect to the custom UI                                    |

Jobs (cover generation, model downloads, training steps) run in background
threads and are polled from the browser; a single public URL serves both the
frontend and the API. The classic Gradio UI remains available on the same
installation through mode 1.

CLI cover generation is also available:
```bash
python -m programs.core -i <song-url-or-path> -dir <rvc_model_folder> -p 1 --no-keep-files
```

# Credits


- [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen)
-  [MuXVS](https://github.com/MuXolotl/MuXVS.git)
