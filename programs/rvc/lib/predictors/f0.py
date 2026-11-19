"""F0 (fundamental frequency) predictors for the training feature pipeline.

Exposes :class:`RMVPEF0`, the predictor used by
``programs/rvc/training/preprocess/preparing_data.py``. It supports the three
pitch algorithms offered by the Training tab:

- ``rmvpe``      — the RMVPE model alone (fast, robust).
- ``rmvpe+``     — a median blend of RMVPE and Praat autocorrelation; Praat
  stabilises sustained notes, RMVPE fills noisy frames.
- ``hpa-rmvpe``  — a median blend of RMVPE and a Harvest-style Praat
  cross-correlation pass (permissive octave cost, wide search window), which
  recovers low / breathy pitches that autocorrelation tends to halve.

The RMVPE checkpoint is downloaded on first use, mirroring the behaviour of
the inference pipeline.
"""

import logging
import os
import time
import warnings

import numpy as np
import requests

from programs.rvc.lib.predictors.RMVPE import RMVPE0Predictor

logger = logging.getLogger("predictors.f0")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
PREDICTORS_DIR = os.path.join(_PROJECT_ROOT, "programs", "rvc", "models", "predictors")
RMVPE_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt"
RMVPE_PATH = os.path.join(PREDICTORS_DIR, "rmvpe.pt")

# RMVPE works at 16 kHz with a hop size of 160 samples (= 10 ms per frame).
_SAMPLE_RATE = 16000
_HOP_SIZE = 160

_F0_METHODS = ("rmvpe", "rmvpe+", "hpa-rmvpe")


def ensure_rmvpe_checkpoint():
    """Download the RMVPE checkpoint if it is not on disk yet."""
    if os.path.isfile(RMVPE_PATH):
        return
    os.makedirs(PREDICTORS_DIR, exist_ok=True)
    tmp_path = RMVPE_PATH + ".part"
    logger.info("Downloading RMVPE checkpoint to %s ...", RMVPE_PATH)
    last_error = None
    try:
        import torch

        torch.hub.download_url_to_file(RMVPE_URL, tmp_path)
        os.replace(tmp_path, RMVPE_PATH)
        return
    except Exception as error:  # noqa: BLE001 - fall back to plain HTTPS
        last_error = error
        logger.warning("torch.hub download failed (%s); trying plain HTTPS", error)
    try:
        with requests.get(RMVPE_URL, stream=True, timeout=120,
                          headers={"User-Agent": "Mozilla/5.0"}) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        os.replace(tmp_path, RMVPE_PATH)
    except Exception as error:  # noqa: BLE001 - nothing else to try
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to download the RMVPE checkpoint from '{RMVPE_URL}': "
            f"{last_error} / {error}"
        ) from error


def _praat_f0(audio, f0_min, f0_max, algorithm):
    """Estimate F0 with Praat (via parselmouth) at the RMVPE frame grid.

    Args:
        audio: mono float waveform at 16 kHz.
        f0_min / f0_max: pitch search bounds in Hz.
        algorithm: ``"ac"`` (autocorrelation, used by ``rmvpe+``) or
            ``"cc"`` (cross-correlation with Harvest-like permissive
            settings, used by ``hpa-rmvpe``).
    """
    import parselmouth

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=_SAMPLE_RATE)
    # 10 ms time step == one RMVPE frame; voicing thresholds are relaxed so
    # Praat reports a candidate on almost every frame and lets the median
    # blend decide which source to trust.
    common = {"time_step": _HOP_SIZE / _SAMPLE_RATE, "pitch_floor": f0_min, "pitch_ceiling": f0_max}
    if algorithm == "ac":
        pitch = sound.to_pitch_ac(very_accurate=False, voicing_threshold=0.2, octave_cost=0.01, **common)
    else:
        pitch = sound.to_pitch_cc(very_accurate=False, voicing_threshold=0.15, octave_cost=0.005,
                                  octave_jump_cost=0.2, **common)
    frequencies = pitch.selected_array["frequency"].astype(np.float64)
    return frequencies


def _blend(rmvpe_f0, praat_f0, target_len, f0_min):
    """Median-blend two F0 contours, aligning Praat's grid to RMVPE's.

    Unvoiced frames (value 0) are treated as missing data. Frames where both
    sources agree they are unvoiced stay unvoiced. Praat occasionally reports
    sub-audio artefacts far below the search floor (octave-halving); those are
    clamped to the valid range.
    """
    time_praat = np.arange(len(praat_f0), dtype=np.float64)
    time_target = np.arange(target_len, dtype=np.float64)
    praat_aligned = np.interp(time_target, time_praat, praat_f0) if len(praat_f0) else np.zeros(target_len)

    stack = np.vstack([
        np.where(rmvpe_f0 > 0, rmvpe_f0, np.nan),
        np.where(praat_aligned > 0, praat_aligned, np.nan),
    ])
    # All-NaN frames (both sources unvoiced) are expected, not an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(all="ignore"):
            blended = np.nanmedian(stack, axis=0)
    blended = np.nan_to_num(blended)
    blended[(blended > 0) & (blended < f0_min)] = 0
    return blended


class RMVPEF0:
    """RMVPE-based F0 predictor with optional Praat hybrid modes.

    Args:
        device: torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, device):
        self.device = device
        ensure_rmvpe_checkpoint()
        # Half precision is disabled on purpose: training extracts F0 from
        # many short files where accuracy matters more than speed.
        self.model_rmvpe = RMVPE0Predictor(RMVPE_PATH, is_half=False, device=device)

    def get_f0(self, audio, f0_min, f0_max, f0_method):
        """Compute the F0 contour of ``audio`` with the selected method.

        Args:
            audio: mono float waveform at 16 kHz.
            f0_min / f0_max: pitch search bounds in Hz.
            f0_method: one of ``rmvpe``, ``rmvpe+``, ``hpa-rmvpe``.

        Returns:
            F0 contour aligned to the RMVPE frame grid; unvoiced frames are 0.
        """
        if f0_method not in _F0_METHODS:
            raise ValueError(
                f"Unknown F0 method '{f0_method}'. Supported methods: {', '.join(_F0_METHODS)}."
            )
        start_time = time.time()
        f0 = self.model_rmvpe.infer_from_audio(audio, thred=0.03)

        if f0_method != "rmvpe":
            algorithm = "ac" if f0_method == "rmvpe+" else "cc"
            try:
                praat = _praat_f0(audio, f0_min, f0_max, algorithm)
                f0 = _blend(f0, praat, len(f0), f0_min)
            except Exception as error:  # noqa: BLE001 - Praat is an enhancer only
                logger.warning(
                    "Praat %s pass failed (%s); falling back to plain rmvpe.",
                    algorithm,
                    error,
                )

        logger.info(
            "F0 extracted with '%s' in %.2f s (%d frames).",
            f0_method,
            time.time() - start_time,
            len(f0),
        )
        return f0
