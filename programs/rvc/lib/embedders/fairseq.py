"""Load the ContentVec / HuBERT embedding model for training feature extraction.

The training pipeline (``programs/rvc/training/preprocess/preparing_data.py``)
expects the classic fairseq interface::

    model = load_model("assets/models/embedders/contentvec_base.pt")
    logits, padding_mask = model.extract_features(
        source=waveform, padding_mask=padding_mask, output_layer=12
    )

Two loading strategies are supported, in order of preference:

1. **Real fairseq** — when the ``fairseq`` package is installed and the
   ``.pt`` checkpoint exists on disk, the checkpoint is loaded through
   ``fairseq.checkpoint_utils`` exactly like the original RVC WebUI.
2. **Hugging Face fallback** — otherwise the HF-format ContentVec
   (``config.json`` + ``pytorch_model.bin``, downloaded on demand by
   ``programs.rvc.lib.utils.load_embedding``) is wrapped in a small adapter
   that reproduces the fairseq ``extract_features`` signature. This keeps the
   training pipeline usable without the (deprecated and hard to install)
   fairseq dependency.

Both paths return a ``torch.nn.Module`` with ``.to(device)`` / ``.eval()``
support, layer-12 hidden states and fairseq-style padding-mask semantics.
"""

import logging
import os

import torch
from torch import nn

from programs.rvc.lib.utils import load_embedding

logger = logging.getLogger("embedders.fairseq")

# Fairseq "output_layer" corresponds one-to-one with the index into the
# Hugging Face ``hidden_states`` tuple: index 0 is the CNN/positional
# embedding output and index i is the output of transformer layer i.
_DEFAULT_OUTPUT_LAYER = 12


class FairseqStyleHubert(nn.Module):
    """Adapter exposing a Hugging Face HuBERT model through the fairseq API."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    @property
    def config(self):
        """Expose the underlying model config (some callers introspect it)."""
        return self.hf_model.config

    def extract_features(self, source, padding_mask=None, output_layer=_DEFAULT_OUTPUT_LAYER):
        """Return ``(features, padding_mask)`` in fairseq style.

        Args:
            source: float waveform tensor of shape ``(batch, time)`` at 16 kHz.
            padding_mask: bool tensor, ``True`` on padded positions (fairseq
                semantics). ``None`` means no padding.
            output_layer: index (1-based) of the transformer layer whose
                output is returned.
        """
        attention_mask = None
        if padding_mask is not None:
            # Fairseq marks padded positions with True; Hugging Face marks
            # attended positions with True, so the mask must be inverted.
            attention_mask = (~padding_mask.bool()).long()

        output = self.hf_model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states
        layer_index = min(max(int(output_layer), 0), len(hidden_states) - 1)
        return hidden_states[layer_index], padding_mask


def _load_with_fairseq(checkpoint_path):
    """Load a fairseq ``.pt`` checkpoint using the fairseq library."""
    from fairseq import checkpoint_utils, tasks

    models, cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path]
    )
    model = models[0]
    task = tasks.setup_task(cfg.task)
    # The original RVC pipeline loads the model with this method to attach
    # normalization metadata; failing to do it is harmless, so ignore errors.
    try:
        model.load_state_dict(
            checkpoint_utils.load_checkpoint_to_cpu(cfg.path)["model"], strict=True
        )
    except Exception as error:  # noqa: BLE001 - purely cosmetic step
        logger.debug("Skipping fairseq state-dict refresh: %s", error)
    model.eval()
    return model


def load_model(hubert_model_path=None, embedder_model="contentvec"):
    """Return an embedding model exposing the fairseq ``extract_features`` API.

    Args:
        hubert_model_path: optional path to a fairseq ``.pt`` checkpoint
            (typically ``contentvec_base.pt``). Used only when the ``fairseq``
            package is importable.
        embedder_model: named embedder used by the Hugging Face fallback
            (``contentvec`` by default, matching the inference pipeline).
    """
    if (
        hubert_model_path
        and os.path.isfile(hubert_model_path)
        and str(hubert_model_path).endswith(".pt")
    ):
        try:
            import fairseq  # noqa: F401

            model = _load_with_fairseq(hubert_model_path)
            logger.info("Loaded fairseq checkpoint '%s'", hubert_model_path)
            return model
        except Exception as error:  # noqa: BLE001 - fall back to HF loading
            logger.warning(
                "fairseq could not load '%s' (%s); falling back to the "
                "Hugging Face ContentVec weights.",
                hubert_model_path,
                error,
            )

    hf_model = load_embedding(embedder_model, None)
    logger.info("Loaded Hugging Face '%s' embedder with fairseq-style API", embedder_model)
    return FairseqStyleHubert(hf_model)
