import os
from collections import OrderedDict

import torch


def replace_keys_in_dict(d, old_key_part, new_key_part):
    updated_dict = OrderedDict() if isinstance(d, OrderedDict) else {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part) if isinstance(key, str) else key
        updated_dict[new_key] = replace_keys_in_dict(value, old_key_part, new_key_part) if isinstance(value, dict) else value
    return updated_dict


def save_checkpoint(net_g, optim_g, net_d, optim_d, epoch, checkpoint_path):
    """Save a training checkpoint (generator + discriminator + optimizers)."""
    g_state = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
    d_state = net_d.module.state_dict() if hasattr(net_d, "module") else net_d.state_dict()

    checkpoint_data = {
        "epoch": epoch,
        "generator": {
            "model": g_state,
            "optimizer": optim_g.state_dict(),
            "lr": optim_g.param_groups[0]["lr"],
        },
        "discriminator": {
            "model": d_state,
            "optimizer": optim_d.state_dict(),
            "lr": optim_d.param_groups[0]["lr"],
        },
    }

    checkpoint_data = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint_data, ".parametrizations.weight.original1", ".weight_v"),
        ".parametrizations.weight.original0",
        ".weight_g",
    )

    torch.save(checkpoint_data, checkpoint_path)
    print(f"▸ Checkpoint saved: '{os.path.basename(checkpoint_path)}'", flush=True)


def load_checkpoint(checkpoint_path, net_g, optim_g, net_d, optim_d):
    """Load a training checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint = replace_keys_in_dict(
        replace_keys_in_dict(checkpoint, ".weight_v", ".parametrizations.weight.original1"),
        ".weight_g",
        ".parametrizations.weight.original0",
    )

    # Load the generator
    g_model = net_g.module if hasattr(net_g, "module") else net_g
    g_state_dict = g_model.state_dict()
    g_new_state = {k: checkpoint["generator"]["model"].get(k, v) for k, v in g_state_dict.items()}
    g_model.load_state_dict(g_new_state, strict=False)
    optim_g.load_state_dict(checkpoint["generator"]["optimizer"])

    # Load the discriminator
    d_model = net_d.module if hasattr(net_d, "module") else net_d
    d_state_dict = d_model.state_dict()
    d_new_state = {k: checkpoint["discriminator"]["model"].get(k, v) for k, v in d_state_dict.items()}
    d_model.load_state_dict(d_new_state, strict=False)
    optim_d.load_state_dict(checkpoint["discriminator"]["optimizer"])

    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded: '{os.path.basename(checkpoint_path)}' (epoch {epoch})", flush=True)
    return epoch


def extract_model(hps, ckpt, epoch, step, filepath, half=True):
    """Extract and save an inference-ready model from a training checkpoint.

    Args:
        hps: Model hyperparameters.
        ckpt: Model state dict (generator).
        epoch: Epoch number.
        step: Global step number.
        filepath: Full destination file path.
        half: When True, save weights as float16 (smaller file);
              when False, save weights as float32 (full precision).

    Returns:
        A success or error message.

    """
    try:

        def precision(value):
            return value.half() if half else value.float()

        opt = OrderedDict(weight={key: precision(value) for key, value in ckpt.items() if "enc_q" not in key})
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        # Main model metadata
        opt["model_name"] = hps.model_name
        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = hps.data.sample_rate
        opt["f0"] = True
        opt["version"] = "v2"
        opt["vocoder"] = hps.model.vocoder

        # Additional metadata
        opt["learning_environment"] = "MuXVS"

        # Save the model
        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(opt, ".parametrizations.weight.original1", ".weight_v"),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            filepath,
        )

        return f"▸ Model saved: '{os.path.basename(filepath)}'"
    except Exception as e:
        return f"Error while saving the model:\n{e}"


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)
