"""Training file storage paths depending on the runtime environment.

Locally everything lives in the repository folder; in cloud notebooks the code
and trained models are stored separately (the notebook disk is ephemeral, models
go to persistent storage). This applies only to training: everything else is
always written to the repository.

Colab follows the PolTrain_Colab.ipynb scheme: code in /content/, models in
SAVE_DIR on Google Drive. Kaggle follows the PolTrain_kaggle.ipynb scheme: code
in /kaggle/working/, models in SAVE_DIR in the same place.
"""

import os

from assets.notebook_check import colab_check, kaggle_check

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLAB_DRIVE_MOUNT = "/content/drive"
COLAB_TRAINING_DIR = "/content/drive/MyDrive/AsukaRVC"
KAGGLE_TRAINING_DIR = "/kaggle/working/Models"


def training_logs_dir() -> str:
    """Root of training experiments for the current environment. Path only, no checks."""
    if colab_check():
        return COLAB_TRAINING_DIR
    if kaggle_check():
        return KAGGLE_TRAINING_DIR
    return os.path.join(PROJECT_ROOT, "logs")


def is_colab_drive_mounted() -> bool:
    """Whether Google Drive is mounted (to avoid writing to a local shadow folder)."""
    return os.path.ismount(COLAB_DRIVE_MOUNT)


def require_training_logs_dir() -> str:
    """Root of experiments for actions: checks Drive and creates the folder."""
    path = training_logs_dir()
    if colab_check() and not is_colab_drive_mounted():
        raise RuntimeError("Google Drive is not mounted: run drive.mount('/content/drive') in the notebook and try again.")
    os.makedirs(path, exist_ok=True)
    return path
