"""Google Drive support for Google Colab sessions.

When AsukaRVC runs inside a Colab notebook every generated file (covers,
models, logs) lives on the ephemeral VM disk and disappears as soon as the
runtime is recycled. This module copies result files into the user's mounted
Google Drive so they survive the session.

Detection: the official ``google.colab`` package is only importable inside a
Colab VM, so importing it is the most reliable check. The ``/content`` +
``/content/.config`` pair is used as a fallback for notebook images that ship
the package preloaded but not yet in ``sys.modules``.
"""

import os
import shutil
import sys

from config.logging_setup import get_logger

logger = get_logger("colab_drive")

# Default mount point created by ``google.colab.drive.mount``.
DRIVE_MOUNT = "/content/drive"
# Default destination folder under "MyDrive".
DEFAULT_DRIVE_FOLDER = "AsukaRVC"


def in_colab() -> bool:
    """Return True when running inside a Google Colab notebook."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401 — presence alone is the signal

        return True
    except Exception:
        pass
    # Fallback without importing: Colab VMs always ship /content/.config,
    # while an empty "/content" alone could be any local folder.
    return os.path.isdir("/content") and os.path.isdir("/content/.config")


def ensure_drive_mounted(drive_mount: str = DRIVE_MOUNT) -> bool:
    """Make sure Google Drive is mounted. Returns True when "MyDrive" is usable.

    Raises a readable RuntimeError outside Colab or when the mount fails
    (e.g. the user dismissed the permission dialog in the notebook).
    """
    if not in_colab():
        raise RuntimeError(
            "Google Drive save is only available in Google Colab. "
            "Run the app from the Colab notebook to use this feature."
        )
    my_drive = os.path.join(drive_mount, "MyDrive")
    if os.path.isdir(my_drive):
        return True
    try:
        from google.colab import drive

        drive.mount(drive_mount)
    except Exception as error:
        raise RuntimeError(f"Could not mount Google Drive: {error}") from error
    if not os.path.isdir(my_drive):
        raise RuntimeError("Google Drive was mounted, but 'MyDrive' is not accessible.")
    return True


def save_file_to_drive(file_path: str, drive_folder: str = DEFAULT_DRIVE_FOLDER,
                       drive_mount: str = DRIVE_MOUNT) -> str:
    """Copy a result file to Google Drive and return the destination path.

    Files are stored under ``<drive_mount>/MyDrive/<drive_folder>/``. A name
    collision is resolved by appending " (1)", " (2)", … instead of silently
    overwriting an earlier export. ``drive_mount`` can be overridden (tests,
    custom mount points).
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Result file not found: {file_path}")
    ensure_drive_mounted(drive_mount)

    folder = (drive_folder or DEFAULT_DRIVE_FOLDER).strip().strip("/")
    folder = os.path.normpath(folder) if folder else DEFAULT_DRIVE_FOLDER
    # Reject absolute paths and traversal attempts in the user-supplied folder.
    if os.path.isabs(folder) or folder.startswith(".."):
        raise ValueError(f"Invalid Google Drive folder name: {drive_folder!r}")

    destination_dir = os.path.join(drive_mount, "MyDrive", folder)
    os.makedirs(destination_dir, exist_ok=True)

    destination = os.path.join(destination_dir, os.path.basename(file_path))
    stem, extension = os.path.splitext(destination)
    counter = 1
    while os.path.exists(destination):
        destination = f"{stem} ({counter}){extension}"
        counter += 1

    shutil.copy2(file_path, destination)
    logger.info("Saved '%s' to Google Drive: %s", os.path.basename(file_path), destination)
    return destination
