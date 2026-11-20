"""Download and cache RVC pretrain checkpoints (generator + discriminator).

Pretrains are stored in ``assets/models/pretrains`` and shared between all
training runs. ``ensure_pretrains`` is a generator so the Training tab can
show download progress line by line in its live log; it returns the local
paths of the (G, D) pair via ``StopIteration.value``.
"""

import os

import requests

from assets.pretrains import BASE_URL, pretrain_files, rate_label

PRETRAINS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pretrains"
)

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _remote_url(remote_path, rate):
    """Full URL of a pretrain file: BASE_URL/<rate>/<set dir>/<file>.

    Paths in assets/pretrains.py are relative to ``pretrained/v2/<rate>/``,
    so the rate segment (32k/40k/48k) must be inserted into the URL —
    leaving it out produced 404s for every pretrain download.
    """
    directory, filename = os.path.split(remote_path)
    return f"{BASE_URL.rstrip('/')}/{rate}/{directory}/{filename}"


def download_with_progress(url, local_path):
    """Stream ``url`` to ``local_path``, yielding human readable progress lines.

    The download goes to a temporary ``.part`` file first and is moved into
    place only when complete, so an interrupted download never leaves a
    truncated checkpoint that later looks valid.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".part"

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0) or 0)
        downloaded = 0
        with open(tmp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = int(downloaded * 100 / total_size)
                    yield f"  {percent}% ({downloaded // _DOWNLOAD_CHUNK_SIZE}/{total_size // _DOWNLOAD_CHUNK_SIZE} MB)"
                else:
                    yield f"  {downloaded // _DOWNLOAD_CHUNK_SIZE} MB"

    os.replace(tmp_path, local_path)
    yield f"  saved to {os.path.basename(local_path)}"


def _ensure_file(url, local_path):
    """Generator: download one pretrain if missing, yield progress lines,
    return the local path."""
    if os.path.isfile(local_path):
        yield f"  {os.path.basename(local_path)} already downloaded"
        return local_path
    yield f"  downloading {url}"
    yield from download_with_progress(url, local_path)
    return local_path


def ensure_pretrains(vocoder, set_name, sample_rate):
    """Generator: make sure the (G, D) pretrain pair of a built-in set exists.

    Returns ``(g_path, d_path)`` or ``(None, None)`` when the selected set has
    no files for this sample rate. Raises ``ValueError`` with a readable
    message when the set itself is unknown, so the Training tab can turn it
    into a UI error.
    """
    files = pretrain_files(vocoder, set_name, sample_rate)
    if files is None:
        raise ValueError(
            f"Pretrain set '{set_name}' has no {vocoder} files for "
            f"{int(sample_rate)} Hz. Choose another set or 'No pretrain'."
        )
    d_remote, g_remote = files
    rate = rate_label(sample_rate)

    d_url = _remote_url(d_remote, rate)
    g_url = _remote_url(g_remote, rate)
    d_path = os.path.join(PRETRAINS_DIR, os.path.basename(d_remote))
    g_path = os.path.join(PRETRAINS_DIR, os.path.basename(g_remote))

    resolved_g = yield from _ensure_file(g_url, g_path)
    resolved_d = yield from _ensure_file(d_url, d_path)
    return resolved_g, resolved_d
