"""Built-in TensorBoard: background launch and link with pinned charts.

Like in Applio: the server is started inside the interface process (`tensorboard.program`)
and opened via an iframe right in the training tab. A repeated launch returns
the already-running instance. The tensorboard import is lazy — the interface works without it,
and the button shows an error.
"""

import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import types
import urllib.parse
from pathlib import Path

DEFAULT_TENSORBOARD_PORT = 6006

# Scalar tags pinned in the opened TensorBoard (written by rvc/training/train.py).
PINNED_TAGS = ["loss/g/mel", "loss/g/total"]

# Keep running servers so GC doesn't collect them before the interface process ends.
_TENSORBOARD_SERVERS = []
_TENSORBOARD_URL = None
_TENSORBOARD_LOCK = threading.Lock()


def _pinned_url(url: str, tags) -> str:
    """Link to TensorBoard with pinned cards for the required scalars."""
    cards = [{"plugin": "scalars", "tag": tag} for tag in tags]
    query = urllib.parse.quote(json.dumps(cards, separators=(",", ":")))
    return f"{url.rstrip('/')}/?pinnedCards={query}"


def _force_tensorboard_stub() -> None:
    """Make TensorBoard run on its stub TF without importing the real one.

    On startup TensorBoard does `import tensorflow` just to check its
    presence (`tensorboard.compat.tf`). MuXVS doesn't use TensorFlow, and the
    import itself in fresh environments spews junk like "Could not find cuda
    drivers" (Colab images moved to CUDA 13, while TF is built for CUDA 12;
    log filters like TF_CPP_MIN_LOG_LEVEL don't silence this spam in newer TF).

    The proper switch is the `tensorboard.compat.notf` module: when present,
    the server runs on `tensorflow_stub`. We create the file in the package
    installation — this covers TensorBoard child processes too; if the
    installation is read-only, we substitute the module at least for the
    local launch in this process.
    """
    try:
        if importlib.util.find_spec("tensorboard.compat.notf"):
            return
    except ImportError:
        return  # tensorboard not installed — nothing to silence
    except ValueError:
        return  # already substituted in sys.modules

    import tensorboard

    notf_path = Path(tensorboard.__file__).parent / "compat" / "notf.py"
    try:
        if not notf_path.exists():
            notf_path.write_text("# Force tensorflow_stub: AsukaRVC does not use TensorFlow.\n", encoding="utf-8")
    except OSError:
        module = types.ModuleType("tensorboard.compat.notf")
        module.__spec__ = importlib.machinery.ModuleSpec("tensorboard.compat.notf", loader=None)
        sys.modules.setdefault("tensorboard.compat.notf", module)


def launch_tensorboard(logs_dir: str, port: int = DEFAULT_TENSORBOARD_PORT) -> str:
    """Launch TensorBoard in a background thread.

    Returns a link with pinned cards or an 'Error...' string.
    The server lives as long as the interface process.
    """
    global _TENSORBOARD_URL

    os.makedirs(logs_dir, exist_ok=True)

    with _TENSORBOARD_LOCK:
        if _TENSORBOARD_URL:
            return _TENSORBOARD_URL
        try:
            _force_tensorboard_stub()
            from tensorboard import program

            board = program.TensorBoard()
            board.configure(argv=[None, "--logdir", logs_dir, "--port", str(port), "--path_prefix", "/tensorboard"])
            url = board.launch()
        except Exception as error:
            return f"TensorBoard launch error: {error}"
        _TENSORBOARD_SERVERS.append(board)
        _TENSORBOARD_URL = _pinned_url(url, PINNED_TAGS)
        return _TENSORBOARD_URL
