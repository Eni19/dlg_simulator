"""Writable dataset/cache paths for local Python and Vercel functions."""

import os
from pathlib import Path
import tempfile


def get_data_root(root: str | None = None) -> str:
    """Explicit argument > DLG_DATA_DIR > Vercel scratch directory > local cache."""
    if root is not None:
        return os.path.expanduser(root)
    configured = os.environ.get("DLG_DATA_DIR")
    if configured:
        return os.path.expanduser(configured)
    if os.environ.get("VERCEL") == "1":
        return str(Path(tempfile.gettempdir()) / "dlg-lab" / "cifar100")
    return os.path.expanduser("~/.torch")


def configure_huggingface_cache() -> None:
    """Keep hub metadata and dataset downloads on the writable scratch filesystem."""
    if os.environ.get("VERCEL") == "1":
        os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "dlg-lab" / "huggingface"))
