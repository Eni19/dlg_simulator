"""Small CIFAR-100 demonstrations using Hugging Face viewer rows, not archives.

Viewer assets may be JPEG encoded: these are demo inputs, not bit-exact copies
of the original CIFAR tensors. Full scientific evaluations use the local dataset.
"""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from PIL import Image
from .runtime_paths import get_data_root


class OnlineDataUnavailable(RuntimeError):
    """Remote demo input unavailable; no CUDA computation has started."""


def read_remote(url, limit):
    """Retry a transient failure once, with a finite timeout on each attempt."""
    for attempt in range(2):
        try:
            with urlopen(url, timeout=15) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Remote response too large")
            return data
        except (TimeoutError, URLError) as error:
            if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == 1:
                raise


class OnlineCifar100:
    def __init__(self, root=None):
        self.cache = Path(get_data_root(root)) / "hf-viewer-cifar100-v1"

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError("CIFAR-100: índice deve estar entre 0 e 49999.")
        path = self.cache / f"{index}.json"
        if path.exists():
            try:
                return self._decode(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, KeyError, OSError):
                pass  # A corrupt cache entry is replaced by a validated response.
        try:
            query = urlencode(dict(dataset="uoft-cs/cifar100", config="cifar100",
                                   split="train", offset=index, length=1))
            payload = json.loads(read_remote("https://datasets-server.huggingface.co/rows?" + query, 1024 * 1024))
            entry = payload["rows"][0]
            if entry["row_idx"] != index or entry.get("truncated_cells"):
                raise ValueError("Unexpected dataset row")
            row = entry["row"]
            source = row["img"]["src"]
            parsed = urlparse(source)
            if parsed.scheme != "https" or parsed.hostname != "datasets-server.huggingface.co":
                raise ValueError("Unexpected image host")
            image_bytes = read_remote(source, 1024 * 1024)
            cached = {"image": base64.b64encode(image_bytes).decode("ascii"),
                      "label": row["fine_label"]}
            result = self._decode(cached)
        except Exception as error:
            raise OnlineDataUnavailable(
                f"Não foi possível obter a imagem CIFAR-100 {index} no Hugging Face. "
                "Falha na preparação dos dados, antes do cálculo em CPU/GPU. "
                "Verifique a conexão e tente novamente; não será baixado o dataset completo."
            ) from error
        self.cache.mkdir(parents=True, exist_ok=True)
        # Atomic replacement keeps simultaneous readers from seeing partial JSON.
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache,
                                             delete=False) as temporary:
                temp_name = temporary.name
                json.dump(cached, temporary)
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return result

    @staticmethod
    def _decode(cached):
        label = cached["label"]
        if not isinstance(label, int) or not 0 <= label < 100:
            raise ValueError("Invalid CIFAR label")
        with Image.open(io.BytesIO(base64.b64decode(cached["image"], validate=True))) as image:
            if image.size != (32, 32):
                raise ValueError("Invalid CIFAR image dimensions")
            return image.convert("RGB"), label
