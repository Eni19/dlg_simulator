"""Deployment policy shared by every compute entry point, not a client flag."""
import base64
from functools import lru_cache
import io
import json
import os
from pathlib import Path
from PIL import Image

GITHUB_URL = "https://github.com/Eni19/dlg_simulator#executar-localmente"
ITERATIONS = (10, 25, 50)
PAIR_ITERATIONS = (*ITERATIONS, 100, 200)
SAMPLE_IDS = (0, 1, 2)
MAX_BATCH_SIZE = 2


def is_showcase():
    # A public Vercel deployment cannot accidentally enable unrestricted compute.
    return os.environ.get("VERCEL") == "1" or os.environ.get("DLG_APP_MODE") == "showcase"


@lru_cache(maxsize=1)
def sample_manifest():
    return json.loads((Path(__file__).parent / "data" / "showcase_samples.json").read_text(encoding="utf-8"))


class ShowcaseDataset:
    def __len__(self):
        return len(SAMPLE_IDS)

    def __getitem__(self, index):
        if index not in SAMPLE_IDS:
            raise IndexError("Escolha uma das imagens de exemplo da demonstração.")
        sample = sample_manifest()["samples"][index]
        with Image.open(io.BytesIO(base64.b64decode(sample["image"]))) as image:
            return image.convert("RGB"), sample["label"]


def validate_showcase(kind, request):
    if not is_showcase():
        return
    if kind != "attack":
        raise ValueError("Este cenário está disponível na versão completa do GitHub. O showcase online usa reconstrução centralizada, com até duas imagens e CPU.")
    if request.batch_size not in range(1, MAX_BATCH_SIZE + 1):
        raise ValueError("Escolha uma ou duas imagens para o showcase. Na versão completa, você pode explorar lotes de até oito imagens.")
    limits = {"seed": 50, "cosine_weight": 0.5,
              "tv_coeff_start": 1e-5, "tv_coeff_end": 1e-7, "label_anchor_weight": 0.1}
    # These match the frontend's existing defaults; no arbitrary optimization knobs online.
    if any(getattr(request, key) != value for key, value in limits.items()):
        raise ValueError("Para esta demonstração, mantenha os ajustes preparados: seed 50 e os pesos padrão. Os ajustes livres estão na versão completa.")
    allowed_iterations = PAIR_ITERATIONS if request.batch_size == 2 else ITERATIONS
    if request.device not in {"cpu", "auto"} or request.num_iters not in allowed_iterations:
        raise ValueError("O showcase usa CPU: escolha 10, 25 ou 50 iterações para uma imagem, ou até 100 e 200 para duas. GPU e ajustes livres estão na versão completa.")
    if request.image_indices is not None and (
        len(request.image_indices) != request.batch_size
        or len(set(request.image_indices)) != len(request.image_indices)
        or any(index not in SAMPLE_IDS for index in request.image_indices)
    ):
        raise ValueError("Escolha uma ou duas imagens diferentes entre os três exemplos preparados para a demonstração.")
