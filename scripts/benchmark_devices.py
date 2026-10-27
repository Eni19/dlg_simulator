"""Compare the actual DLG optimizer on synthetic data; no dataset download."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models import create_lenet
from src.dlg_attack import DLGAttack


def run(device, iterations, batch_size):
    torch.manual_seed(42)
    model = create_lenet(device=device, seed=42)
    images = torch.rand(batch_size, 3, 32, 32, device=device)
    labels = torch.arange(batch_size, device=device) % 100
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    gradients = [g.detach() for g in torch.autograd.grad(loss, model.parameters())]
    attack = DLGAttack(batch_size=batch_size, num_iters=iterations,
                       device=device, seed=42, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = attack.run(gradients, model)
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert torch.isfinite(result["reconstructed"]).all(), "Non-finite reconstruction"
    assert all(torch.isfinite(torch.tensor(result["loss_history"]))), "Non-finite loss"
    assert result["reconstructed"].device.type == device
    return seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or not 1 <= args.batch_size <= 8:
        parser.error("iterations >= 1; batch-size between 1 and 8")
    timings = {}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        run(device, 1, args.batch_size)  # Warm up kernels before measurement.
        timings[device] = run(device, args.iterations, args.batch_size)
    print(json.dumps({"iterations": args.iterations, "batch_size": args.batch_size,
                      "seconds": timings,
                      "cpu_over_cuda": timings["cpu"] / timings["cuda"] if "cuda" in timings else None}, indent=2))
