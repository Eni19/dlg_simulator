import unittest
import torch
from src.models import create_lenet
from src.dlg_attack import DLGAttack


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware required")
class CrossDeviceInitializationTests(unittest.TestCase):
    def test_same_seed_has_identical_model_weights(self):
        cpu = create_lenet(device="cpu", seed=50)
        gpu = create_lenet(device="cuda", seed=50)
        for left, right in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(left, right.cpu(), rtol=0, atol=0)

    def test_same_seed_has_matching_dummy_inputs(self):
        model = create_lenet(seed=50)
        gradients = [torch.ones_like(p) for p in model.parameters()]
        inputs = []
        for device in ("cpu", "cuda"):
            attack = DLGAttack(batch_size=1, device=device, seed=50, verbose=False)
            images, labels = attack._smart_initialization(
                [g.to(device) for g in gradients], torch.tensor([19], device=device))
            self.assertTrue(images[0].is_leaf and labels[0].is_leaf)
            inputs.append((images[0].detach().cpu(), labels[0].detach().cpu()))
        for cpu, gpu in zip(*inputs):
            torch.testing.assert_close(cpu, gpu, rtol=1e-6, atol=1e-7)
