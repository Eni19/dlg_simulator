import os
import unittest
from unittest.mock import patch

from src.utils import get_device


class DeviceSelectionTests(unittest.TestCase):
    def test_request_selection_overrides_server_environment(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "cpu"}), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="RTX"):
            self.assertEqual(get_device(False, requested="cuda").type, "cuda")
        with patch.dict(os.environ, {"DLG_DEVICE": "cuda"}):
            self.assertEqual(get_device(False, requested="cpu").type, "cpu")

    def test_force_cpu_even_with_cuda(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "cpu"}), patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_device(False).type, "cpu")

    def test_cuda_required_does_not_silently_fall_back(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "cuda"}), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                get_device(False)

    def test_auto_cuda(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "auto"}), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.get_device_name", return_value="RTX"):
            self.assertEqual(get_device(False).type, "cuda")

    def test_auto_cpu(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "auto", "DLG_USE_DML": "0", "DLG_USE_MPS": "0"}), patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(get_device(False).type, "cpu")

    def test_invalid_setting(self):
        with patch.dict(os.environ, {"DLG_DEVICE": "typo"}):
            with self.assertRaises(ValueError):
                get_device(False)


if __name__ == "__main__":
    unittest.main()
