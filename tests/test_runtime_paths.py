"""Exercise cache policy without importing PyTorch or downloading datasets."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("runtime_paths", Path(__file__).parents[1] / "src" / "runtime_paths.py")
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class RuntimePathTests(unittest.TestCase):
    def test_local_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.get_data_root(), os.path.expanduser("~/.torch"))

    def test_vercel_uses_scratch(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=True):
            self.assertEqual(paths.get_data_root(), str(Path(tempfile.gettempdir()) / "dlg-lab" / "cifar100"))

    def test_configured_cache_overrides_vercel_default(self):
        with patch.dict(os.environ, {"VERCEL": "1", "DLG_DATA_DIR": "custom-cache"}, clear=True):
            self.assertEqual(paths.get_data_root(), "custom-cache")

    def test_explicit_dataset_path_wins(self):
        with patch.dict(os.environ, {"DLG_DATA_DIR": "env-cache"}):
            self.assertEqual(paths.get_data_root("caller-cache"), "caller-cache")

    def test_huggingface_cache_is_writable_on_vercel(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=True):
            paths.configure_huggingface_cache()
            self.assertEqual(os.environ["HF_HOME"], str(Path(tempfile.gettempdir()) / "dlg-lab" / "huggingface"))

    def test_user_cache_configuration_is_preserved(self):
        with patch.dict(os.environ, {"VERCEL": "1", "HF_HOME": "custom-hub"}, clear=True):
            paths.configure_huggingface_cache()
            self.assertEqual(os.environ["HF_HOME"], "custom-hub")


if __name__ == "__main__":
    unittest.main()
