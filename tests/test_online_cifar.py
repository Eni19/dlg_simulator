import base64
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from src.online_cifar import OnlineCifar100, read_remote


class OnlineCifarTests(unittest.TestCase):
    def test_timeout_retries_once_then_succeeds(self):
        with patch("src.online_cifar.urlopen", side_effect=[TimeoutError(), io.BytesIO(b"ok")]) as fetch:
            self.assertEqual(read_remote("https://example.test", 100), b"ok")
            self.assertEqual(fetch.call_count, 2)

    def test_persistent_timeout_is_bounded(self):
        with patch("src.online_cifar.urlopen", side_effect=TimeoutError()) as fetch:
            with self.assertRaises(TimeoutError):
                read_remote("https://example.test", 100)
            self.assertEqual(fetch.call_count, 2)

    def test_fetches_one_row_then_works_offline(self):
        image = io.BytesIO()
        Image.new("RGB", (32, 32), (12, 34, 56)).save(image, format="PNG")
        metadata = {"rows": [{"row_idx": 17, "row": {
            "img": {"src": "https://datasets-server.huggingface.co/fixture.png"},
            "fine_label": 42}}]}
        with tempfile.TemporaryDirectory() as root:
            dataset = OnlineCifar100(root)
            with patch("src.online_cifar.urlopen", side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(image.getvalue())]) as fetch:
                result, label = dataset[17]
                self.assertEqual(label, 42)
                self.assertEqual(result.getpixel((0, 0)), (12, 34, 56))
                self.assertIn("offset=17&length=1", fetch.call_args_list[0].args[0])
                self.assertEqual(fetch.call_count, 2)
            with patch("src.online_cifar.urlopen", side_effect=AssertionError("Must use cache")):
                cached, label = OnlineCifar100(root)[17]
                self.assertEqual(cached.tobytes(), result.tobytes())
                self.assertEqual(label, 42)

    def test_bad_indices_do_not_make_requests(self):
        with tempfile.TemporaryDirectory() as root, patch("src.online_cifar.urlopen") as fetch:
            for index in (-1, 50000, "1"):
                with self.assertRaises(IndexError):
                    OnlineCifar100(root)[index]
            fetch.assert_not_called()

    def test_network_failure_does_not_fall_back_to_archive(self):
        with tempfile.TemporaryDirectory() as root, patch("src.online_cifar.urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, "não será baixado o dataset completo"):
                OnlineCifar100(root)[0]

    def test_invalid_labels_and_dimensions_are_rejected(self):
        image = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image, format="PNG")
        for label in (-1, 100, "19", 19):
            with self.assertRaises(ValueError):
                OnlineCifar100._decode({"label": label, "image": base64.b64encode(image.getvalue()).decode()})
