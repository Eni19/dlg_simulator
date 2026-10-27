import base64
import hashlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.showcase import is_showcase, validate_showcase, ShowcaseDataset, sample_manifest


def request(**changes):
    values = dict(batch_size=1, seed=50, cosine_weight=0.5,
                  tv_coeff_start=1e-5, tv_coeff_end=1e-7, label_anchor_weight=0.1,
                  device="cpu", num_iters=50, image_indices=[0])
    return SimpleNamespace(**(values | changes))


class ShowcasePolicyTests(unittest.TestCase):
    def test_vercel_is_always_restricted(self):
        with patch.dict(os.environ, {"VERCEL": "1", "DLG_APP_MODE": "full"}):
            self.assertTrue(is_showcase())
            with self.assertRaises(ValueError):
                validate_showcase("federated", None)

    def test_supported_presets_are_accepted(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}):
            for iterations in (10, 25, 50):
                for sample in (0, 1, 2):
                    validate_showcase("attack", request(num_iters=iterations, image_indices=[sample]))
                for pair in ([0, 1], [0, 2], [1, 2], [2, 0]):
                    validate_showcase("attack", request(num_iters=iterations, batch_size=2, image_indices=pair))
            for iterations in (100, 200):
                for pair in ([0, 1], [0, 2], [1, 2]):
                    validate_showcase("attack", request(num_iters=iterations, batch_size=2, image_indices=pair))

    def test_longer_presets_only_apply_to_two_images(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}):
            for changes in (dict(num_iters=100), dict(num_iters=200),
                            dict(batch_size=2, image_indices=[0, 1], num_iters=201),
                            dict(batch_size=2, image_indices=[0, 1], num_iters=500)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_showcase("attack", request(**changes))

    def test_batches_require_distinct_bundled_images_and_matching_size(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}):
            for changes in (dict(batch_size=3, image_indices=[0, 1, 2]),
                            dict(batch_size=2, image_indices=[0, 0]),
                            dict(batch_size=2, image_indices=[0]),
                            dict(batch_size=1, image_indices=[0, 1]),
                            dict(batch_size=2, image_indices=[0, 3]),
                            dict(batch_size=1, image_indices=[])):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_showcase("attack", request(**changes))

    def test_modified_requests_cannot_enable_full_features(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}):
            for kind in ("federated", "defense", "distribution"):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    validate_showcase(kind, request())
            for changes in (dict(device="cuda"), dict(batch_size=8), dict(num_iters=5000),
                            dict(num_iters=49), dict(seed=123), dict(image_indices=[40055]),
                            dict(cosine_weight=1), dict(label_anchor_weight=0),
                            dict(tv_coeff_start=0.01), dict(tv_coeff_end=0.01)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    validate_showcase("attack", request(**changes))

    def test_local_full_mode_preserves_unrestricted_configuration(self):
        with patch.dict(os.environ, {"VERCEL": "0", "DLG_APP_MODE": "full"}):
            self.assertFalse(is_showcase())
            validate_showcase("federated", request(device="cuda", batch_size=8, num_iters=5000))

    def test_samples_are_bundled_and_match_provenance_hashes(self):
        dataset = ShowcaseDataset()
        self.assertEqual(len(dataset), 3)
        for sample in sample_manifest()["samples"]:
            self.assertEqual(hashlib.sha256(base64.b64decode(sample["image"])).hexdigest(), sample["sha256"])
            image, label = dataset[sample["id"]]
            self.assertEqual(image.size, (32, 32))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(label, sample["label"])


class ShowcaseEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.api = app
        from fastapi.testclient import TestClient
        cls.client = TestClient(app.app)

    def test_restricted_routes_stop_before_dataset_work(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}), patch.object(self.api, "get_dataset") as dataset:
            for route in ("/api/federated", "/api/run-defense", "/api/intercept", "/api/data-distribution"):
                with self.subTest(route=route):
                    self.assertEqual(self.client.post(route, json={}).status_code, 403)
            dataset.assert_not_called()

    def test_stream_validates_configuration_before_starting_worker(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}), patch.object(self.api, "stream_experiment") as stream:
            for changes in (dict(device="cuda"), dict(num_iters=5000), dict(batch_size=3), dict(batch_size=2, image_indices=[0, 0])):
                response = self.client.post("/api/experiments/attack/stream", json=vars(request(**changes)))
                self.assertEqual(response.status_code, 422)
            stream.assert_not_called()

    def test_legacy_compute_endpoint_cannot_bypass_stream_admission(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}), patch.object(self.api, "get_dataset") as dataset:
            response = self.client.post("/api/run-attack", json=vars(request()))
            self.assertEqual(response.status_code, 403)
            dataset.assert_not_called()

    def test_showcase_does_not_evaluate_full_test_dataset(self):
        with patch.dict(os.environ, {"DLG_APP_MODE": "showcase"}):
            self.assertIsNone(self.api.compute_model_accuracy(None, None))
