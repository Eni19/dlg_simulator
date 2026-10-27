import unittest
from unittest.mock import patch
import numpy as np
from src.progress import reporter
from src.federated_flower import start_federated_simulation


class FederatedProgressTests(unittest.TestCase):
    def test_round_is_reported_complete_only_after_all_client_fits(self):
        fits = []
        events = []

        class Client:
            def get_parameters(self, config):
                return [np.array([1.0])]

            def fit(self, parameters, config):
                fits.append(config["round"])
                return [parameters[0] + 1], 2, {"loss": 0.5}

        def receive(event):
            if event["phase"] == "training" and event["current"]:
                events.append((event["current"], len(fits)))

        token = reporter.set(receive)
        try:
            with patch("src.federated_flower.create_clients", return_value=[Client(), Client()]):
                result = start_federated_simulation(num_clients=2, num_rounds=2,
                    device="cpu", attacker_enabled=False, verbose=False)
        finally:
            reporter.reset(token)
        self.assertEqual(events, [(1, 2), (1, 2), (2, 4)])
        self.assertEqual(len(result["history"]), 2)
