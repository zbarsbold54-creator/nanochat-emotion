"""CPU causality regression for the RDA mechanism and capacity control."""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TESTS))

from test_rda_mechanism import make_model


class MechanismCausalityCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        self.tokens = torch.randint(1, 128, (2, 16))

    def assert_causal(self, name, model):
        model.eval()
        with torch.inference_mode():
            reference_logits = model(self.tokens)
            for k in (1, 4, 8, 12):
                changed = self.tokens.clone()
                changed[:, k:] = (changed[:, k:] % 127) + 1
                logits = model(changed)
                difference = (logits[:, :k] - reference_logits[:, :k]).abs().max().item()
                print(f"{name} k={k} max_abs_diff={difference:.9g}", flush=True)
                self.assertEqual(difference, 0.0,
                                 f"future tokens changed {name} logits before position {k}")

    def test_mechanism_and_constant_control_are_causal(self):
        self.assert_causal("mechanism", make_model(17, mechanism=True))
        self.assert_causal("control", make_model(17, mechanism=True, const=True))


if __name__ == "__main__":
    unittest.main()
