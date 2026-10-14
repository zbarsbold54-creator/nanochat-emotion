import sys
import unittest

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA


def make_model(seed, mechanism=False, const=False):
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2,
                    n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    return GPTWithRDA(base, debug_steps=0, check_leakage=False,
                      mechanism=mechanism, const_state=const,
                      const_state_seed=seed)


class MechanismCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = torch.randint(1, 128, (2, 16))
        self.y = torch.randint(1, 128, (2, 16))

    def test_mechanism_trains_and_changes_outputs(self):
        model = make_model(7, mechanism=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with torch.no_grad():
            before = model(self.x, E_t_external=None).clone()
        carry = None
        for _ in range(4):
            out = model(self.x, self.y, E_t_external=carry)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            carry = out["E_next"].detach()
        with torch.no_grad():
            delta = (model(self.x, E_t_external=carry) - before).abs().max()
        self.assertGreater(float(delta), 0.0)

    def test_constant_control_runs_with_matching_content_capacity(self):
        model = make_model(7, mechanism=True, const=True)
        self.assertIsNotNone(model.content_proj)
        self.assertIsNotNone(model.const_state_vec)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        carry = None
        for _ in range(3):
            out = model(self.x, self.y, E_t_external=carry)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            for p in model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            opt.step(); opt.zero_grad(set_to_none=True)
            carry = out["E_next"].detach()


if __name__ == "__main__":
    unittest.main()
