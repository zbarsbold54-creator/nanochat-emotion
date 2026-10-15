"""CPU regression tests for the opt-in gated-write mechanism."""

import unittest
import time

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, RDAStateModule


def make_model(seed=7, gated=True, const=False):
    torch.manual_seed(seed)
    cfg = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2,
                    n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    return GPTWithRDA(base, debug_steps=0, check_leakage=False,
                      gated_write=gated, const_state=const,
                      const_state_seed=seed)


class GatedWriteCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        self.x = torch.randint(1, 128, (2, 16))
        self.y = torch.randint(1, 128, (2, 16))

    def test_flag_off_matches_reference_for_20_steps(self):
        a = make_model(11, gated=False)
        b = make_model(11, gated=False)
        b.load_state_dict(a.state_dict())
        ca = cb = None
        for _ in range(20):
            oa = a(self.x, self.y, E_t_external=ca)
            ob = b(self.x, self.y, E_t_external=cb)
            self.assertEqual(float((oa["loss"] - ob["loss"]).abs()), 0.0)
            self.assertEqual(float((oa["E_next"] - ob["E_next"]).abs().max()), 0.0)
            ca, cb = oa["E_next"].detach(), ob["E_next"].detach()

    def test_sequence_equals_token_by_token(self):
        m = make_model(13, gated=True)
        h = torch.randn(2, 15, 64)
        content, gate = m._gated_terms(h)
        start = torch.randn(2, 8).clamp(-1, 1)
        seq = m.state.gated_update_seq(start, content, gate)
        e = start
        out = []
        for t in range(h.size(1)):
            e = m.state.gated_update(e, content[:, t], gate[:, t])
            out.append(e)
        token = torch.stack(out, dim=1)
        diff = float((seq - token).abs().max())
        print(f"sequence_token_max_abs_diff={diff:.9g}", flush=True)
        self.assertLessEqual(diff, 1e-6)

    def _assert_causal(self, name, model):
        model.eval()
        with torch.inference_mode():
            ref = model(self.x)
            for k in (1, 4, 8, 12):
                changed = self.x.clone()
                changed[:, k:] = (changed[:, k:] % 127) + 1
                got = model(changed)
                diff = float((got[:, :k] - ref[:, :k]).abs().max())
                print(f"{name} k={k} max_abs_diff={diff:.9g}", flush=True)
                self.assertEqual(diff, 0.0)

    def test_causality_mechanism_and_control(self):
        self._assert_causal("mechanism", make_model(17, gated=True))
        self._assert_causal("control", make_model(17, gated=True, const=True))

    def test_initial_gates_and_retention(self):
        m = make_model(19, gated=True)
        h = torch.randn(256, 64)
        with torch.no_grad():
            _, g = m._gated_terms(h)
        means = g.mean(0)
        print("initial_g_mean=" + " ".join(f"{v:.8f}" for v in means), flush=True)
        for power in (192, 832):
            retention = (1.0 - means).pow(power)
            print(f"retention_{power}=" + " ".join(f"{v:.8e}" for v in retention), flush=True)
        self.assertTrue(torch.all((means > 0) & (means < 1)))

    def test_control_optimizer_and_noncontrol_grads(self):
        control = make_model(23, gated=True, const=True)
        out = control(self.x, self.y)
        out["loss"].backward()
        self.assertTrue(all(p.grad is None for p in control.content_proj.parameters()))
        self.assertTrue(all(p.grad is None for p in control.gate_proj.parameters()))
        opt = torch.optim.Adam(control.parameters(), lr=1e-3)
        for p in control.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        opt.step()
        live = make_model(23, gated=True)
        with torch.no_grad():
            live.state.film_scale.weight.normal_(0.0, 0.1)
            live.state.film_shift.weight.normal_(0.0, 0.1)
        live(self.x, self.y)["loss"].backward()
        cg = sum(float(p.grad.abs().sum()) for p in live.content_proj.parameters())
        gg = sum(float(p.grad.abs().sum()) for p in live.gate_proj.parameters())
        print(f"noncontrol_content_grad_sum={cg:.9e} gate_grad_sum={gg:.9e}", flush=True)
        self.assertTrue(torch.isfinite(torch.tensor(cg + gg)))
        self.assertGreater(cg, 0.0)
        self.assertGreater(gg, 0.0)

    def test_loop_cost_and_initial_content_scale(self):
        m = make_model(29, gated=True)
        h = torch.randn(2, 15, 64)
        content, gate = m._gated_terms(h)
        print(f"content_std={content.std().item():.8f} event_term_std={torch.tanh(content).std().item():.8f}", flush=True)
        self.assertTrue(0.35 < float(content.std()) < 0.65)
        start = torch.zeros(2, 8)
        old_drive = m.state.drive_seq(h, torch.zeros(2, 15))
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            m.state.update_seq(start, old_drive)
        old_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            m.state.gated_update_seq(start, content, gate)
        gated_s = time.perf_counter() - t0
        pct = 100.0 * (gated_s / old_s - 1.0)
        print(f"gated_loop_cost_vs_existing_percent={pct:.3f}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
