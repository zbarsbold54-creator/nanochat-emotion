import unittest

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import MuonAdamW
from nanochat.rda import GPTWithRDA


def make_optimizer(model, rda, const=False):
    opt = model.setup_optimizer(
        unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
        weight_decay=0.0, scalar_lr=0.5,
    )
    state_matrix = list(rda.state.film_scale.parameters()) + list(rda.state.film_shift.parameters())
    drive_matrix = (list(rda.state.event_proj.parameters())
                    + list(rda.state.pe_proj.parameters())
                    + list(rda.state.h_norm.parameters())
                    + list(rda.content_proj.parameters())
                    + list(rda.gate_proj.parameters()))
    state_vector = [rda.state.e_init]
    if rda.const_state_vec is not None:
        state_vector.append(rda.const_state_vec)
    predictor_matrix = list(rda.predictor.parameters())
    groups = [
        dict(kind='adamw', params=state_matrix, lr=0.02, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=drive_matrix, lr=0.02, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=state_vector, lr=0.5, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=predictor_matrix, lr=0.02, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for group in groups:
        group['initial_lr'] = group['lr']
        opt.add_param_group(group)
    grad_free = {id(rda.state.e_init)} | {id(p) for p in rda.predictor.parameters()}
    grad_free |= {id(p) for p in rda.state.event_proj.parameters()}
    grad_free |= {id(p) for p in rda.state.pe_proj.parameters()}
    grad_free |= {id(p) for p in rda.state.h_norm.parameters()}
    if const:
        grad_free |= {id(p) for p in rda.content_proj.parameters()}
        grad_free |= {id(p) for p in rda.gate_proj.parameters()}
    for group in opt.param_groups:
        if any(id(p) in grad_free for p in group['params']):
            group['grad_free'] = True
    return opt, grad_free


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for this optimizer regression')
class GatedOptimizerCUDA(unittest.TestCase):
    def test_m_and_const_one_step(self):
        device = torch.device('cuda')
        cfg = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                        n_kv_head=2, n_embd=64, window_pattern='L')
        x = torch.randint(1, 128, (2, 16), device=device)
        y = torch.randint(1, 128, (2, 16), device=device)
        for const in (False, True):
            torch.manual_seed(7)
            base = GPT(cfg).to(device)
            base.init_weights()
            model = GPTWithRDA(base, debug_steps=0, check_leakage=False,
                               gated_write=True, const_state=const,
                               const_state_seed=7).to(device)
            opt, grad_free = make_optimizer(base, model, const)
            out = model(x, y, E_t_external=None)
            out['loss'].backward()
            for group in opt.param_groups:
                if group.get('grad_free'):
                    for param in group['params']:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
            opt.step()
            self.assertTrue(all(torch.isfinite(p).all().item() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
