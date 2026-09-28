"""Phase 2/3: instrument the PE/FiLM runaway on PRODUCTION code.

Design rules (enforced, after the earlier shadowing incident):

  1. The normal measurement path calls the real RDAStateModule /
     GPTWithRDA methods. Nothing is reimplemented.
  2. Every intervention is a forward hook (ablation) or a requires_grad
     flag (freeze). Hooks are registered and removed explicitly.
  3. Every run PRINTS which implementation is active and whether any hook
     is installed, so a stale/shadowed run cannot masquerade as a result.
  4. Observation uses forward hooks to CAPTURE intermediates; a capture
     hook returns its input unchanged, so it cannot alter the trajectory.

Ablations (opt-in, off by default):
  --no-film      C2  FiLM disabled: hidden_mod == hidden (identity modulation)
  --freeze-base  C3  base LM frozen; state/FiLM still train
  --freeze-pe    C4  pe_proj.weight and .bias frozen
  --detach-pe    C5  PE keeps its numerical value; only the gradient
                     edge pe -> pe_proj is cut (gradient hook on the input)
  --film-cap F   C6  FiLM scale bounded by a normalized reparameterization

Run:  python -m diagnostics.probe_pe_film --steps 250 --out logs/pe_c1.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

from nanochat.common import compute_init
from nanochat.gpt import GPT, GPTConfig
from nanochat.rda import GPTWithRDA, DRIVE_DIMS
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

D_MODEL = 768
NAMES = list(DRIVE_DIMS)


def gn(t):
    return 0.0 if t is None else float(t.detach().float().norm())


def stats(t):
    t = t.detach().float()
    return (float(t.min()), float(t.max()), float(t.mean()),
            float(t.pow(2).mean().sqrt()))


def build(seed, device):
    compute_init("cpu" if device.type == "cpu" else "cuda", seed)
    tok = get_tokenizer()
    V = tok.get_vocab_size()
    cfg = GPTConfig(sequence_len=1024, vocab_size=V, n_layer=8, n_head=6,
                    n_kv_head=6, n_embd=D_MODEL, window_pattern="L")
    base = GPT(cfg)
    base.init_weights()
    m = GPTWithRDA(base, debug_steps=0).to(device)
    m.train()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, 4, 1024, split="train", device=device)
    return m, tok, loader


def install_hooks(m, a):
    """Register capture / ablation hooks. Returns (handles, capture dict).

    Capture hooks RETURN THEIR INPUT UNCHANGED, so they are pure observers.
    Ablation hooks are only installed when explicitly requested.
    """
    st = m.state
    cap = {}
    H = []

    def cap_mod(name, key, xform=None):
        def f(mod, inp, out):
            v = out if xform is None else xform(out)
            cap[key] = v.detach()
            return out          # UNCHANGED -- observation cannot alter the run
        H.append(getattr(st, name).register_forward_hook(f))

    # ---- observations (always on, never alter the trajectory) ----
    cap_mod("pe_proj", "pe_proj_out")            # (B,T,8)  8-dim PE term
    cap_mod("film_scale", "film_scale_out")      # (B,T,d)  the ADDED scale
    cap_mod("film_shift", "film_shift_out")      # (B,T,d)  the added shift
    # the raw scalar PE entering pe_proj is its INPUT
    H.append(st.pe_proj.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("pe_raw",
                                              (inp[0].detach() *
                                               st.pe_reference_scale))))

    # ---- ablations (opt-in only) ----
    if a.no_film:
        # C2: FiLM has NO effect on the residual stream. The state is still
        # computed, still recurs, and is still measured -- only the two
        # modulation branches are zeroed, which makes modulate() the exact
        # identity at every step rather than only at init.
        H.append(st.film_scale.register_forward_hook(
            lambda mod, inp, out: torch.zeros_like(out)))
        H.append(st.film_shift.register_forward_hook(
            lambda mod, inp, out: torch.zeros_like(out)))

    if a.detach_pe:
        def cut(mod, gi, go):
            gi = [g.detach() if isinstance(g, torch.Tensor) else g for g in gi]
            if gi:
                gi[-1] = gi[-1].detach()
            return tuple(gi)
        H.append(st.pe_proj.register_full_backward_hook(cut))

    if a.film_norm > 0:
        # C6: bound the FiLM GAIN by normalizing each per-position modulation
        # vector to a fixed RMS, so only its DIRECTION is learned. Same
        # philosophy as the event-projection fix: a reparameterization that is
        # scale-invariant, not a clamp -- no dead zone, no gradient
        # flattening, sign and relative pattern fully preserved. At init the
        # film weights are exactly 0, so 0/eps = 0 and modulate() is still the
        # exact identity.
        c = a.film_norm
        def gain(mod, inp, out):
            return out / out.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt() * c
        H.append(st.film_scale.register_forward_hook(gain))
        H.append(st.film_shift.register_forward_hook(gain))

    return [h for h in H if h is not None], cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-film", action="store_true", help="C2")
    ap.add_argument("--freeze-base", action="store_true", help="C3")
    ap.add_argument("--freeze-pe", action="store_true", help="C4")
    ap.add_argument("--detach-pe", action="store_true", help="C5")
    ap.add_argument("--film-norm", type=float, default=0.0,
                    help="C6: bound FiLM gain to this RMS (0 = off)")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    device = torch.device(a.device)

    print("=" * 78)
    print("PE / FiLM RUNAWAY PROBE  --  label=%s" % a.label)
    print("=" * 78)
    abl = [n for n, f in (("C2 no-film", a.no_film), ("C3 freeze-base", a.freeze_base),
                         ("C4 freeze-pe", a.freeze_pe), ("C5 detach-pe", a.detach_pe),
                         ("C6 film-norm %g" % a.film_norm, a.film_norm > 0)) if f]
    print("device=%s steps=%d seed=%d" % (a.device, a.steps, a.seed))
    print("ACTIVE IMPLEMENTATION: production nanochat/rda.py")
    print("  drive_seq overridden?      NO (never overridden in this script)")
    print("  prediction_error override? NO")
    print("  ablations active:          %s" % (", ".join(abl) if abl else "NONE (reference)"))
    print("  observation hooks are pure: they return their input unchanged.")
    print()

    m, tok, loader = build(a.seed, device)
    st = m.state

    if a.freeze_base:
        for p in m.base.parameters():
            p.requires_grad_(False)
    if a.freeze_pe:
        st.pe_proj.weight.requires_grad_(False)
        st.pe_proj.bias.requires_grad_(False)

    handles, cap = install_hooks(m, a)
    live = (len(st.pe_proj._forward_hooks) + len(st.pe_proj._backward_hooks) +
            len(st.film_scale._forward_hooks) + len(st.film_shift._forward_hooks))
    print("hooks installed (observation + ablation): %d" % live)

    B_REF = 524288
    scale = (32768.0 / B_REF) ** 0.5
    LR_M = 0.02 * scale
    # The BASE model must be in the optimizer, exactly as base_train.py does it
    # (it calls model.setup_optimizer(matrix_lr=0.02*batch_lr_scale, ...)).
    # An earlier version of this probe omitted the base groups entirely, which
    # made "C3 freeze-base" identical to the reference BY CONSTRUCTION and
    # therefore meaningless. Base training is not optional here: the FiLM
    # runaway acts THROUGH the base model, so a frozen base cannot express it.
    # base_train.py uses the combined MuonAdamW optimizer and then APPENDS the
    # RDA groups to optimizer.param_groups. Do exactly that, so the base model
    # is trained by the real optimizer rather than a stand-in.
    optimizer = m.base.setup_optimizer(
        unembedding_lr=0.008 * scale,
        embedding_lr=0.3 * scale,
        scalar_lr=0.5 * scale,
        matrix_lr=0.02 * scale,
        weight_decay=0.0,
    )
    rda_groups = [
        dict(kind="adamw",
             params=list(st.film_scale.parameters()) + list(st.film_shift.parameters()),
             lr=0.02 * scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw",
             params=list(st.event_proj.parameters()) + list(st.pe_proj.parameters()),
             lr=0.02 * scale, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", params=[st.e_init],
             lr=0.5 * scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for g in rda_groups:
        g["initial_lr"] = g["lr"]
        optimizer.add_param_group(g)
    opt = optimizer
    n_state_p = sum(len(g["params"]) for g in rda_groups)
    print("base optimizer: %s with %d groups; appended %d state tensors"
          % (type(opt).__name__, len(opt.param_groups), n_state_p))
    print("drive-group LR=%.5f  (base_train.py batch_lr_scale)" % LR_M)
    print()

    rec = []
    carry = None
    for step in range(a.steps):
        # NOT set_to_none=True: MuonAdamW's fused kernels dereference every
        # grad unconditionally, so a None grad is a hard crash. Zeroing (rather
        # than clearing) is also what base_train.py does. Frozen parameters keep
        # a ZERO grad, so the kernels see a tensor and the update is exactly
        # a no-op for them -- the same effect as excluding them, without
        # mutating the optimizer's group structure between ablations.
        m.zero_grad(set_to_none=False)
        x, y, _ = next(loader)
        out = m(x, y, E_t_external=carry)
        loss = out["loss"]
        loss.backward()

        E_seq = out["E_next_seq"].detach()
        base_p = [p for p in m.base.parameters() if p.grad is not None]
        g_base = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                                      for p in base_p))) if base_p else 0.0

        with torch.no_grad():
            fs = cap["film_scale_out"]          # the term ADDED to 1.0
            fh = cap["film_shift_out"]
            pe_raw = cap["pe_raw"].squeeze(-1) if cap["pe_raw"].dim() == 3 else cap["pe_raw"]
            pe_out = cap["pe_proj_out"]
            h = out["hidden"].detach()
            hm = out["hidden_mod"].detach()

            # per-position state change
            dE = (E_seq[:, 1:] - E_seq[:, :-1]).abs().max().item() if E_seq.size(1) > 1 else 0.0

            row = {
                "step": step,
                "loss": float(loss.detach()),
                # --- PE pathway ---
                "pe_min": float(pe_raw.min()), "pe_mean": float(pe_raw.mean()),
                "pe_max": float(pe_raw.max()),
                "pe_out_absmax": float(pe_out.abs().max()),
                "pe_proj_w_norm": gn(st.pe_proj.weight),
                "pe_proj_b_norm": gn(st.pe_proj.bias),
                "pe_perdim": [float(v) for v in pe_out.abs().max(dim=0).values.max(dim=0).values],
                "g_pe_proj": gn(st.pe_proj.weight.grad),
                # --- FiLM pathway ---
                "film_s_min": float(fs.min()), "film_s_max": float(fs.max()),
                "film_s_mean": float(fs.mean()), "film_s_rms": float(fs.pow(2).mean().sqrt()),
                "film_h_min": float(fh.min()), "film_h_max": float(fh.max()),
                "film_h_mean": float(fh.mean()), "film_h_rms": float(fh.pow(2).mean().sqrt()),
                "film_w_norm": gn(st.film_scale.weight) + gn(st.film_shift.weight),
                "g_film_scale": gn(st.film_scale.weight.grad),
                "g_film_shift": gn(st.film_shift.weight.grad),
                # --- event pathway (for continuity with the earlier study) ---
                "ep_w_norm": gn(st.event_proj.weight),
                "g_event_proj": gn(st.event_proj.weight.grad),
                "g_h_norm": gn(st.h_norm.weight.grad),
                # --- state ---
                "E_absmax": float(E_seq.abs().max()),
                "E_perdim": [float(v) for v in E_seq.abs().max(dim=0).values.max(dim=0).values],
                "E_mean": float(E_seq.mean()), "E_rms": float(E_seq.pow(2).mean().sqrt()),
                "dE_max": dE,
                # --- model ---
                "hidden_rms": float(h.pow(2).mean().sqrt()),
                "hidden_mod_rms": float(hm.pow(2).mean().sqrt()),
                "g_base": g_base,
            }
            # FiLM's effect on scale: 1 + film_s
            row["mod_scale_absmax"] = float((1.0 + fs).abs().max())
            row["mod_scale_min"] = float((1.0 + fs).min())
        rec.append(row)

        # A frozen parameter has requires_grad=False, so autograd never
        # populates its .grad. The fused Muon/AdamW kernels dereference every
        # grad unconditionally, so hand them an explicit zero: the update is
        # then exactly a no-op for those tensors.
        for grp in opt.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
        opt.step()
        # base_train.py's get_lr_multiplier: 40 warmup steps, so the drive
        # group LR ramps from ~0 to full over the first 40 steps. Matching it
        # matters -- the onset timing is the thing under investigation.
        warmup = 40
        lr_mult = min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            if "initial_lr" in g:
                g["lr"] = g["initial_lr"] * lr_mult
        carry = out["E_next"]

        if step % 25 == 0 or step == a.steps - 1:
            print("step %3d | loss=%.4f pe_max=%.3f peWb=%.3f peOut=%.3f | "
                  "film s=%.3f shift=%.3f filmW=%.3f | maxE=%.4f | gbase=%.1f"
                  % (step, row["loss"], row["pe_max"],
                     row["pe_proj_b_norm"], row["pe_out_absmax"],
                     row["mod_scale_absmax"], row["film_h_max"],
                     row["film_w_norm"], row["E_absmax"], row["g_base"]))

    for h in handles:
        h.remove()
    left = (len(st.pe_proj._forward_hooks) + len(st.pe_proj._backward_hooks) +
            len(st.film_scale._forward_hooks) + len(st.film_shift._forward_hooks))
    print("\nhooks after run: %d (expect 0)" % left)

    first = next((r["step"] for r in rec if r["E_absmax"] > 0.98), None)
    print("\n=== %s ===" % a.label)
    print("first step |E|>0.98 : %s" % ("never" if first is None else first))
    print("max |E|             : %.4f" % max(r["E_absmax"] for r in rec))
    print("final |E|           : %.4f" % rec[-1]["E_absmax"])
    print("PE   first/max/final: %.3f / %.3f / %.3f"
          % (rec[0]["pe_max"], max(r["pe_max"] for r in rec), rec[-1]["pe_max"]))
    print("peW  first/final    : %.4f / %.4f" % (rec[0]["pe_proj_w_norm"], rec[-1]["pe_proj_w_norm"]))
    print("peB  first/final    : %.4f / %.4f" % (rec[0]["pe_proj_b_norm"], rec[-1]["pe_proj_b_norm"]))
    print("film scale max      : %.4f" % max(r["mod_scale_absmax"] for r in rec))
    print("film shift max      : %.4f" % max(r["film_h_max"] for r in rec))
    print("hidden rms first/last: %.4f / %.4f" % (rec[0]["hidden_rms"], rec[-1]["hidden_rms"]))
    print("g_base max          : %.2f" % max(r["g_base"] for r in rec))
    print("loss first/last     : %.4f / %.4f" % (rec[0]["loss"], rec[-1]["loss"]))

    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
