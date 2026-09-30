# HANDOVER

## Established
- **E1** baseline 3 seeds 1.301217 / 1.287544 / 1.295109, spread **0.0137**.
- **E3–E5** fixed a 1022-token FiLM and a one-token PE lookahead; per-position `film_state`. Vectorized recurrence 3.33e-16, 222.8x.
- **E9–E11** the runaway is **FiLM self-excitation** (frozen-base control: PE flat at 10.50, FiLM gain 1.00→9.57). Fix holds |E|≤0.94 with **loss unchanged** (5.808 vs 5.805).
- **E14** `pe_seq` read `targets[:,1:]` so logits[p] saw x[p+1]. Fixed 969e067; the **0.0749 BPB gain is withdrawn**.
- **E13-fixed-3000** threat: `pe_proj.bias` = **-1.711 = 80%** of arg (p50 -2.11, residual 4.6e-07); p50 crosses -2 at 1131, never -3. Drive **unimodal at -0.972** (mode 58-64%, f>0.9 = 0.0). State on fixed point E*=d* (diff 3e-05). Applied gain 1.0→3.59, still climbing. |E_mean|/E_std = **436 threat**, **140 novelty**. Reconstruction residual is **TF32** (`precision='high'`): replay bit-exact 0.000e+00.

## Retractions
E2 (single seed + leaky); E13 1000-step (carried the leak); G2/G3 (wrong FiLM gain column, twice); "positive tail" (twice — absmax is |min|); "arousal/lp/homeostatic mirror threat" (they sit near centre); "threat 6th/least offset" (`.std()` with no `dim` over (768,B,T)); reconstruction gate FAIL 3.2e-04 (tolerance finer than TF32); valley test inverted (false BIMODAL); "tanh deriv never <0.05" (crosses 1743 on a sibling run); "shuffle y" and "PE OFF via targets=None".

## Open
- **E16**: PE plateaus ~33.8 then falls to 18.8 — unexplained.
- No valid B' A/B since the leak fix.
- Runs **not bit-reproducible** (5 runs: crossing 1177-1353, peak 0.9848-0.9893).
- Criterion-1 replacement (R_d/M_d/D_d/signed rails) **not implemented**; thresholds uncalibrated.
- Health-flag defect: |bias|/|arg_p50| undefined when |arg|<0.2.

## Next planned experiment — NOT LAUNCHED
End-of-run ablation: mean-ablate and zero-ablate each of the 8 E dimensions, with a baseline-repeat noise floor, to test whether the loss uses each dimension's signal.

## Parked until E13 is stable
Vision and 2D world simulation.

## Uncommitted
`logs/**` (excluded). `base_train.py` unchanged.
