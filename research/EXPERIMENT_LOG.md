# EXPERIMENT LOG

Chronological record. Every entry states its commit, seed, configuration,
method, result, interpretation, and limitations. Retractions are recorded
as first-class entries, not deleted.

Two speedup measurements appear below and they are **not** the same
experiment; see E5.

---

## E1 — Baseline A (3 seeds)

```
Date:        2026-09-27
Git commit:  253ce9f (reproducible FineWeb-EDU baseline), reported in 8d09ecf
Seed:        42, 43, 44
Config:      unmodified nanochat path (no --rda)
Question:    What is the baseline's seed-to-seed spread?
Method:      Three independent runs, validation bits-per-byte.
Result:
    seed 42   1.301217
    seed 43   1.287544
    seed 44   1.295109
    mean      1.294623
    spread    0.0137   (max - min)
    std       ~0.0069
Interpretation:
    The spread is the yardstick. Any claimed improvement smaller than
    0.0137 cannot be distinguished from seed noise with three seeds.
Limitations:
    Single configuration. Does not establish that 0.0137 is the right
    yardstick for a different architecture.
```

---

## E2 — Initial RDA result — **RETRACTED**

```
Date:        2026-09-27
Git commit:  pre-bc3deb8 run; claim withdrawn in 8d09ecf
Seed:        42 only
Config:      B' (RDA) as it stood BEFORE the causality fixes
Question:    Does the persistent state improve on the baseline?
Method:      Single training run, validation bits-per-byte.
Result:
    1.286248   — reported at the time as an improvement over 1.294623
Interpretation (at the time):
    Taken as a ~0.008 improvement, i.e. a successful result.
RETRACTION (recorded 2026-09-28, commit 8d09ecf):
    Two independent reasons invalidate this.
    1. It is a SINGLE SEED. 1.286248 falls inside the baseline's own
       seed range (1.287544 - 1.301217). Once E1 was measured, the
       "improvement" turned out to be smaller than the baseline's own
       noise, so it is not distinguishable from a baseline draw.
    2. The run was LEAKY. It predated bc3deb8, which fixed a
       1022-token FiLM lookahead and a one-token PE lookahead. A leaky
       model is not a valid test of a causal architecture.
Status:
    NOT A RESULT. Recorded here so it is not rediscovered and
    re-reported as a success.
Limitations:
    Not re-run. Any future B' must use the post-bc3deb8 causal model.
```

---

## E3 — Causal leak fixes

```
Date:        2026-09-27
Git commit:  bc3deb8 "Fix two temporal leaks in RDA FiLM, and make the
             model decodable"
Seed:        n/a (structural)
Config:      RDA
Question:    Is the architecture actually causal?
Method:      Code inspection plus targeted perturbation tests.
Result:
    Leak 1 — FiLM. modulate() accepted a single (B, n_drives) state and
      broadcast it across all positions. Position 0 was therefore
      modulated by a state computed from the LAST token: a 1022-token
      lookahead at T=1024. It also made the model unreproducible
      autoregressively, since a decode step has no "final state".
      Fix: make the state per-position.
    Leak 2 — PE. E_next_seq[p] is built from pe_seq[p], which is the
      model's error for token p+1. Modulating position p with it let
      position p's hidden state — and thus the logits predicting token
      p+1 — depend on token p+1. A one-token lookahead that inflates val
      loss, because at eval time that token does not exist.
      Fix: shift the FiLM state by one, using E_next_seq[p-1].
Interpretation:
    E2's number was produced by a model that could see the future. The
    comparison in E2 was not merely underpowered; it was invalid.
Limitations:
    Structural reasoning, not a formal proof. Current tests cover the
    specific leak patterns known at the time; a new leak would not be
    caught automatically.
```

---

## E4 — Per-position causal FiLM documented

```
Date:        2026-09-27
Git commit:  b19d4c7
Seed:        n/a
Question:    Make the causality contract explicit in the repo.
Method:      Documentation; film_state = cat([E_start, E_next_seq], dim=1).
Result:      Position 0 <- E_start; position p <- E_next_seq[p-1].
Interpretation:
    Recording WHY the tempting alternative is wrong matters more than the
    fix itself; the wrong version is shorter and looks correct.
Limitations:
    Documentation only.
```

---

## E5 — Vectorized recurrence

```
Date:        2026-09-28
Git commit:  539007e
Seed:        n/a (mathematical equivalence) + perf measurement
Config:      RDA
Question:    Can the 1023-step Python loop be replaced without changing
             the trajectory?
Method:
    The drive at position t depends only on the base hidden state and PE
    at t, neither of which can depend on E. So the whole E sequence has a
    closed form:
        E[t] = w[t] * (decay*E_start + cumsum(g/w)[t]),  w[t] = decay^t
    float64 accumulation so that decay**-t stays finite.
Result:
    Equivalence: verified against the sequential reference loop. The
    first attempt was WRONG by 2.9e-02 — it omitted the `decay *` factor
    on E_start, which corresponds to seeding the recurrence with
    E_start/decay. Caught only because the test compares VALUES, not
    shapes.
    Speedup: TWO DISTINCT MEASUREMENTS, not to be merged:
      - ~153x  — from the original vectorization verification (E5, run
        during commit 539007e). NOT present in any committed file; it is
        a session measurement.
      - 222.8x — from diagnostics/verify_vectorized_update.py as it stands
        in the current suite (commit ab9b560), re-run 2026-09-28.
        vectorized 47.7 ms/pass vs loop 10629.0 ms/pass.
    Also added: a saturation-abort rule, --seed, and run manifests.
Interpretation:
    A ~200x reduction in state-recurrence cost makes 250-step diagnostics
    affordable, which is what made the saturation investigation possible
    at all. The speedup is a research-infrastructure win, not a result.
Limitations:
    Both speedup numbers are machine-specific (RTX 5060 Ti). The 153x
    figure is not recorded in any committed artifact and cannot be
    re-derived from the repo; treat it as historical.
```

---

## E6 — TREE check (run provenance)

```
Date:        2026-09-28
Git commit:  358993e (and 030a384)
Question:    Can a run be trusted to have executed the committed code?
Method:      A pre-run TREE check reports CLEAN or DIRTY; DIRTY means the
    working tree differed from HEAD.
Result:      Worked, and immediately earned its keep: it reported DIRTY
    during the saturation-abort test run, correctly, because uncommitted
    diagnostic work was present.
Interpretation:
    Without this, a diagnostic result could silently reflect uncommitted
    code. Given that the project later hit a shadowing bug of exactly
    this shape, the check was prescient.
Limitations:
    Reports dirtiness; does not say which file or which direction.
```

---

## E7 — Event saturation investigation

```
Date:        2026-09-28
Git commit:  investigation leading to ab9b560
Seed:        42
Config:      depth 8, d_model 768, T=1024, 30 steps, toy scale
Question:    Why did the event term grow, and was LayerNorm failing?
Method:
    14 signals recorded per step over 30 real steps, then five controls:
      C1 unfrozen reference
      C2 base model frozen
      C3 RMS-normalize event projection output
      C4 bound event via tanh(k*x)/k, k=0.3
      C5 decay 0.5
Result:
    LayerNorm output rms:      1.0000 -> 1.0000  (x1.00)   WORKING
    LayerNorm weight norm:    27.7128 -> 27.7128 (x1.00)   WORKING
    event_proj ||W||:          1.6317 -> 3.5092  (x2.15)   GREW
    event |max|:               0.2508 -> 1.2612  (x5.03)   GREW
    raw h rms:                 0.9999 -> 0.9999  (x1.00)
    grad||base||:              0.6657 -> 11.6698 (x17.53)
    Cauchy-Schwarz ceiling:   16.2898 -> 35.4817 (x2.18)
    Controls:
      C2 frozen base  -> IDENTICAL to C1: ||W|| 1.6317 -> 3.5092,
                         event -> 1.2612, matching to 4 decimals.
      C3 normalize    -> event 0.2715 -> 0.2690 (x0.99)  FLAT
      C4 bound 0.3    -> event -> 0.2983 (x1.45); ||W|| still x1.91
      C5 decay 0.5    -> event -> 1.3471 (x5.37)  WORSE
Interpretation:
    LayerNorm was never the failure — it pins ||h|| to a CEILING of
    sqrt(d) = 27.71, not to a value, and event_proj(h) can still reach
    ||w_row|| * 27.71 with ||w_row|| learned and unbounded.
    C2 is decisive: with h byte-identical every step and grad||base|| = 0
    exactly, the growth still happens. So the cause is the gradient on
    event_proj itself, and hidden-state drift is ruled out.
    C5 rules out "the decay is too slow" as an explanation.
Limitations:
    One seed, 30 steps, toy scale. C1-C5 were all measured through a probe
    that had a shadowing defect (see FAILED_EXPERIMENTS.md, F4), so their
    RELATIVE comparison is valid — all five shared the same path — but C3
    in particular was a prediction of the fix rather than an independent
    measurement of it. Superseded by E9.
```

---

## E8 — Event-output normalization (implemented)

```
Date:        2026-09-28
Git commit:  ab9b560
Seed:        42
Config:      depth 8, d_model 768, 250 steps, toy scale
Question:    Does normalizing the projection OUTPUT control the event term?
Method:      RMS-normalize event_proj output over the 8 drive axes, in both
    update() and drive_seq(). Applied identically to both so the
    reference loop and the vectorized path stay comparable.
Result:
    30 steps: event 0.2715 -> 0.2690 (x0.99); ||W|| -> 2.0126 (x1.23)
    250 steps: event 0.2715 -> 0.1670 (x0.62), range [0.162, 0.278] with
               no upward trend; ||W|| 1.6317 -> 0.3582 (x0.22, SHRINKS)
Interpretation:
    The mechanism is controlled, not delayed: ||W|| now shrinks, because
    a global scale on W became a no-op and only its direction is learned.
    The saturation observed at step 88 in this configuration was NOT
    fixed by delaying it — the amplifying degree of freedom is gone.
Limitations:
    A SECOND mechanism remained: the state still saturated, via a
    different route. See E10.
```

---

## E9 — PE/FiLM dependency graph mapping

```
Date:        2026-09-28
Git commit:  ab9b560 (diagnostics/map_pe_film_graph.py, pe_gradient_onset.py)
Seed:        42
Config:      d_model 128, T=1024
Question:    What is the actual gradient path, and when does it open?
Method:
    Real forward/backward on production rda.py. Observations made with
    forward/backward HOOKS on the production modules; the module methods
    are never replaced.
Result:
    PE == NLL(lm_head(hidden[:T-1]), targets[1:])   to 9.54e-07. CAUSAL.
    Perturbing targets[:,k] leaves E at all positions < k bit-identical
    (8.8e-05, i.e. float noise, zero in exact terms).
    fiLM scale/shift and BASE receive gradient at step 0;
    pe_proj, event_proj and h_norm receive EXACTLY 0.000e+00.
    At step 1 all of them become nonzero (pe_proj 4.9e-04).
    film weight norm at step 0: 0.000000 exactly.
Interpretation:
    dL/dE is proportional to the film weights, which are zero-initialized,
    so the state pathway is gradient-dormant for exactly one step. The
    loop is CLOSED but dormant, and reopens at step 1. This is why the
    runaway could not be diagnosed from any single step, and why
    "is the loop closed?" is not a yes/no question — it is a per-step
    question.
Limitations:
    The Q9 bisection at step 0 is uninformative by construction, because
    at step 0 the state gradient is identically zero. A later-step
    bisection would be needed to probe the closed loop directly.
```

---

## E10 — PE/FiLM ablations C1–C6

```
Date:        2026-09-28
Git commit:  ab9b560 (diagnostics/probe_pe_film.py)
Seed:        42
Config:      depth 8, d_model 768, 250 steps, base model TRAINABLE via
             m.base.setup_optimizer + add_param_group, 40-step warmup
Question:    Is the PE/FiLM runaway caused by PE projection drift, FiLM
             amplification, or a differentiable feedback loop?
Method:
    C1 reference (production code, no ablation)
    C2 FiLM disabled (state still computed and recurs)
    C3 base model frozen
    C4 pe_proj frozen
    C5 PE detached from its gradient, numerical value preserved
    C6 FiLM gain RMS-normalized to a fixed budget (hook, opt-in)
Result:
    control   1stCross  max|E|   peB final  FiLM gain  loss
    C1        217       0.9853   1.3900     2.227      5.805
    C2        never     0.9390   1.5149     1.000      6.098
    C3        82        0.9989   3.7130     9.568      9.920
    C4        217       0.9922   1.5149     2.283      5.795
    C5        217       0.9853   1.3896     2.238      5.793
    C6 (0.5)  never     0.9385   1.7620     1.529      5.801
    C6 (1.0)  never     0.9793   1.6941     2.444      5.742
Interpretation — which conclusions are actually supported:
    SUPPORTED: FiLM amplification is the cause. C2 never saturates, and
      PE still grows to 19.6 without FiLM, so PE growth is downstream.
    NOT SUPPORTED: PE projection drift. pe_proj.bias SHRINKS in C1
      (1.5324 -> 1.3900), and C4 (frozen) still saturates at step 217.
    NOT SUPPORTED: differentiable feedback pathology. C5 gives max|E|
      0.9853 and the same crossing step — indistinguishable from C1.
    DECISIVE EXTRA CONTROL (C3): with the base frozen, PE is FLAT at 10.50
      for 250 steps, yet pe_proj.bias grows 1.53 -> 3.71 and the FiLM gain
      grows 1.00 -> 9.57. A loop that runs with constant input is
      self-exciting. This is the strongest single piece of evidence.
    Growth ORDERING in C1: base grad norm first (step 11), then PE
      (step 61), then FiLM gain (step 206), then |E| crossing (step 217).
      pe_proj.bias NEVER doubles. The ordering is consistent with FiLM
      leading and PE following.
Limitations:
    One seed. The C6 sweep at 0.5 and 1.0 is two points, not a sweep —
    film_rms was chosen from these two, which is a judgement call.
    See FAILED_EXPERIMENTS.md F5: the first version of this probe omitted
    the base model from the optimizer, which made C3 identical to C1 BY
    CONSTRUCTION. All numbers above are from the corrected probe.
```

---

## E11 — Current stabilization, verified through production code

```
Date:        2026-09-28
Git commit:  ab9b560
Seed:        42
Config:      depth 8, d_model 768, 250 steps, toy scale
Question:    Does the fix, implemented in rda.py itself, control the
             runaway? (Not a hook, not a prediction — the real path.)
Method:      diagnostics/probe_pe_film.py with NO ablation flags, so the
    production modulate() and drive_seq() execute unmodified. The probe
    prints which implementation is active and asserts zero hooks remain.
Result:
                          before      after
    first |E| > 0.98      217         never
    max |E|               0.9853      0.9402
    final |E|             0.9025      0.9277
    FiLM gain max         2.227       1.522
    loss                  5.805       5.808
    direction cosine      —           1.000000000000
    identity at init      —           exact (max |diff| 0.00e+00)
Interpretation:
    The loss is UNCHANGED (5.808 vs 5.805). This is the check that matters
    most: the fix did not close off a loss shortcut, so the state pathway
    is still doing whatever work it was doing. The runaway is gone rather
    than postponed.
    The C6 hook prediction (never, max 0.9385, loss 5.801) matched the
    production result (0.9402, 5.808) — the ablation predicted the fix.
Limitations:
    ONE SEED. 250 STEPS — saturation began at 217, so this is only ~15%
    past onset, and cannot distinguish "controlled" from "delayed" at a
    longer horizon. TOY SCALE, not the real B' configuration.
    PE still grows to ~36 and remains unexplained.
```

---

## E12 — Regression tests

```
Date:        2026-09-28
Git commit:  ab9b560
Question:    Do the stability fixes preserve the properties the
             investigation established?
Method:      diagnostics/test_pe_film_invariants.py (new, 18 checks) plus
             the three pre-existing scripts.
Result:
    test_pe_film_invariants.py         18/18 PASS
    verify_vectorized_update.py        PASS (222.8x)
    verify_t1024.py                    PASS
    test_saturation_abort.py           PASS
Covered: PE alignment; no one-token lookahead; position-0 causality; FiLM
    gain bounded even at ||W|| x 1e4; exact identity at init; direction
    preservation; event-normalization regression guard; |E| <= 1; closed
    form == reference loop (3.33e-16).
Limitations:
    Three of these tests initially FAILED and the failures were bugs in
    the tests, not the code — see FAILED_EXPERIMENTS.md F6. Worth noting
    that "the test is wrong" is a conclusion that has to be earned each
    time, not assumed.
```

---

## NOT YET RUN

**E13 — 1000-step production confirmation.** THE GATE.

Saturation previously began at step 217. 250 steps cannot distinguish
"controlled" from "delayed". Required before the 3-seed A/B.

**E14 — 3-seed B' A/B comparison.** Blocked on E13.

Command for E13, when resuming:

```
cd C:\AI\nanochat
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
$env:PYTHONUNBUFFERED="1"; $env:PYTHONPATH="C:\AI\nanochat"
$env:NANOCHAT_DATASET="fineweb-edu"
.\.venv\Scripts\python.exe diagnostics\probe_pe_film.py --steps 1000 `
  --label "E13 1000-step production confirmation" --out logs\pe_1000.json
```

Report: first step |E|>0.98, max|E|, FiLM gain max, loss curve.
