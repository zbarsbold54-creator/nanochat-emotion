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

## E13 — 1000-step production confirmation — **PASS**

```
Date:        2026-09-29
Git commit:  6977c60 (code under test; no code change in this entry)
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Question:    Is the PE/FiLM saturation mechanism CONTROLLED rather than
             merely DELAYED?
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Console output to logs\pe_1000.txt,
             per-step records to logs\pe_1000.json.
Result:
    max |E| over all 1000 steps      0.9401  at step 254
    steps with |E| > 0.98            NONE
    FiLM gain, steps 250-999         mean 1.3617, stdev 0.0386
                                     min 1.2820, max 1.5781
    FiLM gain slope (250-999)        -0.000033 / step  (-0.0033 per 100)
    FiLM gain  0 -> 1.0000
               250 -> 1.4290
               500 -> 1.3561
               750 -> 1.3562
               999 -> 1.4001
    PE max    0 -> 10.491
              250 -> 36.858
              500 -> 38.733
              750 -> 38.861
              999 -> 38.862        (PLATEAUS near 38.9 after ~step 250)
    loss     10.3983 -> 4.7816
    post-peak decay, max |E| per 100-step window:
        250-349  0.9401     350-449  0.9397     450-549  0.9297
        550-649  0.8977     650-749  0.8511     750-849  0.8143
        850-949  0.7575     950-999  0.6724
    windows exceeding the previous window's max   NONE
Interpretation:
    PASS on both pre-registered criteria.
    1. |E| <= 0.98 for all 1000 steps, with 0.040 of margin.
    2. FiLM gain is stationary around 1.36 and the slope is NEGATIVE, so
       the specific failure mode the criterion was written to catch --
       gain drifting upward while |E| stays under threshold -- is absent.

    On "controlled vs delayed": in the unstabilized configuration |E|
    crossed 0.98 at step 217 and was rising. Here |E| PEAKS at step 254 and
    then decays monotonically in every 100-step window, to 0.6724 by step
    999 -- a factor of 1.4 below its own peak while training continues. A
    delayed ratchet can only arrive later; it cannot fall. Combined with
    the flat-to-negative gain slope, the mechanism is controlled.

    Incidental: PE plateaus near 38.9 after ~step 250 rather than growing
    without bound. This partially answers the E16 anomaly -- the growth
    self-limits -- but does not explain it.
Limitations:
    SINGLE SEED (42). TOY SCALE (depth 8, d_model 768), not the real B'
    configuration. 1000 steps is still far shorter than a production
    training run, so this is evidence, not proof. The stronger control is
    the saturation abort enabled on the real training job.
    The 375 raw step-to-step increases after the peak are ordinary
    sampling noise in max|E|; the windowed test is the meaningful one.
```

---

## E13-SUPERSEDED — 1000-step production confirmation — **measured WITH A LABEL LEAK**

> **SUPERSEDED BY E13-fixed. Do not cite the numbers below as evidence of
> stability.** This entry is retained verbatim as history, not deleted.
>
> **What the leak was.** `rda.py` built the per-position prediction error
> from `targets[:, 1:]`. Since `pe_seq` is indexed by input position and
> `E_next_seq[j]` is the state after consuming position `j`, that made
> `pe_seq[j]` a function of `y[j+1] = x[j+2]`. Because
> `film_state[p] = E_next_seq[p-1]` and
> `logits[p] = lm_head(modulate(hidden[p], film_state[p]))`, the drive
> state at position `p` had already seen `x[p+1]` — the very token
> `logits[p]` is scored against, since the dataloader makes
> `y[p] == x[p+1]` and the loss is `cross_entropy(logits[p], targets[p])`
> with no shift. One token of the answer reached the model.
>
> **Fixed in `969e067`** by indexing `pe_seq` as `targets[:, :-1]`.
> `diagnostics/test_no_label_leak.py` fails with 12 violations on
> `6ae93ce` and passes with 0 after the fix; `scripts/base_train.py` now
> runs it as a hard pre-flight gate.
>
> **Commits whose B′ results are affected:** every B′ run from `ab9b560`
> through `6ae93ce`, including the B′-s42 canary and this E13 entry.
> Probe evidence is committed under `diagnostics/audit_label_leak/`.
>
> **What survives and what does not.** The saturation *mechanism* findings
> (event-output normalization, FiLM self-excitation) were diagnosed on
> probe runs that also carried the leak, so they are not independent
> measurements of the fixed architecture. The specific figures below are
> superseded by E13-fixed. The baseline A numbers (1.301217 etc.) are
> unaffected: those runs never passed `--rda`.

```
Date:        2026-09-29
Git commit:  6977c60 (code under test; no code change in this entry)
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Question:    Is the PE/FiLM saturation mechanism CONTROLLED rather than
             merely DELAYED?
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Console output to logs\pe_1000.txt,
             per-step records to logs\pe_1000.json.
Result:
    max |E| over all 1000 steps      0.9401  at step 254
    steps with |E| > 0.98            NONE
    FiLM gain, steps 250-999         mean 1.3617, stdev 0.0386
                                     min 1.2820, max 1.5781
    FiLM gain slope (250-999)        -0.000033 / step  (-0.0033 per 100)
    FiLM gain  0 -> 1.0000
               250 -> 1.4290
               500 -> 1.3561
               750 -> 1.3562
               999 -> 1.4001
    PE max    0 -> 10.491
              250 -> 36.858
              500 -> 38.733
              750 -> 38.861
              999 -> 38.862        (PLATEAUS near 38.9 after ~step 250)
    loss     10.3983 -> 4.7816
    post-peak decay, max |E| per 100-step window:
        250-349  0.9401     350-449  0.9397     450-549  0.9297
        550-649  0.8977     650-749  0.8511     750-849  0.8143
        850-949  0.7575     950-999  0.6724
    windows exceeding the previous window's max   NONE
Interpretation:
    PASS on both pre-registered criteria.
    1. |E| <= 0.98 for all 1000 steps, with 0.040 of margin.
    2. FiLM gain is stationary around 1.36 and the slope is NEGATIVE, so
       the specific failure mode the criterion was written to catch --
       gain drifting upward while |E| stays under threshold -- is absent.

    On "controlled vs delayed": in the unstabilized configuration |E|
    crossed 0.98 at step 217 and was rising. Here |E| PEAKS at step 254 and
    then decays monotonically in every 100-step window, to 0.6724 by step
    999 -- a factor of 1.4 below its own peak while training continues. A
    delayed ratchet can only arrive later; it cannot fall. Combined with
    the flat-to-negative gain slope, the mechanism is controlled.

    Incidental: PE plateaus near 38.9 after ~step 250 rather than growing
    without bound. This partially answers the E16 anomaly -- the growth
    self-limits -- but does not explain it.
Limitations:
    SINGLE SEED (42). TOY SCALE (depth 8, d_model 768), not the real B'
    configuration. 1000 steps is still far shorter than a production
    training run, so this is evidence, not proof. The stronger control is
    the saturation abort enabled on the real training job.
    The 375 raw step-to-step increases after the peak are ordinary
    sampling noise in max|E|; the windowed test is the meaningful one.
    >>> SUPERSEDED: measured with the label leak described above. The
        "gain slope NEGATIVE" and "|E| decays monotonically" conclusions
        do NOT reproduce on the fixed code -- see E13-fixed, where the
        applied gain slope is +5.72e-04/step and max|E| RISES from
        window 500 onward. Do not carry this entry's conclusion forward.
```

---

## E14 — label-leak fix and audit — **the 0.0749 BPB gain is WITHDRAWN**

```
Date:        2026-09-29
Fix commit:  969e067  "Fix a one-token label leak in the RDA drive state"
Question:    Is the B'-s42 result (1.226305 vs baseline 1.301217) real?
Method:      Read-only audit of the saved checkpoints plus a permanent
             regression test. No training. Scripts and raw output under
             diagnostics/audit_label_leak/.
Result:
    LEAK CONFIRMED. The drive state could see the label it was scored on.
    WITHDRAWN: the 0.0749 BPB gain. It measured the leak, not the state.

    Verified indexing facts:
        y[p] == x[p+1]                                    fraction 1.000000
        gpt.py:532  cross_entropy(logits[p], targets[p])  no shift
        rda.py:901  pe_seq[j] from targets[:, 1:]  ->  y[j+1] = x[j+2]
        rda.py:920  E_next_seq[j] depends on x[j+2]
        rda.py:972  film_state[p] = E_next_seq[p-1] -> x[p+1]
        rda.py:1036 logits[p] = lm_head(modulate(hidden[p], film_state[p]))
        => logits[p] depends on the token it is graded against.

    Dynamic proof: perturb y[:,k] alone with x fixed. hidden and logits at
    positions < k are EXACTLY 0 difference; positions >= k all move (the
    recurrence is cumulative). The violation is the diagonal.

    Why the earlier causality tests missed it: they perturbed x, never y.
    Perturbing x can only ever find a forward-token leak, and this leak
    runs through the LABELS. Two existing tests
    (verify_t1024.py, verify_vectorized_update.py) had controls asserting
    that position p DOES change when y[:,p] changes -- i.e. they were
    written to pass with the leak present. Both corrected in 969e067.

    Ablations on the leaky checkpoint (24 identical val batches):
        baseline a-s42  TRUE y  1.242340
        B'-s42          TRUE y  1.137101
        B'-s42  event term zeroed   1.596395
        B'-s42  FiLM forced const   2.990616
        B'-s42  PE term zeroed      4.371596
      Every ablation lands FAR WORSE than the 1.242340 baseline, which is
      the signature of a model leaning on the leaked signal rather than
      one that has genuinely learned something.
Interpretation:
    The configuration was never wrong and the baseline was never
    contaminated: a-s42 and b-stab-s42 differ only by --rda and
    --model-tag (flag-by-flag diff: 0 differences in the shared set), and
    the dataset, shard count and shard-00000 edge hash all match. The
    invalidity is in the METRIC, not the config: the evaluation was not
    label-blind, so A and B were not measured on the same task.

    Two methods used during this audit were themselves wrong and are
    retracted here rather than quietly dropped:
      - "shuffle y and re-score" as a label-blindness test. The BASELINE
        also moves (+1.748) under it, because scoring against wrong
        answers is simply a harder task. It confounds cheating with
        difficulty. Not evidence either way.
      - "PE OFF via targets=None" (4.29). Invalid counterfactual: it puts
        the TRAINED FiLM into a state distribution it never saw.
Limitations:
    Fixing the index changes what the model is asked to do, so every B'
    number must be re-measured. The cheapest valid rerun is B' only
    (600 steps, ~34 min); the A baseline is unaffected.
    E16 (the PE plateau anomaly) is NOT resolved by this and remains open.
```

---

## E13-fixed — 1000-step production confirmation on FIXED code — **FAIL**

```
Date:        2026-09-29
Fix commit:  969e067
Seed:        42
Config:      depth 8, d_model 768, T=1024, TOY SCALE, 1000 steps
Method:      diagnostics/probe_pe_film.py, 1000 steps, production rda.py
             with no ablation flags. Harness byte-identical to E13
             (`git diff 6ae93ce 969e067 -- diagnostics/probe_pe_film.py`
             is 0 lines), so the two runs are directly comparable.
Output:      logs\pe_1000_fixed.json, logs\pe_1000_fixed.txt

VERDICT: FAIL. G1, G2 and G4 pass; G3 fails.

  G1  max|E| <= 0.98 ................................. PASS
      peak 0.9690 at step 921, 0 steps over bound, margin 0.0110
  G2  applied gain in band [1.0, 2.0] (mid 1.5) ..... PASS
      mean 1.8821, median 1.8870, min 1.6190, max 2.0760
  G3  applied gain shows no upward trend ............. FAIL
      least-squares slope +5.7156e-04 per step (tol 1e-04)
      3 of 4 windows rising (starts 450, 650, 850)
  G4  loss finite and decreasing .................... PASS
      10.3983 -> 5.1722

Applied gain, 'film s' = max|1 + _unit_rms(film_scale(E), 0.5)|:
      step    0 : 1.0000
      step  250 : 1.6190
      step  500 : 1.7620
      step  750 : 2.0680
      step  999 : 2.0620

    WHAT "film s" IS, PRECISELY. It is
        max over (batch, position, channel) of |1 + _unit_rms(fs, 0.5)|
    where fs is the captured output of the film_scale module. This is a
    MAXIMUM over a (B, T, d_model) tensor, NOT an RMS and NOT a mean. The
    per-position RMS of the applied term is pinned at exactly 0.5 by
    construction (that is what _unit_rms does), so the only thing that
    varies across steps is the per-channel SPREAD; the printed number is
    the extreme channel. That is why it reaches 2.07 while rms(g) is 0.5.
    It is NOT comparable to the JSON column `film_s_rms`, which is
    max-independent: film_s_rms is the RMS of the RAW projection, taken
    BEFORE _unit_rms.

    HARNESS NOTE. An earlier pass scored G2/G3 against the JSON
    `film_s_rms` column and reported a mean of 0.4472 and a near-zero
    range. That was the wrong column (pre-_unit_rms raw projection, not
    the applied gain) and the resulting FAIL on G2 was wrong. G2 passes on
    the correct series. G1 and G3 were unaffected and stand as reported.

    SEPARATE CONCERN, NOT PART OF THE ORIGINAL PASS RULE. max|E| is
    RISING on the fixed code: 0.9205 / 0.8890 / 0.9133 / 0.9459 / 0.9594
    / 0.9642 / 0.9690 across 100-step windows 300-399 through 900-999,
    i.e. five consecutive rising windows from 500 onward, ending 0.0110
    from the 0.98 abort bound. The original criterion was only
    "|E| <= 0.98 for all steps", which this satisfies, so G1 is a PASS.
    This windowed-trend concern was ADDED AFTER seeing the fixed run and
    is NOT a pre-registered rule; it is reported as an observation, not
    as a gate. It is nonetheless the thing that most warrants follow-up,
    because it is the same self-excitation signature the criterion was
    written to catch, and it did not appear in E13 (which was leaky).

    PE max 33.1564 at step 430, final 20.138 -- LOWER than E13's 38.862.
    So PE growth is not leak-driven: the fixed code plateaus earlier and
    lower. E16 remains open either way.
Interpretation:
    The gate fails on trend, not on level. The gain is inside the
    operational band and the loss converges normally, but the gain is
    climbing at 5.7e-04/step and the state is drifting up toward the
    abort bound. E13 (leaky) showed a NEGATIVE gain slope and a max|E|
    that decayed; the fixed code shows the opposite in both. Whether that
    is the leak's removal changing the optimisation, or run-to-run
    variation at a single seed, is not established by one seed.
Limitations:
    SINGLE SEED (42), same as E13, so the two runs share a seed and differ
    only in the one-index fix. TOY SCALE. 1000 steps.
    The G2 band [1.0, 2.0] is an OPERATIONALIZATION of "gain near 1.5"
    chosen for this re-evaluation, not original wording; a different band
    would change the G2 verdict.
    No B'-s42 rerun was performed: the user's protocol gates it on this
    result, and this result is FAIL.
```

---

## NOT YET RUN

**E14 — 3-seed B' A/B comparison.** GATE MET by E13, but not launched.
Awaiting explicit go-ahead. Do not launch seeds 43/44 without instruction.

