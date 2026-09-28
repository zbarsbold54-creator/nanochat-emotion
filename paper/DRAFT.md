# RDA: A Persistent Low-Dimensional Internal State for a Small Language Model

*Working title. Not final.*

## Abstract

[Placeholder — do not fabricate results.]

The primary hypothesis has not been tested. The one earlier run that
appeared to support it was a single seed whose value fell inside the
baseline's own multi-seed range, and it was produced by a model with two
temporal leaks. See `../research/FAILED_EXPERIMENTS.md` F1 and F2.

What this project can currently report is a set of negative and diagnostic
results: two distinct saturation mechanisms, their causal isolation, and
the reparameterizations that removed them. See Section 8.

---

## 1. Introduction

A transformer carries context in its KV cache, which is bounded by the
context window and is overwritten token by token. An alternative is to
carry a small, slowly-decaying summary across the whole sequence. This
project asks whether such a state is useful, and if so what it encodes.

The interesting difficulty is that a state which is genuinely connected to
the loss can become a liability: it can find configurations that reduce the
training loss locally while destroying the very dynamics the state was
supposed to track. This project encountered that failure twice, in two
different forms, and the process of diagnosing it is a substantial part of
what has been learned.

Contributions claimed here, at this stage:

- A causal architecture for a persistent 8-dimensional affective state
  coupled back into a language model, with the indexing contract stated
  and tested explicitly.
- Two diagnosed failure mechanisms in that architecture, each isolated by
  ablation rather than by plausibility, with a principled stabilization for
  each.
- A measurement discipline that treats an implausibly exact result as
  evidence of a broken experiment.

Not claimed: any performance result.

## 2. Related Work

See `RELATED_WORK.md`. The literature has not yet been surveyed; the
required categories are listed as explicit open slots in
`../research/BIBLIOGRAPHY.md`.

This is a real gap and it is the largest weakness of the current document.
The central claim of RDA — that a small persistent summary adds
information the context window does not already carry — either has
precedent or has a known counterexample in the recurrent-model literature,
and that has not been checked.

## 3. Architecture

All equations below are transcribed from `nanochat/rda.py` at commit
`ab9b560`, not from a design document. Where the README and the
implementation could be read differently, the implementation is what is
described here.

### 3.1 Persistent State

The model carries an 8-dimensional state `E`, one vector per batch element,
persisted across training steps under decay-only persistence:

- `novelty`, `uncertainty`, `threat`, `integrity`, `agency`,
  `learning_progress`, `arousal`, `homeostatic_dev`

These are **primitive state dimensions, not supervised emotion labels**.
Nothing in the objective references emotion or sentiment. The names state
the function each coordinate is intended to acquire, so that its learned
structure can later be inspected. Whether the coordinates actually
differentiate into those functions is hypothesis H2 and is untested.

### 3.2 Event Signal

From the **unmodulated** pre-outcome hidden state:

```
ev = event_proj(LayerNorm(h))
ev = ev / rms(ev)                      # unit RMS over the 8 drives
event_term = 0.1 * ev
```

The output normalization is the stabilization described in Section 8.1.

### 3.3 Prediction Error

The model's own negative log-likelihood of the observed token, computed
from the **unmodulated** logits:

```
pe[p]    = NLL(lm_head(h[:p+1])[p], targets[p+1])
pe_term  = pe_proj(pe / 10.0)
```

`pe[p]` is the error for predicting token `p+1`, from a hidden state that
has consumed only tokens `<= p`. This is causal and is verified
numerically to 9.5e-07.

`pe` carries no `detach`, so it is fully differentiable. Whether it
*should* be is an open research decision, not a settled one; see
`../research/DECISIONS.md`.

### 3.4 Recurrent State Update

```
drive   = tanh(event_term + pe_term)
E_next  = 0.95 * E_prev + 0.05 * drive
```

The convex-combination form means `|E| <= 1` holds by construction rather
than by clamping. An earlier form,
`E_next = tanh(decay*E_t + tanh(event) + tanh(pe))`, could reach a
pre-squash argument of 2.95, which `tanh` maps to 0.9945 — the state was
structurally pinned near the asymptote.

The recurrence is evaluated in closed form using a decay-weighted prefix
sum in float64, which is ~200x faster than the sequential loop and
verified against it to 3.33e-16.

### 3.5 FiLM Conditioning

```
scale = 1.0 + unit_rms(film_scale(E), 0.5)
shift =      unit_rms(film_shift(E), 0.5)
hidden_mod = hidden * scale + shift
```

`hidden_mod` feeds the LM head, and the training loss is recomputed from
it. The base forward's own loss is **discarded**, so the state influences
the loss through exactly one path.

The RMS normalization is the stabilization described in Section 8.2.

### 3.6 Causality Constraints

```
film_state = cat([E_start.unsqueeze(1), E_next_seq], dim=1)
position 0 <- E_start
position p <- E_next_seq[p-1]
```

Position `p` is modulated by the state as it stood **before** the update
that consumed position `p-1`. The alternative, `E_next_seq[p]`, is a
one-token lookahead and is not used: `E_next_seq[p]` is built from
`pe_seq[p]`, which is the model's error on token `p+1`, so modulating
position `p` with it would let the logits predicting token `p+1` depend on
token `p+1`.

A prior implementation broadcast a single sequence-level state across all
positions, giving position 0 a 1022-token lookahead and making the model
unreproducible autoregressively. Both leaks were fixed in commit `bc3deb8`.

## 4. Experimental Method

Specified in `METHODS.md` and `EXPERIMENTS.md`.

### 4.1 Baseline

The unmodified nanochat path. Three seeds: 1.301217, 1.287544, 1.295109.
Mean 1.294623, spread 0.0137.

### 4.2 Treatment

B' — RDA on the stabilized architecture, identical in every other respect.

### 4.3 Seeds

42, 43, 44. Three seeds is a minimum, not a comfortable number, and the
reason is stated explicitly in the limitations: the earlier failure was a
claimed improvement smaller than the seed noise this produces.

### 4.4 Metrics

Validation bits-per-byte. State diagnostics (`max|E|`, per-dimension state,
FiLM gain, PE magnitude) are recorded every step as stability measures
rather than as performance measures.

### 4.5 Stability Diagnostics

A run aborts if any drive dimension holds `|E| > 0.98` for 10 consecutive
steps, naming the offending dimensions. The rationale is that a saturated
tanh cannot respond, so any subsequent loss improvement is not
attributable to the drive. Runs also report a TREE cleanliness state, so
that a result computed from uncommitted code is identifiable.

## 5. Results

[Placeholder — no B' A/B has been run.]

## 6. State Dynamics

[Placeholder — requires a trained B' run.]

## 7. Ablations

[Placeholder — the mechanism ablations are reported in Section 8 rather
than here, since they are about architecture validity rather than about
the hypothesis. The hypothesis ablations have not been run.]

## 8. Related Findings and Failure Analysis

This section is currently the substantive part of the paper.

### 8.1 The event projection ratchet

LayerNorm was applied to the input of `event_proj` and was working
perfectly: over 30 steps its output rms stayed at exactly 1.0000 and its
weight norm at exactly 27.7128. Over the same run `event_proj`'s weight
norm grew 1.63 -> 3.51 and the event term grew 0.25 -> 1.26.

The reason is that `||h_norm|| = sqrt(d) = 27.71` is a **ceiling, not a
value**, and `event_proj(h)` can still reach `||w_row|| * 27.71` with
`||w_row||` learned and unbounded. Freezing the base model entirely —
making `h` identical every step — changed nothing, which ruled out
hidden-state drift and isolated the gradient on `event_proj` itself.

Normalizing the projection **output** removed the ratchet: over 250 steps
the weight norm now *shrinks*, because a global scale on `W` became a no-op
and only its direction is learned.

### 8.2 The FiLM self-excitation

With the event term controlled, the state still saturated. Five ablations
over 250 steps separated three candidate explanations:

- **PE projection drift** — refuted. `pe_proj.bias` *shrinks* in the
  reference run, and freezing it entirely still saturates at the same step.
- **A differentiable feedback pathology** — refuted. Detaching PE's
  gradient, preserving its numerical value, leaves the trajectory
  indistinguishable.
- **FiLM amplification** — supported. Disabling FiLM stops saturation
  entirely, and PE still grows to 19.6 without FiLM, so PE growth is
  downstream.

The decisive control freezes the base model. There is then no varying
input at all — PE is flat at 10.50 for 250 steps — yet `pe_proj.bias` grows
1.53 -> 3.71 and the FiLM gain grows 1.00 -> 9.57. A loop that runs with
constant input is self-exciting: the state modulates the residual stream,
which changes the state, which modulates harder.

Normalizing the FiLM gain's magnitude removed it, at no cost to loss
(5.805 -> 5.808), which is the detail that indicates no shortcut was
closed.

### 8.3 A note on gradient dormancy

The `loss -> PE -> E -> FiLM -> loss` cycle is closed, but FiLM is
zero-initialized, so `dL/dE` is identically zero at step 0 and the state
pathway is gradient-dead for exactly one step. The instability is an
**onset-time** phenomenon, not a steady state — which is why it could not
be diagnosed from any single step, and why "is this loop closed?" is a
question about *when* rather than a yes/no question.

## 9. Limitations

See `LIMITATIONS.md`. The most important: the stabilization evidence is
**one seed over 250 steps**, and saturation previously began at step 217,
so it cannot yet distinguish a mechanism that is gone from one that is
merely slower.

## 10. Discussion

Two general observations, offered as such.

First, a bounded state is not automatically a stable one. Both
instabilities here had the same shape: a magnitude that was structurally
free to grow, sitting on a feedback path. Normalizing inputs did not help
in either case; normalizing the quantity on the feedback path did.

Second, the most dangerous failures in this project were not wrong
answers but *plausible, normal-looking, exactly-repeated* ones. A
diagnostic that shadowed the code it measured, and a freeze-base control
that was in no optimizer group, both returned results identical to their
controls in every digit. Neither was caught by a test. Both were caught
because exact equality after a change is implausible, and implausibility
was treated as a signal.

## 11. Conclusion

Too early. The primary question is open, the B' A/B has not been run, and
the stabilization evidence does not yet cover a horizon that would
distinguish control from delay.

The negative results are solid and the methodology is the part of this
work most likely to generalize.
