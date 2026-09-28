# LIMITATIONS

What this work cannot claim. Written to be read before the paper is
judged, so that the boundaries are visible rather than discovered.

---

## 1. The stabilization evidence does not cover enough time

**The single most important limitation.**

Saturation in the unstabilized configuration first crossed `|E| = 0.98` at
step **217**. The evidence that the fix works runs to **250 steps** — about
15% past onset.

That is not enough to distinguish a mechanism that is *gone* from one that
is *slower*. A ratchet that took 217 steps to cross a threshold could,
after a 2x suppression, take 434 — still comfortably inside a training run
of thousands of steps, and entirely consistent with the 250-step result.

**What would fix it:** the 1000-step run (E13). It is not optional and it
has not been run.

**Residual risk even after E13:** 1000 steps is still shorter than the real
training runs. The stronger control is to instrument the actual training
run with the saturation abort enabled, which already exists.

---

## 2. One seed

Every stabilization number in this project comes from seed 42. No variance
estimate exists for the stabilized configuration.

For the *ablation* comparisons this is less serious, because the
contrasts are large (max|E| 0.9853 vs 0.9402; FiLM gain 9.568 vs 1.522)
and the mechanisms are structural rather than statistical. For the
*forward-looking claim* that B' is stable, it is serious.

---

## 3. One scale, and it is not the treatment scale

Stability was established at depth 8, d_model 768, T=1024. The final B'
configuration is larger. There is no reason to expect the same behaviour,
and no evidence that it will be the same.

A mechanism that depends on a gain being free to grow could reappear at a
different scale if the gradient magnitudes differ.

---

## 4. The primary question is unanswered

The claim that a persistent low-dimensional state improves a language
model has **not been tested**. The one run that appeared to support it was
withdrawn for two independent reasons: it was a single seed whose value
(1.286248) fell inside the baseline's own range (1.287544–1.301217), and it
was produced by a model with two temporal leaks.

Everything this project has established is about whether the architecture
is *valid*, not about whether it is *useful*.

---

## 5. Three seeds is a weak experimental design

The baseline spread of 0.0137 is estimated from n=3, so the spread estimate
is itself unstable. The pre-registered comparison rule ("a difference
smaller than the spread is not a result") is a floor on rigour, not a
substitute for adequate power.

At n=3, a real but small effect could easily be missed. A null result here
should be read as "not demonstrated at this scale", never as "refuted".

---

## 6. Related work has not been surveyed

The central design claim — that a small persistent summary adds information
a bounded context window does not carry — has not been checked against the
recurrent-model literature. See `RELATED_WORK.md`.

This is a limitation on the *framing*, not just on the citations. The
design may be redundant with, or already refuted by, prior work, and that
has not been determined.

---

## 7. The dimension names are unjustified

The 8 coordinates are named `novelty`, `uncertainty`, `threat`, `integrity`,
`agency`, `learning_progress`, `arousal`, `homeostatic_dev`. Nothing
supervises them and no interpretability experiment has been run.

They are motivated names, and the names are doing real work in the
write-up — they set what a reader expects to see. Until E15, they are a
hypothesis about what the coordinates will acquire, not a description of
what they are.

They are explicitly **not** emotion labels, and no correspondence to human
affective categories is claimed or tested.

---

## 8. PE behaviour is unexplained

PE grows from 10.5 to ~36 over 250 steps in the stabilized configuration.
It no longer drives state saturation, but why it grows is not understood.

An unexplained growing input to a state signal is a genuine problem for
any claim about that signal's usefulness, because the state is being
driven by a signal whose dynamics are unknown. See E16.

---

## 9. `dpe` is unconsumed and uninterpreted

`dpe` is computed and logged and nothing uses it. This is deliberate (see
`../research/DECISIONS.md` Decision 4) — it is preserved as a clean
instrument — but it means the project has no evidence about whether the
state is structured or arbitrary.

---

## 10. Bits-per-byte may be the wrong metric

Validation bits-per-byte measures predictive quality. If the persistent
state helps with calibration, adaptation, or anomaly sensitivity, this
metric will not detect it.

That is a possible **false negative** mechanism in the primary
experiment, and it is not addressed by running more seeds. It is worth
pre-registering how such an outcome would be interpreted before the runs
are launched, precisely so it cannot be reinterpreted afterwards as a
positive.

---

## 11. The measurement instruments failed in ways that were not obvious

Two controls in this project returned results identical to their
comparisons in every digit, because the experiment was not connected to
the thing it was changing. Both were caught only by treating implausible
exact equality as a signal.

This is recorded because it bounds confidence in the positive results
too. The same class of error could produce a result that looks
*different* when it should be identical, and nothing in the current test
suite would distinguish that. See `../research/FAILED_EXPERIMENTS.md`.
