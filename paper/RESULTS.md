# RESULTS

**EMPTY. NO RESULTS HAVE BEEN PRODUCED.**

The primary experiment — the 3-seed B' A/B comparison — has not been run.
It is blocked on the 1000-step stability confirmation (E13).

---

## What would go here

When results exist, this file will contain, in order:

1. **Baseline A**, all three seeds, mean, spread, standard deviation.
   (Available now: 1.301217 / 1.287544 / 1.295109, mean 1.294623,
   spread 0.0137.)

2. **B'**, all three seeds, mean, spread, standard deviation.

3. **The comparison**, stated as `mean(B') - mean(A)` alongside the
   baseline spread of 0.0137, with an explicit statement of whether the
   difference exceeds it.

4. **Stability during the B' runs**: whether any run saturated, and the
   maximum `|E|` reached.

5. **State dynamics**: per-dimension trajectories, if E15 has been run.

---

## Rules for this file

1. **No number appears here that is not in `../research/EXPERIMENT_LOG.md`.**
   This file is a presentation layer, not a source.

2. **Spread before mean.** The project's first headline result was a
   single-seed number that turned out to sit inside the baseline's own
   range. That is the specific failure this file exists to make impossible
   to repeat.

3. **A negative result is reported as a negative result.** "Not supported
   at this scale", not a softened positive and not "refuted" — three seeds
   on one model cannot refute a general claim.

4. **Retracted numbers appear only with their retraction attached.** The
   value 1.286248 may appear in this paper only alongside the statement
   that it was withdrawn and why.

---

## What is currently known, and is not a result

For completeness, so that this file is not mistaken for an absence of
findings. The following are **architecture-validity** findings, reported
in `DRAFT.md` Section 8 and in `../research/EXPERIMENT_LOG.md`:

- The event-projection ratchet was isolated to the gradient on
  `event_proj` itself, with hidden-state drift ruled out by a frozen-base
  control.
- The PE/FiLM runaway was isolated to FiLM amplification, with PE
  projection drift and a differentiable-feedback pathology both refuted by
  dedicated controls.
- The `loss -> PE -> E -> FiLM -> loss` cycle is closed but gradient-dormant
  at step 0 under zero-initialization.
- Both stabilizations are reparameterizations that leave loss essentially
  unchanged (5.805 -> 5.808), which indicates no loss shortcut was closed.

None of these say anything about whether the persistent state helps. They
say the architecture is now in a state where the question can be asked
without the answer being an artefact.
