# Open Items

## Author decisions and missing evidence

- Confirm or replace the manuscript's minimum-effect-of-interest value before
  confirmatory interpretation. The current `docs/recall_spec.md` records
  0.02, while the manuscript deliberately leaves an author-facing slot.
- The recall design is incomplete: the older claim rule requires three
  training seeds, while the later power amendment plans five. Only seeds 43
  and 101 have paired D/D_const results. Decide which version of the
  pre-registration governs any future confirmatory claim.
- The main driver log ends after eval-D_const-101 without the normal
  completion record. The operational record shows the run was incomplete, but
  the exact cause is [UNVERIFIED].
- The arm-A-without-segment-masking positive-control result
  `recall-A-nomask-s43-w20` was not found. Its outcome is [PENDING: result not
  yet available].
- Add verified citations at the five marked locations in `DRAFT.md`: nanochat,
  FiLM, Muon/AdamW, recurrent memory in language models, and computational
  affect/drive representations. Do not add references from memory.
- Reconcile the older planning notes in `METHODS.md`, `EXPERIMENTS.md`, and
  `LIMITATIONS.md` against the current manuscript before submission. In
  particular, `EXPERIMENTS.md` still labels the A/B comparison not run,
  whereas the device-batch-2 summary reports the completed comparison.

## Repository evidence conflicts and qualifications

- **Arm-C slowdown:** the earlier pace report's VRAM-pressure explanation is
  explicitly withdrawn in `research/EXPERIMENT_LOG.md` E18. That diagnostic
  reports no shared-memory spill and leaves the cause open. A later
  device-batch-2 comparison has C faster by its median-step metric. The draft
  does not claim a VRAM-spill cause; further profiling would be needed to
  reconcile the different run conditions.
- **Calibration history:** `calibration_stop.txt` records a pre-repair
  weight-20 failure and stop. The corrected generator audit and later
  calibration record report a weight-20 pass. The draft treats these as
  versioned runs and uses the repaired calibration, not as contradictory
  measurements on unchanged data.
- **Seed count:** the earlier three-seed claim rule and the later
  five-seed power amendment are both present in `docs/recall_spec.md`. The
  two available seed pairs meet neither target.
- **Eval-time ablations:** the D checkpoint's constant-state replacement
  result differs from its cross-row shuffle and time-reversal results. These
  are distinct interventions on one checkpoint, not substitutes for the
  separately trained B'/C comparison.
- **Evidence status:** no claim of cross-segment recall, subjective emotion,
  or a dynamic-state language-model benefit is supported by the current
  results. The recall difference-in-differences has only two seed pairs and
  no confirmatory interval or test.
