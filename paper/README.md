# Paper

This is a **living research manuscript**. It is under construction and most
sections are empty on purpose.

## Status

The paper has **no results**. The primary hypothesis — that a persistent
low-dimensional internal state improves a language model — has not been
tested. The single run that appeared to support it was withdrawn; see
`../research/FAILED_EXPERIMENTS.md` F1.

Sections 5, 6 and 7 (`Results`, `State Dynamics`, `Ablations`) are
intentionally placeholders. **Do not write numbers into them until the
experiment that produces those numbers has been run and recorded in
`../research/EXPERIMENT_LOG.md`.**

## Files

| File | Contents |
|---|---|
| `DRAFT.md` | The manuscript skeleton, in section order |
| `RELATED_WORK.md` | Positioning; literature not yet surveyed (see the open slots in `../research/BIBLIOGRAPHY.md`) |
| `METHODS.md` | Experimental protocol: baseline, treatment, seeds, metrics |
| `EXPERIMENTS.md` | The experiments as specified, before results exist |
| `RESULTS.md` | **Empty.** To be filled only from EXPERIMENT_LOG.md |
| `LIMITATIONS.md` | What this work cannot claim |
| `figures/` | Empty. No figures have been generated |

## Rules for this directory

1. **Every number must point back to a reproducible experiment.** If a
   figure is not produced by a script in `diagnostics/` or a run recorded
   in `../research/EXPERIMENT_LOG.md`, it does not go in the paper.
2. **Spread before mean.** Report the seed-to-seed spread alongside any
   mean. The project's most instructive failure was a claimed improvement
   smaller than the baseline's own noise.
3. **Cite the retraction, not just the result.** If a number was
   withdrawn, the paper says so.
4. **No placeholder numbers.** `[Placeholder]` is acceptable;
   `1.29 (TBD)` is not — it reads as a measurement.
5. **Negative results are findings.** The two saturation mechanisms and
   their diagnosis are the most substantial results this project has
   produced so far, and they belong in the paper regardless of whether the
   primary hypothesis survives.

## Relationship to `research/`

`../research/` is the record of what was done and what it means.
`paper/` is the argument built on top of it. Where they disagree, `research/`
is correct — the paper is the derived artifact and is expected to lag.
