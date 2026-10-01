# Phase 2 — B' A/B setup (READ ONLY, nothing launched)

Date: 2026-10-01. No command below was executed.

## Canonical commands (from logs/a_s42.log, logs/b_stab_s42.log COMMAND: line)

Common flags (`run_sweep.ps1` `$BASE_FLAGS`, identical in every A/B run):

    --depth=8 --aspect-ratio=96 --head-dim=128 --max-seq-len=1024
    --window-pattern=L --device-batch-size=4 --total-batch-size=32768
    --num-iterations=600 --eval-every=50 --eval-tokens=524288
    --core-metric-every=-1 --sample-every=-1

So d_model = 8 * 96 = 768, T = 1024. Toy scale, matching the probe.

Baseline arm A (no --rda):

    .\.venv\Scripts\python.exe -m scripts.base_train --model-tag a-s42 --seed 42 <BASE_FLAGS>
    ... --model-tag a-s43 --seed 43 <BASE_FLAGS>
    ... --model-tag a-s44 --seed 44 <BASE_FLAGS>

Treatment arm B' (identical, plus --rda):

    .\.venv\Scripts\python.exe -m scripts.base_train --model-tag b-s42 --seed 42 <BASE_FLAGS> --rda
    ... --model-tag b-s43 --seed 43 <BASE_FLAGS> --rda
    ... --model-tag b-s44 --seed 44 <BASE_FLAGS> --rda

Run order per `run_sweep.ps1` header: A 42,43,44 then B' 42,43,44.
Each run is preceded by `diagnostics/write_manifest.py --label <tag> --seed <n> [--rda] --num-iterations 600`.

## Step count and wall time

- `--num-iterations=600` for every arm; `--eval-every=50` -> 12 eval points.
- Measured: b-stab-s42 (B', seed 42) = **33.85 m**, peak 7445.49 MiB, "Total training time: 33.85m".
- Baseline A is the same architecture and step count; A and B' differ only by `--rda`,
  so ~34 min/run is the expectation for both arms. No per-arm A wall time is
  recoverable (see Ambiguities).

## Metric

- **Validation bits-per-byte (BPB)**, the quantity compared.
- Written by `base_train.py` to stdout as `Minimum validation bpb: <x>` at end of run
  (observed in logs/b_stab_s42.log).
- Independently reproducible via `scripts/base_eval.py --eval bpb --model-tag <tag> [--step N]`,
  which prints `{train,val} bpb: <x>` using
  `tokenizing_distributed_data_loader_bos_bestfit(..., split_name)` over splits
  ["train","val"]; val is the LAST parquet shard, and base_eval BPB uses split "val".

## Is the leak fix (969e067) in both arms?

**NO — and this is the blocking finding.** No B' A/B run has ever been executed on
post-`969e067` code.

Evidence:

- `b-stab-s42` ran at TREE: CLEAN, commit **`6ae93ce`**, which is the commit
  IMMEDIATELY BEFORE the fix. `969e067` ("Fix a one-token label leak in the RDA
  drive state") is dated 2026-09-29 20:24:33; `6ae93ce` is 2026-09-29 19:10:43.
- The 1.226305 BPB that the A/B would compare is from that pre-fix B' run. It is
  the already-WITHDRAWN number (E14); it measured the leak.
- The A baselines (a-s42/43/44) ran at commit `030a384` (2026-09-28), also pre-fix.
  This is harmless: those runs never passed `--rda`, and E14 established the
  baseline is unaffected by the leak. But they are at a DIFFERENT commit than any
  post-fix B' would be, so a re-run must decide whether that matters (it should
  not -- `969e067` touches only `nanochat/rda.py` -- but it is unverified here).

`run_sweep.ps1` now runs `diagnostics/test_no_label_leak.py` as a hard pre-flight
and refuses to start on failure, so any NEW sweep is protected.

## Baseline arm record at the same step count and split

- Step count: yes, 600 for both arms, same flags.
- Split: A used `--eval-tokens=524288`, and B' used the identical value.
- Seeds: baseline has 42/43/44; B' has only seed 42 (b-stab-s42, pre-fix).
  So there is NO post-fix baseline-vs-B' pair at any seed, and no post-fix B' at all.

## Ambiguities I cannot resolve from the repo

1. **Which checkpoint the A/B evaluates.** `base_train.py` saves `meta/optim` at the
   final step; `base_eval.py` defaults to `--step=None` = "last". Whether the
   A/B is defined on the final-600 checkpoint or on the best ("Minimum validation
   bpb") checkpoint is not stated in any doc. These can differ.
2. **Baseline wall time.** logs/a_s42.log, a_s43.log, a_s44.log are TRUNCATED
   (690/691/692 lines, no `Total training time`, no `Minimum validation bpb`, no
   per-step lines). The 1.301217 / 1.287544 / 1.295109 baseline numbers in E1
   therefore come from a source not present in `logs/`. Cannot confirm from where.
3. **B' tag naming.** The manifest uses `b-causal-s42`; the log uses `b-stab-s42`;
   E14 calls the checkpoint `B'-s42`. Three names, unclear whether one run or two.
4. **Whether a-s42 should be re-run.** `a-s42-post.json` exists in manifests/ but
   its purpose is undocumented; it may be a re-run after the causality fixes, or
   something else.
5. **Eval batch size for the recorded BPB.** `--eval-every=50` path vs
   `base_eval.py --device-batch-size` (default 32) are different code paths and
   need not produce identical BPB. The A/B number's provenance is not pinned.