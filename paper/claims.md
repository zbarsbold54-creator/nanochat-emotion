# Claims and Source Ledger

Every factual or numerical manuscript claim is tagged below. Paths are
relative to this directory. JSON locators are structural paths, not generated
line numbers. Values rounded in the manuscript are rounded from the cited
source values.

| ID | Claim or numeric group | Exact source |
|---|---|---|
| C01 | RDA has eight named state dimensions; recurrent update and default decay | `../research/CURRENT_STATE.md:17-25, 80-100` |
| C02 | Per-position FiLM state alignment and causal shift; later no-label-leak test | `../research/CURRENT_STATE.md:106-135`; `../research/LEAK_STATUS.md:60-73` |
| C03 | FineWeb-Edu, arms A/B'/C, depth 8, width 768, device batch 2, 600 iterations, training seeds 42/43/44 | `../results/ab_v3_dbs2/SUMMARY.md:3,28-31`; width also in `../paper/LIMITATIONS.md:50` and `../research/CURRENT_STATE.md:308` |
| C04 | A final validation BPB by seed: 42=1.300680, 43=1.287114, 44=1.293590 | `../results/ab_v3_dbs2/SUMMARY.md:7-9` |
| C05 | B' final validation BPB by seed: 42=1.277996, 43=1.275198, 44=1.275858 | `../results/ab_v3_dbs2/SUMMARY.md:10-12` |
| C06 | C final validation BPB by seed: 42=1.277748, 43=1.274950, 44=1.274079 | `../results/ab_v3_dbs2/SUMMARY.md:13-15` |
| C07 | A/B'/C mean BPB and SD | `../results/ab_v3_dbs2/SUMMARY.md:18-20` |
| C08 | Paired B'-minus-A and C-minus-B' differences, each seed and mean | `../results/ab_v3_dbs2/SUMMARY.md:22-24` |
| C09 | Baseline seed spread 0.0137 BPB and its use as an early decision yardstick | `../research/EXPERIMENT_LOG.md:12-32`; `../paper/EXPERIMENTS.md:44-56` |
| C10 | D/D_const multi-timescale screen scores and seed-43 reference scores; D decay grid | `../results/multiscale_decay/SUMMARY.md:6-14`; `../results/recall_benchmark/main/main_d_dconst.txt:144` |
| C11 | V1 constant-state replacement, V2 cross-row shuffle, V3 time reversal; D deltas +0.071142/+0.001628/+0.001070 and D_const zero deltas | `../diagnostics/ablate_state.py:9-17,181-191`; `../results/multiscale_decay/SUMMARY.md:17-24` |
| C12 | Recall row/class layout, 16-way chance 0.0625, delay sets, 128-token segment mask, primary estimand and original three-seed rule | `../docs/recall_spec.md:1-12` |
| C13 | Corrected generator: 2,048 eval rows, 4,096 events, 512/512 triples, 16 values x 16 events per class/delay, disjoint streams, assertions; repair description | `../results/recall_benchmark/calib/generator_audit.txt:1-28`; `../docs/recall_spec.md:31-35` |
| C14 | Calibration gate, answer weight 20 selection, weight-5 failure and weight-20 pass with delay-16 accuracies 0.996094/1.000000 | `../results/recall_benchmark/calib/answer_weight_selection.txt:6-15`; `../results/recall_benchmark/calib/calibration_repair_summary.txt:17-26,57-61` |
| C15 | D seed-43 manipulation check, paired Cohen's d and CI; A without-mask positive-control result pending | `../results/recall_benchmark/calib/manip_check-D-s43-w20.txt:1`; no matching `recall-A-nomask-s43-w20` file under `../results/recall_benchmark/` at draft time |
| C16 | D/43 per-delay SALIENT/ORDINARY accuracies at delays 16, 48, 192, 320, 448, 576, 704, 832 | `../results/recall_benchmark/main/main_d_dconst.json`, each `entries[0].per_delay["<delay>"].SALIENT|ORDINARY` for `<delay>` in `16,48,192,320,448,576,704,832` |
| C17 | D_const/43 per-delay SALIENT/ORDINARY accuracies at the eight listed delays | `../results/recall_benchmark/main/main_d_dconst.json`, each `entries[1].per_delay["<delay>"].SALIENT|ORDINARY` for `<delay>` in `16,48,192,320,448,576,704,832` |
| C18 | D/101 per-delay SALIENT/ORDINARY accuracies at the eight listed delays | `../results/recall_benchmark/main/main_d_dconst.json`, each `entries[3].per_delay["<delay>"].SALIENT|ORDINARY` for `<delay>` in `16,48,192,320,448,576,704,832` |
| C19 | D_const/101 per-delay SALIENT/ORDINARY accuracies at the eight listed delays | `../results/recall_benchmark/main/main_d_dconst.json`, each `entries[4].per_delay["<delay>"].SALIENT|ORDINARY` for `<delay>` in `16,48,192,320,448,576,704,832` |
| C20 | 1,024 rows per delay, 8,192 total rows per seed, 6,144 pooled CROSS rows per class; “seeds” in earlier tables means eval blocks | `../docs/recall_spec.md:53-64`; `../results/recall_benchmark/calib/power_spec_draft.md:10-15` |
| C21 | Planned K=5, primary label-permutation p<=0.05 and minimum p=1/252; secondary sign-flip minimum 0.03125; threshold definition/null size; 80%-power null reporting | `../docs/recall_spec.md:53-70,89-100`; `../results/recall_benchmark/calib/power_spec_draft.md:57-83` |
| C22 | At delta 0.02, K=5, N=6,144 and sigma_seed 0.01: power 0.601 sign-flip, 0.158 threshold, 0.816 permutation | `../docs/recall_spec.md:72-86`; `../results/recall_benchmark/calib/power_analysis_did_gpu_v6_n6144.txt:11` |
| C23 | CPU-GPU validation differences 0.016, 0.025, 0.0195 and scope limitation to earlier paired-row analysis | `../docs/recall_spec.md:98-102`; `../results/recall_benchmark/calib/power_spec_draft.md:282-284` |
| C24 | Historical one-token label leak, withdrawn 0.0749 BPB result; leaked checkpoint values 1.226305 and 1.301217; fix and regression test | `../research/EXPERIMENT_LOG.md:566-633`; `../research/LEAK_STATUS.md:60-73` |
| C25 | Event-projection growth and output-normalization correction | `../research/EXPERIMENT_LOG.md:183-255` |
| C26 | 1,000-step bound check passed; later long-horizon state bound: 3,000-step criterion, first crossing at 1,353, peak 0.9848, declared failure | `../research/EXPERIMENT_LOG.md:633-662,725-761` |
| C27 | Arm-C short-run pace, 2.5x comparison, static VRAM, 14.5 MiB shared usage, no spill, unresolved cause | `../research/EXPERIMENT_LOG.md:2288-2375` |
| C28 | Later DBS=2 full comparison C/B' median ms per step; this comparison does not identify cause of earlier slowdown | `../results/ab_v3_dbs2/SUMMARY.md:10-15` |
| C29 | Optimizer named MuonAdamW | `../paper/METHODS.md:74` |
| C30 | Earlier calibration-stop result differs from repaired calibration; the former says both settings failed and stopped before D/D_const | `../results/recall_benchmark/calib/calibration_stop.txt:1-13`; repaired result in C13-C14 |
| C31 | JSON run IDs/seeds and auxiliary null entry | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[0..4].run_id|arm|seed` |
| C32 | D/43 pooled CROSS SALIENT=0.062337, ORDINARY=0.062337, gap=0 | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[0].cross_pooled.salient_accuracy|ordinary_accuracy|gap` |
| C33 | D_const/43 pooled CROSS SALIENT=0.0611979167, ORDINARY=0.0646158854, gap=-0.00341796875 | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[1].cross_pooled.salient_accuracy|ordinary_accuracy|gap` |
| C34 | D/101 pooled CROSS SALIENT=0.0626627604, ORDINARY=0.0625, gap=0.0001627604 | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[3].cross_pooled.salient_accuracy|ordinary_accuracy|gap` |
| C35 | D_const/101 pooled CROSS SALIENT=0.0626627604, ORDINARY=0.0625, gap=0.0001627604 | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[4].cross_pooled.salient_accuracy|ordinary_accuracy|gap` |
| C36 | D-minus-D_const gap contrast: seed 43=+0.00341796875; seed 101=0; two-pair mean=+0.001708984375 | Computed from source values C32-C35 using the primary contrast in C12 |
| C37 | No-context pooled accuracies/CIs and flags; CROSS pooled CIs and comparison with chance 0.0625 | `../results/recall_benchmark/main/main_d_dconst.txt:67-68,134-135,1857-1858,3526-3527` |
| C38 | Four primary run records cover only seeds 43 and 101; D_const-101 has an eval subprocess exit but no later RUN_COMPLETE in the log | `../results/recall_benchmark/main/main_d_dconst.json`, `entries[0..4]`; `../results/recall_benchmark/main/main_d_dconst.txt:3526-3533` |
| C39 | D-101 and D_const-101 are present in the promoted temporary artifact; its bytes match the current JSON; normal completion is absent after eval-D_const-101 | `../results/recall_benchmark/main/main_d_dconst.json`; `../results/recall_benchmark/main/main_d_dconst.json.tmp.promoted-20261014-191304`; `../results/recall_benchmark/main/main_d_dconst.txt:3526-3533` |
| C40 | Exact cause of the driver interruption is not recorded | `../results/recall_benchmark/main/main_d_dconst.txt:3533` is the last line; no subsequent completion or cause is recorded. Marked [UNVERIFIED] in the manuscript. |
| C41 | One natural-language corpus, separate synthetic recall task, toy-scale model/configuration | `../results/ab_v3_dbs2/SUMMARY.md:3,28`; `../paper/LIMITATIONS.md:48-53`; recall settings in `../results/recall_benchmark/main/main_d_dconst.txt:144`; design in `../docs/recall_spec.md:1-8` |
| C42 | Seed-43 recall rows were viewed descriptively before the power amendment; amendment changed N definition, was prompted by the denominator, and did not alter test, threshold, MEI, or K | `../docs/recall_spec.md:104-112`; `../results/recall_benchmark/calib/power_spec_draft.md:45-50` |

## Unverified and pending items

- [UNVERIFIED] The exact reason the recall driver stopped after eval-D_const-101 is not in the retained log.
- [PENDING] No `recall-A-nomask-s43-w20` positive-control result file was found under `results/recall_benchmark/`.
- The manuscript's minimum-effect-of-interest slot is intentionally left for author confirmation. The current `docs/recall_spec.md` records 0.02; see C21-C22.
