# Persistent State and Feature-wise Conditioning in a Small Transformer

*A toy-scale evaluation of the Regulatory Drive Architecture (RDA)*

## Abstract

We evaluate whether a persistent, low-dimensional functional state changes
language-model prediction or recall beyond the model's ordinary attention
path. RDA carries eight named drive dimensions, updates them recurrently, and
uses per-position causal FiLM conditioning. The names are design labels: this
study does not measure or claim subjective emotion. [C01, C02]

In a FineWeb-Edu comparison at depth 8 and width 768, the three-seed mean
final validation BPB was 1.293795 for baseline A, 1.276351 for dynamic-state
B', and 1.275592 for constant-state C. The mean paired B'-minus-A difference
was -0.017444; C was -0.000758 BPB below B'. Under the pre-set baseline-spread
rule, the A-to-RDA change clears the recorded 0.0137 threshold, whereas the
B'-to-C contrast does not. The result is consistent with a contribution from
the shared FiLM pathway or capacity, not evidence that dynamic state content
is necessary. [C03-C09]

On the recall task, both D and D_const solve the two within-segment buckets
but remain near 16-way chance on pooled cross-segment buckets. The paired
training-seed difference-in-differences is +0.003418 for seed 43 and 0 for
seed 101; only two seed pairs are available, below both the earlier
three-seed claim rule and the later five-seed powered design. These results
do not establish cross-segment recall or a state-content benefit.
[C16-C20, C32-C38]

This is a toy-scale, single-dataset study with few training seeds. It is
descriptive rather than confirmatory; its corrections, incomplete grid, and
unresolved artifacts are reported alongside the measurements. [C03, C20,
C23-C29]

## 1. Introduction

Transformers normally carry information through attention and their
context-dependent activations. RDA adds a persistent functional state and
asks whether it changes what the model remembers or predicts. The hypothesis
is that a recurrent summary of prediction error and event information, fed
back through causal, per-position conditioning, can change model behavior.
This is a functional hypothesis, not a claim that the model feels, experiences,
or has subjective emotion. [C01, C02]

The study separates two possible sources of an observed language-model
difference: the availability of the extra FiLM pathway and the content of a
varying state carried by that pathway. A constant-state control retains the
conditioning route while removing position-varying state. A separate recall
task tests whether dynamic state supports cross-segment retrieval when
attention is blocked between segments. [C03, C12]

## 2. Related Work

The manuscript needs verified prior-work coverage before publication:

- Nanochat and the base language-model training setup. [CITE: nanochat
  architecture and training setup]
- Feature-wise linear modulation and related conditioning methods.
  [CITE: FiLM and feature-wise conditioning]
- The optimizer components used by the training setup.
  [CITE: Muon and AdamW]
- Recurrent or persistent memory in language models.
  [CITE: recurrent memory in language models]
- Functional affect- or drive-like representations in machine learning.
  [CITE: computational affect and drive representations; avoid claims about
  subjective experience]

No authors, titles, venues, or publication dates are supplied here from
memory. The open-items list tracks these citation slots.

## 3. Architecture and Experimental Arms

### 3.1 Persistent functional state

The state E has eight dimensions named novelty, uncertainty, threat,
integrity, agency, learning progress, arousal, and homeostatic deviation.
The recurrent update is

> E[t] = decay * E[t-1] + (1 - decay) * drive[t].

The state conditions the transformer through FiLM. Conditioning is aligned
per position using the state available before the current position's update;
the later audit describes and tests this causal shift. The drive names are
not emotion labels learned from supervision, and no measure of subjective
experience is part of the objective or evaluation. [C01, C02]

### 3.2 Arms

The language-model comparison used A (no RDA flags), B' (RDA with dynamic
state), and C (RDA with constant-state FiLM control). The later recall
screen used D (multi-timescale decay) and D_const (its constant-state
control). D used decay values 0.9, 0.95, 0.975, 0.9875, 0.99375, 0.996875,
0.9984375, and 0.99921875. [C03, C10]

## 4. Experimental Method

### 4.1 Language-model comparison

The controlled A/B'/C comparison used FineWeb-Edu, depth 8, width 768,
device batch size 2, and 600 training iterations. The comparison used
training seeds 42, 43, and 44; B' and C results for seed 42 were reused from
prior device-batch-2 logs rather than retrained in the comparison run. The
optimizer configuration is recorded as MuonAdamW. [C03, C29]

The baseline-spread decision threshold was set to 0.0137 BPB from an earlier
three-seed A run. This is a rule for this small experiment, not a universal
noise threshold or a formal population-level uncertainty estimate. [C09]

### 4.2 Recall benchmark

Each evaluation row contains one SALIENT and one ORDINARY event. The answer
is selected from 16 values, giving chance accuracy 0.0625. WITHIN delays are
16 and 48 tokens; CROSS delays are 192, 320, 448, 576, 704, and 832. A
block-diagonal causal attention mask uses 128-token segments, so information
cannot pass between segments through attention. The primary contrast is the
D-versus-D_const difference in the SALIENT-minus-ORDINARY gap pooled over
CROSS delays 192-832. [C12]

The repaired generator audit found all 512 possible class/key/value triples,
multiple values for each class/key combination, balanced value IDs within
each class-delay stratum, and disjoint training and evaluation row identities.
All recorded audit assertions passed. [C13]

Calibration froze answer weight 20 for every arm: weight 5 failed the
delay-16 accuracy gate, while weight 20 passed it for both classes. The
repaired calibration record gives 0.996094 SALIENT and 1.000000 ORDINARY
accuracy at delay 16. An earlier pre-repair calibration file reports a
different outcome; the repaired generator/calibration is the result used
here, and the version distinction is documented in the open-items list.
[C14, C15]

The D manipulation check on seed 43 passed, with paired Cohen's d 5.710474
and a 95% interval [5.533349, 5.899485]. This check is not the separate
positive control of arm A without segment masking. That positive-control
result remains [PENDING: result not yet available]. [C15]

### 4.3 Planned contrast and power analysis

The protocol defines N as total paired evaluation rows per training seed
across all eight delay buckets; the power-analysis N is the pooled CROSS
row count per class. The planned primary design is five training seeds per
arm and 8,192 total evaluation rows per seed, equivalent to 6,144 pooled
CROSS rows per class. The earlier claim rule called for three seeds; the
later amendment set the five-seed powered design. “Seeds” in the earlier
power tables should not be confused with evaluation blocks. [C20, C21]

The protocol names the one-sided exhaustive label-permutation test over all
2K per-seed evaluation gaps as primary, rejecting at p <= 0.05; its minimum
achievable p-value at K=5 is 1/252. The paired exact sign-flip test by seed
index and the threshold rule are secondary; the K=5 sign-flip minimum is
0.03125. The sign-flip pairing is by seed index even though the D and
D_const training seeds are independent. The threshold rejects when the
estimated contrast exceeds the larger of the two within-arm spreads, each
computed as the maximum minus minimum of that arm's K eval gaps. Its
simulated null size ranged from 0.0000 to 0.0275 and was about 0.001 at
K=5, so it is conservative rather than a calibrated test. The 5,000
bootstrap repetitions in an earlier draft were removed because no reported
quantity used them. The pre-existing protocol records a minimum effect of
interest of 0.02, but this draft leaves the author-facing value open for
confirmation: **Minimum effect of interest: [AUTHOR TO CONFIRM OR ENTER]**.
Power at delta 0.02, K=5, N=6,144, and assumed seed SD 0.01 was 0.601 for
sign-flip, 0.158 for the threshold rule, and 0.816 for permutation. These
figures are conditional on the assumed seed variance, not observed power
from the incomplete recall grid. [C21, C22]

Seed-43 D/D_const recall rows had already been viewed descriptively before
the power amendment. The amendment corrected the definition of N and added
the matching N=6,144 row; the record says it did not change the test,
threshold, minimum effect, or K and was prompted by the denominator rather
than outcome values. [C21, C42]

The null-reporting rule is to report the minimum detectable effect at 80%
power for the chosen K and N rather than interpret a non-significant result
as evidence of no effect. The recorded CPU-GPU validation differences
(0.016, 0.025, and 0.0195) are described in the protocol as within Monte
Carlo error; that validation concerned an earlier paired-row analysis, not
the later model with training-seed variance. [C23]

## 5. Errors and Corrections

### 5.1 One-token label leak

An audit found that an earlier prediction-error slice allowed the FiLM state
at a position to depend on the exact target token scored at that position.
The apparent B'-seed-42 result, 1.226305 BPB against a 1.301217 baseline,
and its 0.0749 BPB difference were measured with this leak and are withdrawn.
The indexing was corrected and a no-label-leak regression test was added;
the later audit reports that test passing at its probed positions. [C24]

### 5.2 Saturation

LayerNorm did not bound the learned event projection. A short investigation
isolated growth in the projection itself; normalizing its output controlled
that mechanism. A separate PE/FiLM amplification path remained. Although a
shorter production confirmation passed its bound, the subsequent 3,000-step
no-abort run crossed the 0.98 state bound at step 1,353, peaked at 0.9848,
and failed the predeclared long-horizon criteria. The stabilization should
therefore not be described as establishing long-run state stability.
[C25, C26]

### 5.3 Arm-C slowdown

A short-run diagnostic measured C at about 2.5 times B' step time, but the
reported explanation based on VRAM pressure was explicitly withdrawn:
memory was static and similar across arms, measured shared-memory use was
14.5 MiB, and the diagnostic concluded that there was no shared-memory
spill. The cause remained open. A later controlled comparison reported C
faster than B' by its median-step metric; this does not identify the cause
of the earlier short-run slowdown. The manuscript therefore does not call
the slowdown a VRAM-spill artifact. [C27, C28]

### 5.4 Recall calibration correction

The pre-repair calibration stop file says both answer weights failed and
training should stop. The repaired generator audit and subsequent
calibration record show that weight 20 passed; the main D/D_const recall
runs followed that repaired calibration. These are versioned results, not
two measurements of one unchanged calibration. [C13-C15, C30]

## 6. Language-Model Results

### 6.1 A, B', and C

Final validation BPB by seed:

| Arm | Seed 42 | Seed 43 | Seed 44 | Mean (SD) |
|---|---:|---:|---:|---:|
| A | 1.300680 | 1.287114 | 1.293590 | 1.293795 (0.006785) |
| B' | 1.277996 | 1.275198 | 1.275858 | 1.276351 (0.001463) |
| C | 1.277748 | 1.274950 | 1.274079 | 1.275592 (0.001917) |

The paired B'-minus-A differences were -0.022684, -0.011916, and -0.017732
for seeds 42, 43, and 44, respectively (mean -0.017444 BPB). The magnitude
of the mean clears the recorded 0.0137 baseline-spread threshold. This is
the result under that decision rule, not a general claim of improvement.
[C04-C09]

The paired C-minus-B' differences were -0.000248, -0.000248, and -0.001779
(mean -0.000758 BPB). Their magnitude is below the 0.0137 threshold. Under
the pre-set rule, the B'-versus-A difference is present in both dynamic and
constant-state FiLM arms, while the dynamic-versus-constant contrast does
not clear the threshold. The result is consistent with an effect of the
shared RDA/FiLM pathway or capacity rather than evidence that dynamic state
content is required. It does not identify a unique mechanism. [C08, C09]

### 6.2 Multi-timescale screen and eval-time ablations

At the final screen checkpoint, D had validation BPB 1.275386 and D_const
had 1.274947; the screen used seed 43. The difference does not favor D.
The evaluation-time ablations on the D checkpoint changed BPB by +0.071142
for V1 (constant-state replacement), +0.001628 for V2 (cross-row state
shuffle), and +0.001070 for V3 (within-row time reversal). The corresponding
D_const changes were all zero. Thus the small 0.001-0.002 changes apply to
the shuffle/order perturbations, not to every meaning of “state content”;
the larger V1 result is a single-checkpoint evaluation-time replacement
and is not equivalent to comparing separately trained D and D_const arms.
[C10, C11]

## 7. Recall Results

### 7.1 Per-delay accuracy

Each cell below is SALIENT / ORDINARY accuracy. N was 1,024 paired rows per
delay bucket and class; each row contains both event classes. [C12, C20]

| Arm / training seed | 16 | 48 | 192 | 320 | 448 | 576 | 704 | 832 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D / 43 | .996094/.998047 | .998047/.999023 | .063477/.062500 | .063477/.063477 | .060547/.061523 | .062500/.063477 | .061523/.061523 | .062500/.061523 |
| D_const / 43 | .997070/1.000000 | 1.000000/1.000000 | .063477/.061523 | .055664/.059570 | .067383/.061523 | .058594/.066406 | .064453/.070312 | .057617/.068359 |
| D / 101 | .997070/.999023 | 1.000000/1.000000 | .063477/.062500 | .062500/.062500 | .062500/.062500 | .062500/.062500 | .062500/.062500 | .062500/.062500 |
| D_const / 101 | .997070/.999023 | 1.000000/1.000000 | .063477/.062500 | .062500/.062500 | .062500/.060547 | .061523/.063477 | .063477/.062500 | .062500/.063477 |

Each row is extracted from its matching JSON `entries[]` record; the
per-delay sample count and delay definitions are shared across rows.
[C16-C19]
The JSON's separate seed-43 record has a null arm and run_id, so it is not
counted as a trained-arm result in this table. [C31]

### 7.2 Pooled CROSS contrast and null controls

| Arm / seed | CROSS SALIENT | CROSS ORDINARY | SALIENT-minus-ORDINARY gap |
|---|---:|---:|---:|
| D / 43 | 0.062337 | 0.062337 | 0.000000 |
| D_const / 43 | 0.061198 | 0.064616 | -0.003418 |
| D / 101 | 0.062663 | 0.062500 | +0.000163 |
| D_const / 101 | 0.062663 | 0.062500 | +0.000163 |

The resulting paired D-minus-D_const differences in gaps are +0.003418
(seed 43) and 0 (seed 101), a descriptive two-pair mean of +0.001709.
There is no three- or five-seed contrast interval or confirmatory test in
these results. For all four runs the no-context pooled control is flagged
within its permutation-null interval; the reported pooled CROSS intervals
include chance accuracy 0.0625. No cross-segment recall or state-content
advantage is established by these data. [C20, C32-C37]

### 7.3 Incomplete grid and result provenance

The older claim rule required three training seeds; the later powered plan
specified five. The available paired grid contains only seeds 43 and 101,
so it is incomplete under either plan. The driver log ends after the
eval-D_const-101 subprocess exit and has no corresponding run-completion
record. The precise reason it stopped is [UNVERIFIED]. [C21, C38, C40]

The D-101 and D_const-101 entries in the current results JSON were recovered
from the retained temporary-result artifact and promoted after the driver
stopped before its normal result-commit step. The promoted artifact and
current JSON contain the same data; the log records successful eval
subprocess exits but not normal completion of the D_const-101 item. [C39]

## 8. Discussion and Limitations

The language-model comparison supports a narrow capacity-level reading:
B' and C both outperform A under the recorded baseline-spread rule, while
B' and C do not differ by that rule. The comparison therefore does not
establish that dynamic state content accounts for the BPB difference. The
recall evidence is likewise negative for cross-segment memory: D and
D_const remain near chance in the CROSS buckets, and the available
two-seed difference-in-differences is descriptive only. [C08-C09,
C32-C38]

These findings do not show that RDA improves language modeling generally,
that the state carries useful cross-segment memories, or that any drive
corresponds to subjective emotion. The language-model screen uses one
natural-language corpus and a small model configuration, alongside a
separate synthetic recall task. The primary recall grid has fewer training
seeds than planned; the multi-timescale result and its ablations come from
one seed and one checkpoint. The long-horizon saturation test also failed
its declared criteria. [C03, C10-C11, C20, C26, C28, C41]

The next tests should complete the pre-specified paired training-seed
design, keep evaluation-row pairing and seed indexing explicit, estimate
training-seed variance, and report the 80%-power minimum detectable effect
for the chosen K and N. The author should confirm the minimum effect of
interest before treating a null result as informative. The missing A
without-mask positive control and the cause of the interrupted driver run
remain open. [C15, C20-C23, C40]

## 9. Conclusion

At this toy scale, the observed language-model difference is shared by
dynamic-state and constant-state FiLM arms, and the recall runs do not
establish cross-segment retrieval. The data support neither a claim of
subjective emotion nor a demonstrated state-content benefit. The incomplete
recall grid and unresolved artifacts limit stronger conclusions.
