Purpose: test whether a persistent internal state retains surprising events better than ordinary ones, beyond the model's attention range.
Row layout (T=1024, BOS first): filler stream from a first-order Markov chain over filler vocab F (|F|=512 reserved token ids). The transition matrix is doubly stochastic (stationary distribution uniform, verified to 1e-3), each token has ~4 likely successors.
Events: each row has exactly 2 events, 1 SALIENT and 1 ORDINARY, order randomized 50/50. An event = lead region (8 tokens) + [K][V]. K from key pool (16 ids), V from value pool (16 ids), distinct keys within a row, both sampled uniformly and independently of class. SALIENT lead = 8 tokens sampled uniformly from F (anomaly burst, breaks the chain). ORDINARY lead = 8 tokens of normal chain filler. Nothing else differs between classes.
Queries: one per event, block [Q][K] where the target at K's position is V (the V token then follows teacher-forced). Both events are queried every row, so queries carry no information about salience. Chance = 1/16 = 0.0625.
Delay = tokens from event [K] position to query [K] position. Eval delay grid: WITHIN-segment {16, 48}; CROSS-segment {192, 320, 448, 576, 704, 832}. Training delays: uniform over the same feasible range. Event position p sampled uniformly over the feasible interval for the given delay, by the SAME rule for both classes. For WITHIN delays the event and query must be in the same 128-token segment; for CROSS delays in different segments. Events/queries must not overlap.
Attention: block-diagonal causal mask with segment length 128 (flag --segment-len 128). Across segments, attention carries no information; only the RDA state can.
Controls and arms: A (no RDA), B', D (multi-timescale decay), plus constant-state controls C and D_const. A with the segment mask is the hard no-memory floor.
Primary outcome: recall accuracy on the answer token. Primary estimand: within-row paired difference (SALIENT minus ORDINARY) on CROSS delays, per arm, and its difference-in-differences against the arm's constant-state control.
Claim rules (fixed in advance):
 (i) a manipulation check must pass: the state is measurably more elevated at salient than at ordinary events for the arm tested;
 (ii) the arm must exceed chance on CROSS delays (bootstrap 95% CI over eval rows above 0.0625);
 (iii) the difference-in-differences vs the constant-state control must have a 95% CI excluding 0 in the predicted direction, pooled over 3 seeds, with the same sign in all 3 seeds;
 (iv) FineWeb BPB is reported for every arm against A trained on the same data mixture.
If an arm is at or below chance on CROSS delays, report a CAPACITY NULL, not a salience null.
Effect-size threshold and eval row counts: set by a power analysis (next prompt) and committed before any arm is evaluated.
Seeds: train rows from seed stream 0 upward, eval rows from seed base 1,000,000,000, never overlapping. Rows are deterministic from (seed, index).

Clarifications after segment-invariance validation
(a) WITHIN delays (16, 48) are attention-reachable; untrained models can recall there via in-context induction (untrained A: 0.203 at delay 16, SALIENT). WITHIN buckets are a positive control after training, and the no-memory floor test applies only to CROSS delays.
(b) Scoring uses argmax restricted to the 16 value ids (chance 0.0625), and full-vocabulary accuracy is reported as a secondary column.
(c) The first floor-check rule was misapplied to WITHIN buckets and was corrected on 2026-10-11.
