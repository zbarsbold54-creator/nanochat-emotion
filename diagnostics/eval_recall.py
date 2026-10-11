"""Evaluate the synthetic delayed-recall task with paired bootstrap intervals."""

import argparse
import json
import os
import time

os.environ["NANOCHAT_DTYPE"] = "float32"

import numpy as np
import torch

from diagnostics.recall_models import load_model, parse_decays
from nanochat.recall_data import (
    DELAY_GRID,
    N_VALUES,
    VALUE_START,
    build_batch,
    build_eval_grid,
)


def bootstrap_interval(values, rng, repetitions=5000):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


@torch.inference_mode()
def evaluate(checkpoint_dir, step, seed, n_rows, batch_size, device,
             decay_override=None, check_cross_leak=True):
    started_at = time.perf_counter()
    model, _, _ = load_model(
        checkpoint_dir, step, device, seed, segment_len=128,
        decay_override=decay_override,
    )
    rows = build_eval_grid(n_rows=n_rows)
    class_names = ("SALIENT", "ORDINARY")
    class_index = {name: index for index, name in enumerate(class_names)}
    correct = np.zeros((n_rows, 2), dtype=np.bool_)
    full_vocab_correct = np.zeros((n_rows, 2), dtype=np.bool_)
    buckets = np.zeros(n_rows, dtype=np.int64)
    predictions = np.zeros((n_rows, 2), dtype=np.int64)
    true_values = np.zeros((n_rows, 2), dtype=np.int64)
    row_value_equal = np.zeros(n_rows, dtype=np.bool_)
    for start in range(0, n_rows, batch_size):
        subset = rows[start:start + batch_size]
        inputs, targets, _, metadata = build_batch(subset, device=device)
        logits = model(inputs)
        for row_offset, row_meta in enumerate(metadata):
            row_index = start + row_offset
            row_events = row_meta["events"]
            row_value_equal[row_index] = (
                row_events[0]["value"] == row_events[1]["value"]
            )
            for event in row_events:
                class_id = class_index[event["class"]]
                query_position = event["query_position"]
                # The task is 16-way answer retrieval, so score within the
                # value vocabulary; full-vocabulary argmax is separate.
                value_logits = logits[
                    row_offset, query_position,
                    VALUE_START:VALUE_START + N_VALUES,
                ]
                predicted = VALUE_START + int(value_logits.argmax().item())
                full_vocab_predicted = int(
                    logits[row_offset, query_position].argmax().item()
                )
                expected = int(targets[row_offset, query_position].item())
                predictions[row_index, class_id] = predicted
                true_values[row_index, class_id] = expected
                correct[row_index, class_id] = predicted == expected
                full_vocab_correct[row_index, class_id] = (
                    full_vocab_predicted == expected
                )
                buckets[row_index] = event["delay_bucket"]
        del logits
    del model
    torch.cuda.empty_cache()

    rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed + 7919)
    permutation_matches = np.zeros((n_rows, 2), dtype=np.bool_)
    other_event_matches = np.zeros((n_rows, 2), dtype=np.bool_)
    for bucket_index in range(len(DELAY_GRID)):
        bucket_rows = np.flatnonzero(buckets == bucket_index)
        paired_rows = permutation_rng.permutation(bucket_rows)
        while np.any(paired_rows == bucket_rows):
            paired_rows = permutation_rng.permutation(bucket_rows)
        permutation_matches[bucket_rows] = (
            predictions[bucket_rows] == true_values[paired_rows]
        )
        other_event_matches[bucket_rows, 0] = (
            predictions[bucket_rows, 0] == true_values[bucket_rows, 1]
        )
        other_event_matches[bucket_rows, 1] = (
            predictions[bucket_rows, 1] == true_values[bucket_rows, 0]
        )
    print(
        f"ROW_VALUE_COLLISION rows={n_rows} equal={int(row_value_equal.sum())} "
        f"percent={100.0 * row_value_equal.mean():.6f}",
        flush=True,
    )
    all_results = {}
    for bucket_index, delay in enumerate(DELAY_GRID):
        bucket_mask = buckets == bucket_index
        class_accuracy = {}
        for class_id, class_name in enumerate(class_names):
            values = correct[bucket_mask, class_id].astype(np.float64)
            interval = bootstrap_interval(values, rng)
            accuracy = float(values.mean())
            class_accuracy[class_name] = accuracy
            permutation_values = permutation_matches[
                bucket_mask, class_id
            ].astype(np.float64)
            permutation_interval = bootstrap_interval(
                permutation_values, rng
            )
            permutation_accuracy = float(permutation_values.mean())
            full_values = full_vocab_correct[bucket_mask, class_id].astype(np.float64)
            full_accuracy = float(full_values.mean())
            print(
                f"ACCURACY delay={delay} class={class_name} n={len(values)} "
                f"accuracy={accuracy:.6f} ci95=[{interval[0]:.6f},{interval[1]:.6f}] "
                f"permutation_null={permutation_accuracy:.6f} "
                f"permutation_ci95=[{permutation_interval[0]:.6f},"
                f"{permutation_interval[1]:.6f}] "
                f"full_vocab_accuracy={full_accuracy:.6f}",
                flush=True,
            )
            if check_cross_leak and delay >= 192 and interval[0] > 0.09:
                raise SystemExit(
                    f"LEAK: CROSS delay {delay} {class_name} "
                    f"CI lower bound exceeds 0.09"
                )
        paired = (
            correct[bucket_mask, 0].astype(np.float64)
            - correct[bucket_mask, 1].astype(np.float64)
        )
        paired_interval = bootstrap_interval(paired, rng)
        print(
            f"PAIRED delay={delay} salient_minus_ordinary={paired.mean():.6f} "
            f"ci95=[{paired_interval[0]:.6f},{paired_interval[1]:.6f}]",
            flush=True,
        )
        all_results[str(delay)] = {
            "SALIENT": class_accuracy["SALIENT"],
            "ORDINARY": class_accuracy["ORDINARY"],
            "SALIENT_permutation_null": float(
                permutation_matches[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_permutation_null": float(
                permutation_matches[bucket_mask, 1].astype(np.float64).mean()
            ),
            "SALIENT_full_vocab": float(
                full_vocab_correct[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_full_vocab": float(
                full_vocab_correct[bucket_mask, 1].astype(np.float64).mean()
            ),
            "paired_difference": float(paired.mean()),
        }
    cross_rows = buckets >= 2
    cross_accuracy_by_row = correct[cross_rows].astype(np.float64).mean(axis=1)
    cross_permutation_by_row = permutation_matches[cross_rows].astype(
        np.float64
    ).mean(axis=1)
    cross_accuracy_ci = bootstrap_interval(cross_accuracy_by_row, rng)
    cross_permutation_ci = bootstrap_interval(cross_permutation_by_row, rng)
    print(
        f"CROSS_POOLED accuracy={cross_accuracy_by_row.mean():.6f} "
        f"ci95=[{cross_accuracy_ci[0]:.6f},{cross_accuracy_ci[1]:.6f}] "
        f"permutation_null={cross_permutation_by_row.mean():.6f} "
        f"permutation_ci95=[{cross_permutation_ci[0]:.6f},"
        f"{cross_permutation_ci[1]:.6f}]",
        flush=True,
    )
    for class_id, class_name in enumerate(class_names):
        print(
            f"CROSS_MATCH class={class_name} "
            f"same_row_other_event="
            f"{other_event_matches[cross_rows, class_id].mean():.6f} "
            f"permuted_other_row="
            f"{permutation_matches[cross_rows, class_id].mean():.6f}",
            flush=True,
        )
    print(
        "CROSS_MATCH pooled "
        f"same_row_other_event={other_event_matches[cross_rows].mean():.6f} "
        f"permuted_other_row={permutation_matches[cross_rows].mean():.6f}",
        flush=True,
    )
    print(f"EVAL_WALL_SECONDS={time.perf_counter() - started_at:.3f}", flush=True)
    print("RESULT_JSON=" + json.dumps(all_results), flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=600)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--n-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rda-decays", default=None)
    parser.add_argument("--allow-cross-above-0.09",
                        dest="allow_cross_above_0_09",
                        action="store_true",
                        help="do not apply the untrained-A cross-delay leak stop")
    args = parser.parse_args()
    if args.n_rows <= 0 or args.n_rows % len(DELAY_GRID):
        parser.error(f"--n-rows must be a positive multiple of {len(DELAY_GRID)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for recall evaluation")
    decays = parse_decays(args.rda_decays) if args.rda_decays else None
    evaluate(
        args.checkpoint_dir, args.step, args.seed, args.n_rows,
        args.device_batch_size, torch.device("cuda"), decay_override=decays,
        check_cross_leak=not args.allow_cross_above_0_09,
    )


if __name__ == "__main__":
    main()
