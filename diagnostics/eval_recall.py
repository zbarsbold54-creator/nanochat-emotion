"""Evaluate the synthetic delayed-recall task with paired bootstrap intervals."""

import argparse
import json
import os

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
             decay_override=None):
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
    for start in range(0, n_rows, batch_size):
        subset = rows[start:start + batch_size]
        inputs, targets, _, metadata = build_batch(subset, device=device)
        logits = model(inputs)
        for row_offset, row_meta in enumerate(metadata):
            event = next(
                item for item in row_meta["events"]
                if item["delay_bucket"] == (start + row_offset) // (
                    n_rows // len(DELAY_GRID)
                )
            )
            class_id = class_index[event["class"]]
            query_position = event["query_position"]
            # The task is 16-way answer retrieval, so score within the value
            # vocabulary; full-vocabulary argmax is reported separately.
            value_logits = logits[row_offset, query_position,
                                  VALUE_START:VALUE_START + N_VALUES]
            predicted = VALUE_START + int(value_logits.argmax().item())
            full_vocab_predicted = int(
                logits[row_offset, query_position].argmax().item()
            )
            expected = int(targets[row_offset, query_position].item())
            correct[start + row_offset, class_id] = predicted == expected
            full_vocab_correct[start + row_offset, class_id] = (
                full_vocab_predicted == expected
            )
            buckets[start + row_offset] = event["delay_bucket"]
        del logits
    del model
    torch.cuda.empty_cache()

    rng = np.random.default_rng(seed)
    all_results = {}
    for bucket_index, delay in enumerate(DELAY_GRID):
        bucket_mask = buckets == bucket_index
        class_accuracy = {}
        for class_id, class_name in enumerate(class_names):
            values = correct[bucket_mask, class_id].astype(np.float64)
            interval = bootstrap_interval(values, rng)
            accuracy = float(values.mean())
            class_accuracy[class_name] = accuracy
            full_values = full_vocab_correct[bucket_mask, class_id].astype(np.float64)
            full_accuracy = float(full_values.mean())
            print(
                f"ACCURACY delay={delay} class={class_name} n={len(values)} "
                f"accuracy={accuracy:.6f} ci95=[{interval[0]:.6f},{interval[1]:.6f}] "
                f"full_vocab_accuracy={full_accuracy:.6f}",
                flush=True,
            )
            if delay >= 192 and interval[0] > 0.09:
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
            "SALIENT_full_vocab": float(
                full_vocab_correct[bucket_mask, 0].astype(np.float64).mean()
            ),
            "ORDINARY_full_vocab": float(
                full_vocab_correct[bucket_mask, 1].astype(np.float64).mean()
            ),
            "paired_difference": float(paired.mean()),
        }
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
    args = parser.parse_args()
    if args.n_rows <= 0 or args.n_rows % len(DELAY_GRID):
        parser.error(f"--n-rows must be a positive multiple of {len(DELAY_GRID)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for recall evaluation")
    decays = parse_decays(args.rda_decays) if args.rda_decays else None
    evaluate(
        args.checkpoint_dir, args.step, args.seed, args.n_rows,
        args.device_batch_size, torch.device("cuda"), decay_override=decays,
    )


if __name__ == "__main__":
    main()
