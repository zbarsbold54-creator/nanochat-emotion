#!/usr/bin/env python
"""Diag wrapper: exact M command, instrumented, output to diag/"""
import os, sys, torch
os.environ['DIAG_STEP'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # only for any CPU-only probes; GPU runs without it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Import base_train module and monkey-patch optimizer step + forward hooks
import importlib.util
spec = importlib.util.spec_from_file_location("base_train", "../scripts/base_train.py")
bt = importlib.util.module_from_spec(spec)
# We won't fully exec; instead run via subprocess with env so logs go to diag/
print("[diag] run via: python ../scripts/base_train.py --run followup-gated-M-diag --seed 43 --batch_size 32768 --compute_dtype bf16 --rda --gated_write --num_iterations 40 --output_dir diag/followup-gated-M-diag")
