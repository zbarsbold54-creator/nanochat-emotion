import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Keep this diagnostic CPU-only even when invoked from a CUDA-capable host.
torch.cuda.is_available = lambda: False

import diagnostics.test_no_label_leak as check

_GPTWithRDA = check.GPTWithRDA


def _gated(*args, **kwargs):
    kwargs["gated_write"] = True
    return _GPTWithRDA(*args, **kwargs)


check.GPTWithRDA = _gated


if __name__ == "__main__":
    raise SystemExit(check.main())
