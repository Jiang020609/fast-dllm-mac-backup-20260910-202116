import runpy
import sys
from pathlib import Path

import torch


EVAL_PATH = Path.home() / "Fast-dLLM" / "v1" / "llada" / "eval_llada.py"

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS is unavailable")

# 让 eval_llada.py 看到正确的程序名和原有命令行参数。
sys.argv[0] = str(EVAL_PATH)

# 不开启 wait_until_completed，避免为了 trace 强制每个 GPU operation
# 同步完成，从而明显扭曲真实性能。
with torch.mps.profiler.profile(
    mode="interval,event",
    wait_until_completed=False,
):
    runpy.run_path(
        str(EVAL_PATH),
        run_name="__main__",
    )
