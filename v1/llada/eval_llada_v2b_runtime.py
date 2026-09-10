# Apply the m4-rotated-k-v2b monkey patches before loading eval_llada.
import rotated_k_v2b_runtime_patch

import runpy
runpy.run_path("eval_llada.py", run_name="__main__")
