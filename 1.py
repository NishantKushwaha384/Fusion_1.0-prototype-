# run_fusion_eval.py
import fusion_lm_eval_adapter  # registers "fusion_adapter" with lm-eval
from lm_eval.__main__ import cli_evaluate
import sys

sys.argv = [
    "lm_eval",
    "--model", "fusion_adapter",
    "--tasks", "mmlu_generative",
    "--batch_size", "1",
    "--limit", "50",
]

cli_evaluate()