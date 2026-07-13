"""
fusion_lm_eval_adapter.py
==========================
Custom lm-eval-harness model wrapper for the Fusion 2.0 / ESCF pipeline.

Why this exists:
lm-eval's built-in MMLU task scores answers via token loglikelihoods
(logprob of "A" vs "B" vs "C" vs "D"). Fusion doesn't expose logprobs —
it dispatches to multiple providers and returns fused free text. So we
implement the `generate_until` interface instead, and rely on an MMLU
task variant configured with `output_type: generate_until` (you'll need
to point --tasks at that variant, or write a small custom task YAML —
see notes at the bottom of this file).

Usage:
    lm_eval --model fusion_adapter \
        --tasks mmlu_generative \
        --batch_size 1

Register the model with lm-eval via the @register_model decorator below,
and make sure this file is importable (e.g. installed alongside lm-eval
or added to PYTHONPATH) before running.
"""

import asyncio
import re
from typing import List

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import sys
sys.path.insert(0, "C:\\Users\\L E N O V O\\OneDrive\\Desktop\\Programming\\python\\Fusion_1.0\\OG\\main.py")  # <-- point this at main.py's dir

from classifier import classify_question
from dispatcher import dispatch_parallel
from fusion2 import fuse_answers  # or: from fusion1 import fuse_answers


LETTER_RE = re.compile(r"\b([ABCD])\b")
print("program started")

@register_model("fusion_adapter")
class FusionLM(LM):
    """
    Wraps the Fusion pipeline (classify -> dispatch_parallel -> fuse_answers)
    behind lm-eval's generate_until interface.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # kwargs come from --model_args, e.g. category_override=factual
        self.category_override = kwargs.get("category_override")

    def generate_until(self, requests) -> List[str]:
        """
        requests: list of Instance objects, each with .args = (context, gen_kwargs)
        Returns: list of generated strings, one per request, in order.
        """
        return asyncio.run(self._generate_all(requests))

    async def _generate_all(self, requests) -> List[str]:
        outputs = []
        for req in requests:
            context, gen_kwargs = req.args
            outputs.append(await self._run_one(context))
        return outputs

    async def _run_one(self, prompt: str) -> str:
        # MMLU few-shot prompts already contain the question + choices.
        if self.category_override:
            classification = {"category": self.category_override, "complexity": "medium"}
        else:
            # Use the real classifier — category will vary per question
            # (factual, reasoning, etc.) instead of being forced.
            classification = await classify_question(prompt)

        dispatch_result = await dispatch_parallel(
            question=prompt,
            category=classification["category"],
            complexity=classification["complexity"],
        )

        fusion_result = fuse_answers(
            question=prompt,
            answers=dispatch_result["answers"],
            confidences=dispatch_result["confidence_scores"],
            category=classification["category"],
        )

        return self._extract_letter(fusion_result["answer"])

    def _extract_letter(self, text: str) -> str:
        """
        MMLU scoring (generative variant) expects a bare letter.
        Fusion's fused answer is prose, so pull the first standalone
        A/B/C/D out of it. Falls back to "A" if nothing matches
        (counts as wrong rather than crashing the eval run).
        """
        match = LETTER_RE.search(text)
        return match.group(1) if match else "A"

    # ── Required by the LM interface but unused for generate_until tasks ──
    def loglikelihood(self, requests):
        raise NotImplementedError(
            "FusionLM does not expose logprobs — use a generate_until "
            "MMLU task variant, not the default loglikelihood-based one."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Not supported by FusionLM.")


# ─────────────────────────────────────────────────────────────────────────
# NOTES
# ─────────────────────────────────────────────────────────────────────────
# 1. You need an MMLU task config with `output_type: generate_until` and a
#    prompt that explicitly asks for "Answer with a single letter (A, B, C,
#    or D)." Standard lm-eval MMLU ships as loglikelihood-based; check if a
#    generative variant exists in your lm-eval version, or copy mmlu.yaml
#    and change output_type + doc_to_text to request a letter explicitly.
#
# 2. Cost/latency: full MMLU is ~14,000 questions across 57 subjects. Each
#    question triggers a full Fusion dispatch (3-4 provider calls +
#    ESCF/fusion overhead). At even 2s/question that's ~8 hours serial,
#    and you'll hit provider rate limits fast. Recommended: run a subset
#    first, e.g. --tasks mmlu_abstract_algebra or --limit 50.
#
# 3. ESCF-specific consideration: MMLU questions are single-answer factual
#    questions, so they'll mostly land in Consensus or Confident_Dissenter
#    states rather than Collective_Doubt/Epistemic_Void — those states are
#    more about open-ended reasoning questions like "what caused the 2008
#    crisis," not "what is the derivative of x^2." You may see less ESCF
#    signal on MMLU than on your dry.py test cases. If you want an eval
#    that actually stresses ESCF's disagreement-handling, MMLU may not be
#    the most informative benchmark — something with more ambiguous or
#    contested questions would show off Collective_Doubt/Epistemic_Void
#    handling better.
#
# 4. Run command once wired up:
#      lm_eval --model fusion_adapter \
#          --tasks mmlu_generative \
#          --batch_size 1 \
#          --limit 100
print("program finished")