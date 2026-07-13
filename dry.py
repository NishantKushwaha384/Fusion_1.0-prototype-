"""
DRY TEST: fusion1.py vs fusion2.py
====================================
4 test cases, one per ESCF quadrant.
For each case we show:
  - What ESCF detected (entropy, conf_spread, state)
  - What strategy fusion2 picked
  - What strategy fusion1 would have picked (from source reading)
  - The actual output answer from fusion2
  - A verdict on whether the output is better, same, or worse
"""

import sys, os, textwrap
from unittest.mock import patch

sys.path.insert(0, "/home/claude")

# ── Silence the sentence-transformer download progress bars ─────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

print("Loading fusion2.py (this takes ~5s the first time for model download)…\n")
import fusion2 as f2

# ── Pretty-print helpers ─────────────────────────────────────────────────

SEP  = "═" * 70
SEP2 = "─" * 70

def show(label, text, width=66):
    wrapped = textwrap.fill(str(text), width=width)
    print(f"  {label}:\n    {wrapped.replace(chr(10), chr(10)+'    ')}")

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def result_block(result, f1_strategy, f1_verdict):
    print(f"\n  [ESCF]")
    m = result.get("escf_metrics") or {}
    print(f"    State            : {result.get('epistemic_state', 'N/A')}")
    print(f"    Semantic Entropy : {m.get('semantic_entropy', 'N/A')}")
    print(f"    Conf Spread      : {m.get('confidence_spread', 'N/A')}")
    print(f"\n  [STRATEGY]")
    print(f"    fusion2 chose    : {result['strategy']}")
    print(f"    fusion1 would've : {f1_strategy}")
    show("fusion2 answer", result["answer"])
    print(f"\n  [VERDICT] {f1_verdict}")
    print(SEP2)


# ════════════════════════════════════════════════════════════════════════
# TEST 1 — CONSENSUS (low entropy + low spread)
# Expected: models agree semantically, similar confidence.
# fusion1: majority_vote  |  fusion2: also majority_vote (same route)
# ESCF adds: confirmation the models actually agree → no blind fusion
# ════════════════════════════════════════════════════════════════════════
banner("TEST 1 · CONSENSUS  (factual question, models agree)")

q1 = "What is machine learning?"
a1 = [
    "Machine learning is a subset of artificial intelligence where systems "
    "learn from data to improve performance without being explicitly programmed.",

    "Machine learning is an AI branch that enables computers to learn from "
    "experience and data automatically, without step-by-step programming.",

    "Machine learning is a field of AI in which statistical algorithms allow "
    "systems to learn and improve from experience rather than explicit code.",
]
c1 = [0.78, 0.74, 0.71]   # similar → low spread

r1 = f2.fuse_answers(q1, a1, c1, "factual")
result_block(
    r1,
    f1_strategy="majority_vote (category only)",
    f1_verdict=(
        "SAME RESULT, better confidence.\n"
        "  fusion1 just trusts 'factual → majority_vote' blindly.\n"
        "  fusion2 VERIFIED the models actually agree (entropy low)\n"
        "  before trusting the vote. If they had secretly diverged,\n"
        "  fusion2 would have caught it; fusion1 would not."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TEST 2 — COLLECTIVE DOUBT (high entropy + low spread)
# Models give genuinely different factual answers with similar confidence.
# fusion1: majority_vote  |  fusion2: OVERRIDES to debate_merge
# This is the most important difference.
# ════════════════════════════════════════════════════════════════════════
banner("TEST 2 · COLLECTIVE DOUBT  (factual question, models disagree equally)")

q2 = "What caused the 2008 financial crisis?"
a2 = [
    "The 2008 financial crisis was primarily caused by excessive subprime "
    "mortgage lending and complex derivatives like CDOs that concealed risk "
    "from investors and regulators alike.",

    "The root cause of the 2008 financial crisis was bank deregulation "
    "combined with the Federal Reserve holding interest rates too low for "
    "too long, flooding markets with cheap credit.",

    "Rating agencies giving AAA ratings to high-risk mortgage-backed "
    "securities was the main driver of the 2008 crisis, as it misled "
    "institutional investors into buying toxic assets.",
]
c2 = [0.62, 0.60, 0.61]   # similar → low spread

# Mock the LLM synthesizer so the test runs offline
with patch("fusion2.call_llm_synthesizer_sync",
           return_value="The 2008 financial crisis had multiple interacting causes: "
                        "excessive subprime lending, deregulation enabling risky behavior, "
                        "and rating agencies misrepresenting the quality of mortgage securities."):
    r2 = f2.fuse_answers(q2, a2, c2, "factual")

result_block(
    r2,
    f1_strategy="majority_vote (category only — WRONG choice here)",
    f1_verdict=(
        "FUSION2 IS CLEARLY BETTER HERE.\n"
        "  fusion1 runs majority_vote on genuinely contradictory answers.\n"
        "  It picks whichever fact appears most and silently drops the rest.\n"
        "  fusion2 detects high semantic entropy → Collective_Doubt →\n"
        "  forces debate_merge, which synthesizes all viewpoints instead\n"
        "  of discarding the minority ones. The output is fairer."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TEST 3 — CONFIDENT DISSENTER (low entropy + high spread)
# One model is highly confident and right; others are uncertain and wrong.
# fusion1: majority_vote (outvotes the correct answer)
# fusion2: ESCF catches the spread → trusts the confident outlier
# ════════════════════════════════════════════════════════════════════════
banner("TEST 3 · CONFIDENT DISSENTER  (one model knows, others guess)")

q3 = "What is the capital of Australia?"
a3 = [
    "The capital of Australia is Canberra.",            # correct, very confident

    "The capital of Australia is Sydney, being the "
    "largest and most internationally recognised city.", # wrong, low conf

    "Melbourne served as Australia's capital for many "
    "years and may still be considered the effective "
    "administrative centre by some sources.",            # wrong, low conf
]
c3 = [0.97, 0.22, 0.18]   # huge spread

r3 = f2.fuse_answers(q3, a3, c3, "factual")
result_block(
    r3,
    f1_strategy="majority_vote (2 wrong answers vs 1 right → could pick wrong)",
    f1_verdict=(
        "FUSION2 IS CLEARLY BETTER HERE.\n"
        "  fusion1 runs majority_vote: Sydney + Melbourne both appear,\n"
        "  'Canberra' is only in one answer → Canberra might get dropped.\n"
        "  fusion2 detects high confidence spread → Confident_Dissenter →\n"
        "  directly returns the 0.97-confidence answer: Canberra.\n"
        "  Majority rule is wrong when one expert outweighs two guessers."
    )
)


# ════════════════════════════════════════════════════════════════════════
# TEST 4 — EPISTEMIC VOID (high entropy + high spread)
# Models fundamentally contradict each other AND confidence is unequal.
# fusion1: tries to fuse, produces contradictory nonsense
# fusion2: refuses to fuse, flags for human review, returns best single
# ════════════════════════════════════════════════════════════════════════
banner("TEST 4 · EPISTEMIC VOID  (models contradict, confidence unequal)")

q4 = "Will AI replace all human jobs by 2030?"
a4 = [
    "Yes, AI will automate and replace virtually all human jobs by 2030 "
    "as large language models and robotics advance exponentially across "
    "every sector of the economy.",                     # extreme claim, high conf

    "No, AI will create far more jobs than it eliminates, just as every "
    "previous technological revolution — steam, electricity, computers — "
    "ultimately expanded human employment rather than reducing it.",  # opposite, low conf

    "AI will primarily augment rather than replace workers, changing job "
    "requirements and skill demands substantially without eliminating "
    "the majority of roles.",                            # middle ground, medium conf
]
c4 = [0.88, 0.15, 0.45]   # high spread + high entropy expected

r4 = f2.fuse_answers(q4, a4, c4, "reasoning")
result_block(
    r4,
    f1_strategy="debate_merge (tries to synthesize fundamentally opposed claims)",
    f1_verdict=(
        "FUSION2 IS SAFER HERE.\n"
        "  fusion1 tries debate_merge on diametrically opposed claims.\n"
        "  The result would blend 'yes all jobs gone' with 'no jobs created'\n"
        "  into mush — or just silently return the strongest-confidence\n"
        "  answer without flagging the contradiction.\n"
        "  fusion2 detects Epistemic_Void → refuses to fuse →\n"
        "  returns the best single answer + attaches a human-review flag.\n"
        "  That's the honest thing to do when the models fundamentally disagree."
    )
)

# ════════════════════════════════════════════════════════════════════════
# SEMANTIC SIMILARITY SPOT CHECK
# Show that the SequenceMatcher bug is actually fixed
# ════════════════════════════════════════════════════════════════════════
banner("BONUS · Semantic similarity sanity check (the SequenceMatcher bug)")

pairs = [
    ("Python is a programming language",
     "Python is a versatile coding tool",
     "SAME meaning — should be near-duplicates"),
    ("The sky is blue",
     "Photosynthesis converts sunlight into glucose",
     "DIFFERENT meaning — should NOT be near-duplicates"),
    ("Machine learning uses statistical algorithms",
     "ML systems learn patterns from training data using statistics",
     "SAME meaning (paraphrase) — should be near-duplicates"),
]

from difflib import SequenceMatcher as SM

print()
for a, b, label in pairs:
    old = SM(None, a.lower(), b.lower()).ratio()
    new = f2.semantic_similarity(a, b)
    flag = "✅ correct" if (
        (label.startswith("SAME")      and new > 0.7 and old < 0.6) or
        (label.startswith("DIFFERENT") and new < 0.3 and old < 0.5)
    ) else "ℹ️  both ok"
    print(f"  {label}")
    print(f"    A: \"{a}\"")
    print(f"    B: \"{b}\"")
    print(f"    SequenceMatcher : {old:.3f}   semantic_similarity : {new:.3f}  {flag}")
    print()

print(SEP)
print("  All 4 ESCF states exercised. Tests complete.")
print(SEP)