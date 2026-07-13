# ============================================================
# FUSION 2.0 — ESCF ENGINE
# Epistemic-State Conditioned Fusion: core classifier
# ============================================================
#
# PURPOSE:
#   Analyzes the "epistemic state" of model answers to decide which
#   fusion strategy (majority vote, debate, confidence-weighting, etc.)
#   will produce the best combined answer.
#
# INTEGRATION WITH FUSION PIPELINE (main.py → fusion2.py → escf_engine):
#
#   1. main.py:/ask
#      ├─→ question → dispatcher.dispatch_parallel()
#      └─→ [models run in parallel] → answers[] + confidences[]
#
#   2. main.py → fusion2.py:fuse_answers()
#      ├─→ answers[] + confidences[] + category
#      └─→ calls escf_engine.classify_answers()
#
#   3. escf_engine.classify_answers()
#      ├─→ compute_semantic_entropy(answers)   // Do models DISAGREE?
#      ├─→ compute_confidence_spread(confidences) // Is trust SPLIT?
#      └─→ 2×2 matrix → epistemic state
#
#   4. fusion2.py uses epistemic state to SELECT fusion strategy:
#      ├─→ CONSENSUS          → majority_vote()
#      ├─→ CONFIDENT_DISSENTER → trust highest-confidence model
#      ├─→ COLLECTIVE_DOUBT    → debate_merge() (extract facts, synthesize)
#      └─→ EPISTEMIC_VOID      → fallback to best model + flag for review
#
#   5. main.py returns FusionResponse with:
#      ├─→ final_answer (from selected fusion strategy)
#      ├─→ epistemic_state (tells client WHY this strategy was chosen)
#      └─→ fusion_notes (e.g., "Confident dissenter" or "Disagreement flagged")
#
# THE 2×2 EPISTEMIC MATRIX:
#
#   Four states based on (semantic entropy, confidence spread):
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  ENTROPY  │  SPREAD  │  STATE             │  STRATEGY    │
#   ├─────────────────────────────────────────────────────────┤
#   │  LOW      │  LOW     │ CONSENSUS          │ Majority vote│
#   │           │          │ (models agree      │              │
#   │           │          │  + confident)      │              │
#   ├─────────────────────────────────────────────────────────┤
#   │  LOW      │  HIGH    │ CONFIDENT_DISSENTER│ Trust best   │
#   │           │          │ (models agree BUT  │ confidence   │
#   │           │          │  one far more      │              │
#   │           │          │  confident)        │              │
#   ├─────────────────────────────────────────────────────────┤
#   │  HIGH     │  LOW     │ COLLECTIVE_DOUBT   │ Debate merge │
#   │           │          │ (models disagree   │ (extract &   │
#   │           │          │  but equally       │ synthesize)  │
#   │           │          │  uncertain)        │              │
#   ├─────────────────────────────────────────────────────────┤
#   │  HIGH     │  HIGH    │ EPISTEMIC_VOID     │ Fallback +   │
#   │           │          │ (total chaos:      │ flag for     │
#   │           │          │  disagree + split  │ human review │
#   │           │          │  trust)            │              │
#   └─────────────────────────────────────────────────────────┘
#
# WHY NOT USE ESCF FOR MATH/CODING/CREATIVE?
#   Math and coding have *objectively correct* answers — semantic
#   disagreement isn't "in doubt", it's simply wrong vs. right.
#   These categories use specialized strategies (step extraction,
#   syntax validation) instead. ESCF is disabled for these via
#   BYPASS_CATEGORIES, letting fusion2.py use domain-specific logic.
#
# ============================================================

import re
from math import log2
from typing import List, Dict, Any, Optional

# ── THRESHOLDS ────────────────────────────────────────────────
ENTROPY_THRESHOLD = 0.60            # semantic entropy >= this -> "high" (models diverge)
SPREAD_THRESHOLD = 0.35             # confidence spread >= this -> "high" (trust is split)
CONFIDENT_TIEBREAKER = 0.90         # any single model >= this overrides quadrant ambiguity
CLUSTER_SIMILARITY_THRESHOLD = 0.40  # min Dice similarity to merge two answers into one cluster

# Categories where ESCF classification does not apply.
BYPASS_CATEGORIES = {"math", "coding", "creative", "procedural"}

EPISTEMIC_STATES = {
    "CONSENSUS",
    "COLLECTIVE_DOUBT",
    "CONFIDENT_DISSENTER",
    "EPISTEMIC_VOID",
}


# ── TEXT NORMALIZATION ──────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "or", "but", "if", "because", "as",
    "until", "while", "this", "that", "these", "those", "it", "its",
    "it's", "they", "them", "their", "which", "who", "whom", "what",
    "where", "when", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "i", "we", "you", "he", "she", "do",
    "does", "did", "having", "has", "have", "had", "also", "may", "might",
    "must", "shall", "would", "could", "being",
}

# Domain-specific synonym normalization so semantically equivalent
# AI/ML vocabulary doesn't get treated as unrelated tokens (e.g. "ML"
# vs "machine learning", "AI" vs "artificial intelligence").
_AI_ML_SYNONYMS = {
    "ml": "machinelearning",
    "machine": "machinelearning",
    "learning": "machinelearning",
    "learn": "machinelearning",
    "ai": "artificialintelligence",
    "artificial": "artificialintelligence",
    "intelligence": "artificialintelligence",
    "algorithms": "algorithm",
    "algorithmic": "algorithm",
    "statistics": "statistical",
    "statistically": "statistical",
    "programmed": "program",
    "programming": "program",
    "systems": "system",
}

# Ordered longest-suffix-first so e.g. "ational" is tried before "s".
_SUFFIXES = (
    "ational", "tional", "alize", "icate", "iciti", "ative", "ical",
    "ness", "ment", "able", "ible", "ing", "ies", "ied", "ion", "ers",
    "er", "ed", "ly", "es", "s",
)


def _stem(token: str) -> str:
    """
    Lightweight suffix-stripping stemmer. Intentionally not a full
    Porter implementation — ESCF has no external data downloads
    (unlike e.g. nltk's stopword/stemmer corpora), so this stays
    dependency-free and deterministic.
    """
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for tok in text.split():
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        tok = _AI_ML_SYNONYMS.get(tok, tok)
        tok = _stem(tok)
        if tok:
            tokens.append(tok)
    return tokens


def normalized_token_set(text: str) -> set:
    return set(_tokenize(text))


# ── SEMANTIC SIMILARITY (Dice coefficient) ──────────────────────

def semantic_similarity(a: str, b: str) -> float:
    """
    Dice coefficient over stemmed, stopword-filtered, AI/ML-normalized
    token SETS (not character n-grams). Replaces the old SequenceMatcher
    character-diff approach used in fusion1, which scores two paraphrases
    with different wording as dissimilar even when they mean the same
    thing (see dry.py's bonus semantic-similarity check).

    0.0 = no shared meaning-bearing tokens. 1.0 = identical token sets.
    """
    set_a = normalized_token_set(a)
    set_b = normalized_token_set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return round((2.0 * intersection) / (len(set_a) + len(set_b)), 4)


# ── CLUSTERING & SEMANTIC ENTROPY ────────────────────────────────

def _cluster_answers(
    answers: List[str],
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
) -> List[List[int]]:
    """
    Greedy single-linkage clustering: an answer joins the first existing
    cluster where it is similar enough to ANY current member; otherwise
    it starts a new cluster. Order-dependent, but stable enough for the
    small per-query answer counts (3-5 models) ESCF operates on.
    """
    clusters: List[List[int]] = []
    for i, ans in enumerate(answers):
        placed = False
        for cluster in clusters:
            if any(
                semantic_similarity(ans, answers[j]) >= threshold
                for j in cluster
            ):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def compute_semantic_entropy(answers: List[str]) -> float:
    """
    Shannon entropy over the cluster-size distribution, normalized to
    [0, 1] by dividing by log2(n).

        0.0 -> every answer clusters together (full agreement)
        1.0 -> every answer is its own cluster (total divergence)
    """
    n = len(answers)
    if n <= 1:
        return 0.0

    clusters = _cluster_answers(answers)
    sizes = [len(c) for c in clusters]
    total = sum(sizes)

    raw_entropy = 0.0
    for size in sizes:
        p = size / total
        raw_entropy -= p * log2(p)

    max_entropy = log2(n)
    if max_entropy == 0:
        return 0.0

    return round(raw_entropy / max_entropy, 4)


def compute_confidence_spread(confidences: List[float]) -> float:
    """
    Max-min spread of self-reported confidence scores.

    Chosen over standard deviation deliberately: std-dev under-reacts to
    a single outlier confidence in a small sample (n=3-5), and detecting
    exactly that outlier is what CONFIDENT_DISSENTER needs to catch
    (one model at 0.97 against two guessers at ~0.20 must register as
    "high spread", which max-min captures cleanly).
    """
    if not confidences:
        return 0.0
    return round(max(confidences) - min(confidences), 4)


# ── MAIN CLASSIFIER ───────────────────────────────────────────────

def classify_epistemic_state(
    answers: List[str],
    confidences: List[float],
    category: str = "factual",
) -> Optional[Dict[str, Any]]:
    """
    Classifies the epistemic state of a set of model answers.

    Returns None when `category` is in BYPASS_CATEGORIES — callers
    (fusion2.py) should skip ESCF entirely and fall through to that
    category's dedicated strategy.

    Returns, otherwise:
        {
            "epistemic_state": one of EPISTEMIC_STATES,
            "escf_metrics": {
                "semantic_entropy": float,
                "confidence_spread": float,
            },
        }

    Classification order:
      1. Category bypass check.
      2. High-confidence tiebreaker: if any single model reports
         confidence >= CONFIDENT_TIEBREAKER, classify as
         CONFIDENT_DISSENTER regardless of the raw entropy signal.
         Rationale: a model that is 0.97 confident and correct (e.g.
         "Canberra") can still register high semantic entropy against
         two *differently* wrong answers ("Sydney" vs "Melbourne") —
         without the tiebreaker that would misclassify as
         EPISTEMIC_VOID. A single strong trust signal is a
         fundamentally different situation from genuine collective
         uncertainty, so it takes priority over the entropy reading.
      3. Otherwise, standard 2x2 quadrant lookup on
         (entropy >= ENTROPY_THRESHOLD, spread >= SPREAD_THRESHOLD).
    """
    if category in BYPASS_CATEGORIES:
        return None

    if len(answers) < 2 or len(confidences) < 2:
        # Nothing to classify epistemic disagreement over with <2 answers.
        return {
            "epistemic_state": "CONSENSUS",
            "escf_metrics": {"semantic_entropy": 0.0, "confidence_spread": 0.0},
        }

    entropy = compute_semantic_entropy(answers)
    spread = compute_confidence_spread(confidences)

    if max(confidences) >= CONFIDENT_TIEBREAKER:
        state = "CONFIDENT_DISSENTER"
    else:
        high_entropy = entropy >= ENTROPY_THRESHOLD
        high_spread = spread >= SPREAD_THRESHOLD

        if not high_entropy and not high_spread:
            state = "CONSENSUS"
        elif high_entropy and not high_spread:
            state = "COLLECTIVE_DOUBT"
        elif not high_entropy and high_spread:
            state = "CONFIDENT_DISSENTER"
        else:
            state = "EPISTEMIC_VOID"

    return {
        "epistemic_state": state,
        "escf_metrics": {
            "semantic_entropy": entropy,
            "confidence_spread": spread,
        },
    }