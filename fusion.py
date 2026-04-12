# ============================================================
# FUSION 1.0 — FUSION ENGINE
# ============================================================
# This is the most important file in the entire project.
# This is where YOUR original algorithm lives.
#
# Current state: PLACEHOLDER implementations
# These are starting points — not the real algorithm.
# Your job is to discover and replace these with real logic.
#
# What fusion means:
#   You have 2-3 answers from different models.
#   You need to combine them into ONE answer that is
#   BETTER than any individual answer.
#
# The strategies in this file (in order of sophistication):
#   1. single()             — only one model, no fusion needed
#   2. majority_vote()      — most common facts win (factual questions)
#   3. confidence_weighted()— higher confidence = more influence
#   4. creative_blend()     — blend different creative styles
#   5. debate_merge()       — synthesize multiple analytical views
#
# ─────────────────────────────────────────────────────────────
# HOW TO DEVELOP YOUR REAL FUSION ALGORITHM:
#
# Step 1: Run the system for 1 week, collect 100+ logs
# Step 2: Open logger.txt
# Step 3: For each entry, manually read all 3 model answers
# Step 4: Write your own "ideal" merged answer for each
# Step 5: Compare your manual merge to what fusion.py produced
# Step 6: Find the patterns where yours is better
# Step 7: Translate those patterns into code here
# Step 8: Measure if the new code produces answers closer to
#         your manually-merged versions
# That process = your research contribution
# ─────────────────────────────────────────────────────────────

import re
from typing import List


async def fuse_answers(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str
) -> dict:
    """
    Main fusion function — routes to the right strategy.

    Args:
        question:    Original user question
        answers:     List of answers from each model
        confidences: Confidence score per model (0.0 - 1.0)
        category:    Question type from classifier

    Returns:
        dict with:
            answer:   The final fused answer string
            weights:  How much each model contributed (list of floats)
            strategy: Which strategy was used
            notes:    Any observations about the fusion
    """

    # ── EDGE CASE: Only one answer ───────────────────────────
    # If only one model responded (others failed or routing chose 1),
    # there's nothing to fuse — just return that single answer.
    if len(answers) == 1:
        return await _single(answers, confidences)

    # ── EDGE CASE: No answers at all ─────────────────────────
    if len(answers) == 0:
        return {
            "answer":   "No models were able to respond. Please try again.",
            "weights":  [],
            "strategy": "none",
            "notes":    "All models failed"
        }

    # ── ROUTE TO STRATEGY ────────────────────────────────────
    # Different question types benefit from different fusion strategies.
    # This routing mirrors the ROUTING_TABLE in dispatcher.py
    #
    # TODO: After your research, you may discover that category-based
    # routing isn't the best approach. Maybe confidence-weighted works
    # best for ALL categories. Test it. Measure it. That's a finding.

    strategy_map = {
        "factual":   _majority_vote,
        "math":      _confidence_weighted,
        "coding":    _confidence_weighted,
        "creative":  _creative_blend,
        "reasoning": _debate_merge,
        "general":   _confidence_weighted,   # Default fallback
    }

    strategy_fn = strategy_map.get(category, _confidence_weighted)
    return await strategy_fn(question, answers, confidences)


# ════════════════════════════════════════════════════════════
# FUSION STRATEGIES
# Each strategy below is a different way to combine answers.
# All return the same format:
# { "answer": str, "weights": list, "strategy": str, "notes": str }
# ════════════════════════════════════════════════════════════

async def _single(answers: List[str], confidences: List[float]) -> dict:
    """
    No fusion — only one answer available.
    Used when: routing chose only 1 model, or others failed.
    """
    return {
        "answer":   answers[0],
        "weights":  [1.0],
        "strategy": "single",
        "notes":    "Only one model responded — no fusion applied"
    }


async def _majority_vote(question: str, answers: List[str], confidences: List[float]) -> dict:
    """
    MAJORITY VOTE STRATEGY
    Used for: Factual questions (history, science, facts)

    Logic:
    - Intelligently merges all factual answers into ONE comprehensive answer
    - Uses highest-confidence answer as base
    - Enriches it with unique insights from other answers
    - Creates a unified, coherent response
    """

    # ── MERGE FACTUAL ANSWERS INTO ONE ───────────────────────
    # Start with the highest confidence answer as the base
    total_conf = sum(confidences) or 1
    weights = [c / total_conf for c in confidences]
    
    sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    
    # Start with highest confidence answer
    fused = answers[sorted_idx[0]]
    
    # Extract unique points from other answers and add them
    # This creates a richer, more complete answer
    for idx in sorted_idx[1:]:
        other_answer = answers[idx]
        # If this answer is significantly different, append key points
        if len(other_answer) > 50 and other_answer not in fused:
            # Add complementary information from other models
            additional_insights = other_answer.split('\n')[0]  # Get first paragraph
            if additional_insights not in fused:
                fused += f" {additional_insights}"
    
    return {
        "answer":   fused.strip(),
        "weights":  weights,
        "strategy": "majority_vote",
        "notes":    f"Merged {len(answers)} factual answers (confidence: {max(confidences):.2f})"
    }


async def _confidence_weighted(question: str, answers: List[str], confidences: List[float]) -> dict:
    """
    CONFIDENCE-WEIGHTED FUSION
    Used for: Math, coding, general questions

    Logic:
    - Calculate weight for each model = its confidence / total confidence
    - Higher confidence = more influence on final answer
    - Blends multiple complete answers while respecting confidence scores

    Enhanced for full-length answers:
    - Presents all perspectives with their confidence weights
    - Highlights the most confident answer
    - Preserves detail from all models
    """

    # ── CALCULATE WEIGHTS ────────────────────────────────────
    total_confidence = sum(confidences)
    if total_confidence == 0:
        # All models have 0 confidence — equal weights
        weights = [1.0 / len(answers)] * len(answers)
    else:
        weights = [c / total_confidence for c in confidences]

    # ── BUILD FUSED ANSWER ───────────────────────────────────
    # Merge all answers intelligently: start with highest confidence,
    # enriched with unique insights from other answers
    
    sorted_idx = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    
    # Start with highest confidence answer as base
    fused = answers[sorted_idx[0]]
    
    # Add unique insights from other answers
    for i in sorted_idx[1:]:
        other_answer = answers[i]
        # If the other answer adds unique value, incorporate it
        if len(other_answer) > 100 and other_answer not in fused:
            # Extract unique parts - take last paragraph as it usually contains unique insights
            lines = other_answer.strip().split('\n')
            unique_parts = [line for line in lines if line.strip() and line not in fused]
            if unique_parts:
                fused += "\n\n" + "\n".join(unique_parts[:3])  # Add up to 3 lines of new info

    return {
        "answer":   fused.strip(),
        "weights":  weights,
        "strategy": "confidence_weighted",
        "notes":    f"Merged {len(answers)} answers with confidence weighting (highest: {max(confidences):.2f})"
    }


async def _creative_blend(question: str, answers: List[str], confidences: List[float]) -> dict:
    """
    CREATIVE BLEND STRATEGY
    Used for: Creative writing, poems, stories, brainstorming

    Logic:
    - Intelligently merges multiple creative responses into ONE unified piece
    - Takes the best creative elements from each model
    - Creates a richer, more sophisticated creative work
    """

    # ── BLEND CREATIVE ANSWERS ───────────────────────────────
    # Start with the best creative response and enhance it with
    # unique creative elements from other models
    
    total_conf = sum(confidences) or 1
    weights = [c / total_conf for c in confidences]
    
    # Sort by confidence - highest confidence gets the primary voice
    sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    
    # Start with highest confidence creative answer
    fused = answers[sorted_idx[0]]
    
    # Add unique creative elements from other answers
    for i in sorted_idx[1:]:
        other_answer = answers[i]
        if len(other_answer) > 50 and other_answer not in fused:
            # Extract unique creative phrases/sentences
            lines = other_answer.strip().split('\n')
            # Add interesting new lines that aren't already in the fused answer
            unique_lines = [l for l in lines if l.strip() and l not in fused and len(l) > 20]
            if unique_lines:
                # Add 1-2 best unique lines to enhance creativity
                fused += "\n\n" + "\n".join(unique_lines[:2])

    return {
        "answer":   fused.strip(),
        "weights":  weights,
        "strategy": "creative_blend",
        "notes":    f"Merged {len(answers)} creative responses into unified piece"
    }


async def _debate_merge(question: str, answers: List[str], confidences: List[float]) -> dict:
    """
    DEBATE MERGE STRATEGY
    Used for: Reasoning, analysis, opinion questions

    Logic:
    - Intelligently merges multiple analytical perspectives into ONE comprehensive answer
    - Takes the strongest arguments from each model
    - Creates a unified, multi-faceted response that incorporates key insights
    """

    # ── BUILD DEBATE ANSWER ──────────────────────────────────
    # Merge all perspectives intelligently: start with strongest analysis,
    # enhance with unique insights from other models
    
    total_conf = sum(confidences) or 1
    weights = [c / total_conf for c in confidences]
    
    # Sort by confidence - strongest analysis gets primary position
    sorted_idx = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    
    # Start with highest confidence reasoning
    fused = answers[sorted_idx[0]]
    
    # Add unique analytical insights from other models
    for i in sorted_idx[1:]:
        other_answer = answers[i]
        if len(other_answer) > 100 and other_answer not in fused:
            # Extract key analytical points that aren't already covered
            lines = other_answer.strip().split('\n')
            unique_insights = [l for l in lines if l.strip() and l not in fused and len(l) > 30]
            if unique_insights:
                # Add complementary insights
                fused += "\n\n" + "\n".join(unique_insights[:3])

    return {
        "answer":   fused.strip(),
        "weights":  weights,
        "strategy": "debate_merge",
        "notes":    f"Merged {len(answers)} analytical perspectives into comprehensive answer"
    }


# ════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# Helper functions used by strategies above
# ════════════════════════════════════════════════════════════

def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences.
    Used for sentence-level fusion (when you implement it).

    TODO: This naive split breaks on abbreviations like "Dr." or "U.S."
    A research finding could be: naive sentence splitting causes
    fusion errors in X% of cases. Then fix it with a proper
    sentence tokenizer (like nltk.sent_tokenize).
    """
    # Naive split on period/question mark/exclamation mark
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _calculate_overlap(text1: str, text2: str) -> float:
    """
    Calculates word overlap between two texts.
    High overlap = models are saying similar things.
    Low overlap = models are saying different things.

    Returns: float between 0.0 (no overlap) and 1.0 (identical)

    ─────────────────────────────────────────────────────────
    YOUR RESEARCH TASK:
    Test if word overlap is a good proxy for semantic similarity.
    Compare word overlap scores vs human judgments of similarity.
    If they correlate → use this function in majority voting.
    If they don't → you need embeddings-based similarity instead.
    That comparison = research finding.
    ─────────────────────────────────────────────────────────
    """
    # Remove common stop words (they inflate overlap artificially)
    stop_words = {"the", "a", "an", "is", "are", "was", "were",
                  "it", "this", "that", "to", "of", "in", "for",
                  "and", "or", "but", "not", "with", "be", "been"}

    words1 = set(text1.lower().split()) - stop_words
    words2 = set(text2.lower().split()) - stop_words

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity: intersection / union
    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)
