# ============================================================
# FUSION 1.0 — QUESTION CLASSIFIER
# ============================================================
# This file figures out WHAT TYPE of question the user asked.
#
# Why we need this:
#   Different questions need different models and strategies.
#   A math question should go to models good at math.
#   A creative question should go to models good at creativity.
#   This classifier is the "brain" that decides the routing.
#
# How it works:
#   Sends the question to the CHEAPEST/FASTEST model (Groq)
#   with a very specific prompt that forces it to return JSON.
#   We parse that JSON to get: category, complexity, confidence.
#
# YOUR RESEARCH TASK:
#   After 2 weeks of running this, collect all questions where
#   confidence < 0.75. These are your "hard to classify" cases.
#   Study them. They reveal the boundaries of your categories.
#   That study leads to improving this classifier — which is
#   an original contribution you can document in your report.
# ============================================================

import json
import os
from groq import AsyncGroq          # Groq = free, fastest model API
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# ── GROQ CLIENT ─────────────────────────────────────────────
# We use Groq for classification because:
# 1. It's FREE (no cost limit)
# 2. It's the FASTEST (0.3s response time)
# 3. Classification is a simple task — doesn't need a big model
# 4. Why waste expensive API credits on a simple routing decision?
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# ── CATEGORY DEFINITIONS ────────────────────────────────────
# These are the question types Fusion 1.0 understands.
# Each maps to a different routing strategy in dispatcher.py
#
# TODO (your research task):
#   After running the system for a week, check if these
#   categories are sufficient. You may discover that "reasoning"
#   splits into "logical reasoning" vs "ethical reasoning"
#   for example. That discovery = original contribution.

CATEGORIES = {
    "math":      "Mathematical calculations, equations, proofs, statistics",
    "coding":    "Writing code, debugging, explaining algorithms",
    "factual":   "Facts, history, science, definitions, general knowledge",
    "creative":  "Stories, poems, brainstorming, creative writing",
    "reasoning": "Logic puzzles, analysis, 'why' questions, opinion",
    "general":   "Casual questions, greetings, simple everyday queries"
}

# ── CLASSIFIER PROMPT ────────────────────────────────────────
# This is the most important part of the classifier.
# A good prompt = reliable classification.
# A bad prompt = random garbage JSON = broken routing.
#
# Key techniques used here:
# 1. "Return ONLY JSON" — forces structured output
# 2. Few-shot examples — shows the model exactly what we want
# 3. Exact format specified — no ambiguity
# 4. Confidence score — lets us know when the model is unsure
#
# YOUR IMPROVEMENT TASK:
#   This prompt is a starting point. After you find misclassified
#   questions, add them as examples here to fix the classifier.
#   Each fix is a documented iteration = research contribution.

CLASSIFIER_SYSTEM_PROMPT = """You are a question classifier for an AI routing system.

Analyze the user's question and classify it. Return ONLY valid JSON — no explanation, no markdown, no extra text.

Output format:
{"category": "math", "complexity": "high", "confidence": 0.92, "reasoning": "one sentence why"}

Categories (pick exactly one):
- math: equations, calculations, proofs, statistics, algebra, geometry
- coding: write code, debug code, explain code, algorithms, data structures
- factual: facts, history, science, geography, definitions, "what is X"
- creative: stories, poems, essays, brainstorming, "write me a..."
- reasoning: logic puzzles, analysis, ethics, "why", "should I", opinions
- general: casual chat, greetings, simple everyday questions

Complexity (pick one):
- low: answerable in 1-2 sentences, basic concept
- medium: needs some explanation, moderate depth
- high: complex, multi-part, requires deep expertise

Confidence: float between 0.0 (very unsure) and 1.0 (very sure)

Examples:
Q: "What is the derivative of x squared?" 
A: {"category":"math","complexity":"low","confidence":0.99,"reasoning":"Straightforward calculus derivative question"}

Q: "Write a Python function to sort a list by second element of tuples"
A: {"category":"coding","complexity":"medium","confidence":0.97,"reasoning":"Code writing task with specific requirements"}

Q: "Why did World War 2 start?"
A: {"category":"reasoning","complexity":"high","confidence":0.85,"reasoning":"Historical causation requires analytical reasoning"}

Q: "Write a short story about a robot learning to paint"
A: {"category":"creative","complexity":"medium","confidence":0.96,"reasoning":"Creative writing task with clear parameters"}"""


async def classify_question(question: str) -> dict:
    """
    Classifies the user's question into a category.

    Args:
        question: The raw user question string

    Returns:
        dict with keys: category, complexity, confidence, reasoning

    Example return value:
        {
            "category": "math",
            "complexity": "high",
            "confidence": 0.92,
            "reasoning": "This is a calculus problem"
        }

    ─────────────────────────────────────────────────────────
    FLOW:
    1. Send question to Groq (fast, free)
    2. Parse the JSON response
    3. Validate the response has all required fields
    4. If confidence < 0.75, run validation (see below)
    5. Return the classification
    ─────────────────────────────────────────────────────────
    """

    try:
        # ── FIRST CLASSIFICATION ATTEMPT ───────────────────
        result = await _call_classifier(question)

        # ── CONFIDENCE VALIDATION ───────────────────────────
        # This is YOUR innovation over a simple classifier:
        # If confidence is low, we ask a second time to verify.
        # If both agree → we trust it even with low confidence
        # If they disagree → question is ambiguous → use "general"
        #
        # TODO: After you discover more edge cases, you can add
        # a third model as a tiebreaker here. That's a research finding.

        if result["confidence"] < 0.75:
            print(f"[CLASSIFIER] Low confidence ({result['confidence']}), running validation...")
            result = await _validate_classification(question, result)

        return result

    except Exception as e:
        # If classifier completely fails, default to "general"
        # This ensures the system never crashes — just degrades gracefully
        print(f"[CLASSIFIER ERROR] {str(e)} — defaulting to general/medium")
        return {
            "category": "general",
            "complexity": "medium",
            "confidence": 0.5,
            "reasoning": f"Classifier failed: {str(e)}"
        }


async def _call_classifier(question: str) -> dict:
    """
    Internal function — makes the actual Groq API call.
    Separated from classify_question() to keep code clean.
    """

    response = await groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",     # Groq's free Openai model
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this question: {question}"}
        ],
        temperature=0.1,            # Low temp = more consistent/deterministic output
        max_tokens=150,             # Classification needs very few tokens
    )

    raw_text = response.choices[0].message.content.strip()

    # Parse JSON — handle cases where model adds extra text despite instructions
    try:
        # Sometimes models wrap JSON in ```json ... ``` even when told not to
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        result = json.loads(raw_text)

    except json.JSONDecodeError:
        # If JSON parsing fails completely, try to extract category at minimum
        print(f"[CLASSIFIER] JSON parse failed on: {raw_text}")
        # Fallback: scan for category keywords in raw text
        result = _extract_category_from_text(raw_text)

    # Ensure all required fields exist with defaults
    return {
        "category":   result.get("category", "general"),
        "complexity":  result.get("complexity", "medium"),
        "confidence": float(result.get("confidence", 0.7)),
        "reasoning":  result.get("reasoning", "No reasoning provided")
    }


async def _validate_classification(question: str, first_result: dict) -> dict:
    """
    Called when first classification has low confidence.
    Runs classification again to check if result is consistent.

    ─────────────────────────────────────────────────────────
    THIS IS YOUR ORIGINAL ALGORITHM — document it in your report:

    "When the classifier returns confidence < 0.75, Fusion 1.0
    runs a second classification pass. If both passes agree on
    the category, confidence is bumped to 0.82 (validated).
    If they disagree, the question is flagged as ambiguous and
    routed through the general strategy which uses all models."
    ─────────────────────────────────────────────────────────
    """

    try:
        second_result = await _call_classifier(question)

        if second_result["category"] == first_result["category"]:
            # Both agree — increase confidence slightly
            first_result["confidence"] = max(first_result["confidence"], 0.82)
            first_result["reasoning"] += " [validated by second pass]"
            return first_result
        else:
            # They disagree — this is a boundary/ambiguous question
            # Route as general → all models get called → safest bet
            print(f"[CLASSIFIER] Disagreement: {first_result['category']} vs {second_result['category']} — using general")
            return {
                "category": "general",
                "complexity": first_result.get("complexity", "medium"),
                "confidence": 0.60,
                "reasoning": f"Ambiguous: could be {first_result['category']} or {second_result['category']}"
            }

    except Exception:
        # Validation failed — just return the original result
        return first_result


def _extract_category_from_text(text: str) -> dict:
    """
    Fallback parser when JSON completely fails.
    Scans raw text for category keywords.
    Not elegant but prevents total failure.
    """
    text_lower = text.lower()

    for category in CATEGORIES.keys():
        if category in text_lower:
            return {
                "category": category,
                "complexity": "medium",
                "confidence": 0.6,
                "reasoning": "Extracted from malformed response"
            }

    # Nothing found — default
    return {
        "category": "general",
        "complexity": "medium",
        "confidence": 0.5,
        "reasoning": "Could not extract from response"
    }
