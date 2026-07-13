# FUSION 1.0 — QUESTION CLASSIFIER (fixed)

import json
import re
from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_CLASSIFIER_MODEL, CATEGORIES

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

CLASSIFIER_SYSTEM_PROMPT = """You are a question classifier for an AI routing system.

Analyze the user's question and classify it. Return ONLY valid JSON — no explanation, no markdown, no extra text.

Output format:
{"category": "math", "complexity": "high", "confidence": 0.92, "reasoning": "one sentence why"}

Categories (pick exactly one):
- math: equations, calculations, proofs, statistics, algebra, geometry
- coding: write code, debug code, explain code, algorithms, data structures
- factual: WHAT/WHEN/WHERE questions with objective answers — facts, history, science, geography, definitions, "what is X", "when did X happen", "where is X"
- creative: stories, poems, essays, brainstorming, templates, letters, formats, sample writing, "write me a..."
- reasoning: WHY/SHOULD/HOW-COULD questions requiring analysis — cause & effect, ethics, opinions, "why did X happen", "should I do X", "what are the implications of X"
- procedural: how-to, tutorials, step-by-step instructions, recipes, cooking, processes, "how do you make/build/do..."
- general: casual chat, greetings, simple everyday questions

IMPORTANT DISAMBIGUATION — creative vs procedural:
- "Write me a format of job application" → creative (template/formal writing, not a step-by-step procedure)
- "Write a cover letter for a software engineer role" → creative (composition of a document)
- "How do you make biryani?" → procedural (cooking recipe with ordered steps)

IMPORTANT DISAMBIGUATION — factual vs reasoning:
- "When did World War 2 start?" → factual (objective date, no analysis needed)
- "Why did World War 2 start?" → reasoning (requires causal analysis)
- "What is machine learning?" → factual (definition)
- "Should I learn machine learning?" → reasoning (requires opinion/analysis)
- "How do you bake a cake?" → procedural (step-by-step process)
- "How does photosynthesis work?" → factual (scientific explanation, not a how-to guide)

Complexity (pick one):
- low: answerable in 1-2 steps, simple process
- medium: needs some explanation, 3-5 steps, moderate depth
- high: complex, multi-step, requires expertise or many details

Confidence: float between 0.0 (very unsure) and 1.0 (very sure)

Examples:
Q: "What is the derivative of x squared?"
A: {"category":"math","complexity":"low","confidence":0.99,"reasoning":"Straightforward calculus derivative question"}

Q: "Write a Python function to sort a list by second element of tuples"
A: {"category":"coding","complexity":"medium","confidence":0.97,"reasoning":"Code writing task with specific requirements"}

Q: "When did World War 2 end?"
A: {"category":"factual","complexity":"low","confidence":0.99,"reasoning":"Objective historical date with a single correct answer"}

Q: "Why did World War 2 start?"
A: {"category":"reasoning","complexity":"high","confidence":0.91,"reasoning":"Historical causation requires multi-factor analytical reasoning"}

Q: "Write a short story about a robot learning to paint"
A: {"category":"creative","complexity":"medium","confidence":0.96,"reasoning":"Creative writing task with clear parameters"}

Q: "Write a job application letter for a software engineer role"
A: {"category":"creative","complexity":"medium","confidence":0.95,"reasoning":"Writing a formal application letter is a creative composition task"}

Q: "Write a sample cover letter for a marketing internship"
A: {"category":"creative","complexity":"medium","confidence":0.93,"reasoning":"Template-style writing request that should be handled as creative content"}

Q: "How do you make biryani?"
A: {"category":"procedural","complexity":"high","confidence":0.94,"reasoning":"Cooking recipe with step-by-step process"}

Q: "How should I structure a job application letter?"
A: {"category":"procedural","complexity":"medium","confidence":0.90,"reasoning":"Requesting a format/procedure for creating a letter, which is a step-by-step guidance task"}

Q: "How does photosynthesis work?"
A: {"category":"factual","complexity":"medium","confidence":0.93,"reasoning":"Scientific explanation of a natural process, not a how-to guide"}

Q: "How do you translate this sentence to French?"
A: {"category":"procedural","complexity":"medium","confidence":0.92,"reasoning":"Translation process requiring technique explanation"}"""


async def classify_question(question: str) -> dict:
    """Classify a question into a category with confidence scoring."""
    try:
        override =_override_category(question)
        if override:
            print(f"[CLASSIFIER] Override category to {override['category']} for question: {question[:80]}")
            return override

        result = await _call_classifier(question)

        if result["confidence"] < 0.75:
            print(f"[CLASSIFIER] Low confidence ({result['confidence']:.2f}), running validation...")
            result = await _validate_classification(question, result)

        return result

    except Exception as e:
        print(f"[CLASSIFIER ERROR] {str(e)} — defaulting to general/medium")
        return {
            "category":   "general",
            "complexity": "medium",
            "confidence": 0.5,
            "reasoning":  f"Classifier failed: {str(e)}"
        }


async def _call_classifier(question: str) -> dict:
    """Call Groq API and parse the JSON response."""
    response = await groq_client.chat.completions.create(
        model=GROQ_CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Classify this question: {question}"}
        ],
        temperature=0.1,
        max_tokens=150,
    )

    raw_text = response.choices[0].message.content
    if raw_text is None:
        raw_text = ""
    else:
        raw_text = raw_text.strip()

    try:
        result = _parse_json_response(raw_text)
    except json.JSONDecodeError:
        print(f"[CLASSIFIER] JSON parse failed on: {raw_text}")
        result = _extract_category_from_text(raw_text)

    return {
        "category":   result.get("category",  "general"),
        "complexity":  result.get("complexity", "medium"),
        "confidence": float(result.get("confidence", 0.7)),
        "reasoning":  result.get("reasoning",  "No reasoning provided")
    }


def _override_category(question: str) -> dict:
    """Force creative category for writing/template requests that are not procedural."""
    q = question.lower()

    # Creative writing or document-template requests should not be classified as procedural
    # unless they explicitly ask for step-by-step instructions.
    creative_patterns = [
        r'\b(write|draft|compose|create|prepare|generate)\b.*\b(job application|application letter|cover letter|resume|cv|application format|job application format)\b',
        r'\b(job application|application letter|cover letter|resume|cv|application format|job application format)\b.*\b(write|draft|compose|create|prepare|generate)\b',
        r'\b(format|template|example|sample)\b.*\b(job application|application letter|cover letter|resume|cv)\b',
    ]

    if any(re.search(pattern, q) for pattern in creative_patterns):
        return {
            "category":   "creative",
            "complexity": "medium",
            "confidence": 0.88,
            "reasoning":  (
                "Detected a document or template writing request, "
                "which is best handled as creative content rather than procedural steps."
            )
        }

    return None


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON response, stripping markdown code fences if present."""
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
    if fence_match:
        clean = fence_match.group(1).strip()
    else:
        clean = raw_text.strip()

    return json.loads(clean)


async def _validate_classification(question: str, first_result: dict) -> dict:
    """
    Run a second classification pass if confidence is low.

    FIX L3: Replaced hardcoded 0.82 confidence boost with real average.
    The old code inflated confidence on agreement without measuring it.
    Now we average both pass scores so the returned confidence is honest.
    """
    try:
        second_result = await _call_classifier(question)

        if second_result["category"] == first_result["category"]:
            # FIX: use real average instead of max(conf, 0.82)
            averaged_confidence = (first_result["confidence"] + second_result["confidence"]) / 2
            first_result["confidence"] = averaged_confidence
            first_result["reasoning"] += " [validated by second pass]"
            return first_result
        else:
            print(
                f"[CLASSIFIER] Disagreement: "
                f"{first_result['category']} vs {second_result['category']} — using general"
            )
            return {
                "category":   "general",
                "complexity": first_result.get("complexity", "medium"),
                "confidence": 0.60,
                "reasoning":  (
                    f"Ambiguous: could be {first_result['category']} "
                    f"or {second_result['category']}"
                )
            }

    except Exception:
        return first_result


def _extract_category_from_text(text: str) -> dict:
    """Fallback parser when JSON parsing fails."""
    text_lower = text.lower()

    for category in CATEGORIES.keys():
        if category in text_lower:
            return {
                "category":   category,
                "complexity": "medium",
                "confidence": 0.6,
                "reasoning":  "Extracted from malformed response"
            }

    return {
        "category":   "general",
        "complexity": "medium",
        "confidence": 0.5,
        "reasoning":  "Could not extract from response"
    }