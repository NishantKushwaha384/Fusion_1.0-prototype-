# FUSION 1.0 — QUESTION CLASSIFIER

import json
import os
import re
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_CLASSIFIER_MODEL = "llama-3.3-70b-versatile"


CATEGORIES = {
    "math":      "Mathematical calculations, equations, proofs, statistics",
    "coding":    "Writing code, debugging, explaining algorithms",
    "factual":   "Facts, history, science, definitions, general knowledge",
    "creative":  "Stories, poems, brainstorming, creative writing",
    "reasoning": "Logic puzzles, analysis, 'why' questions, opinion",
    "general":   "Casual questions, greetings, simple everyday queries"
}

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
    """Classify a question into a category with confidence scoring."""
    try:
        result = await _call_classifier(question)

        if result["confidence"] < 0.75:
            print(f"[CLASSIFIER] Low confidence ({result['confidence']:.2f}), running validation...")
            result = await _validate_classification(question, result)

        return result

    except Exception as e:
        print(f"[CLASSIFIER ERROR] {str(e)} — defaulting to general/medium")
        return {
            "category":  "general",
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

    raw_text = response.choices[0].message.content.strip()

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


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON response, stripping markdown code fences if present."""
    fence_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
    if fence_match:
        clean = fence_match.group(1).strip()
    else:
        clean = raw_text.strip()

    return json.loads(clean)


async def _validate_classification(question: str, first_result: dict) -> dict:
    """Run a second classification pass if confidence is low."""
    try:
        second_result = await _call_classifier(question)

        if second_result["category"] == first_result["category"]:
            first_result["confidence"] = max(first_result["confidence"], 0.82)
            first_result["reasoning"] += " [validated by second pass]"
            return first_result
        else:
            print(
                f"[CLASSIFIER] Disagreement: "
                f"{first_result['category']} vs {second_result['category']} — using general"
            )
            return {
                "category":  "general",
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
                "category":  category,
                "complexity": "medium",
                "confidence": 0.6,
                "reasoning":  "Extracted from malformed response"
            }

    return {
        "category":  "general",
        "complexity": "medium",
        "confidence": 0.5,
        "reasoning":  "Could not extract from response"
    }
