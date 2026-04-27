# FUSION 1.0 — PARALLEL DISPATCHER

import asyncio
import os
import time
import requests
from groq import AsyncGroq
import google.generativeai as genai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GROQ_ANSWER_MODEL = "llama-3.3-70b-versatile"


ROUTING_TABLE = {
    "math": {
        "models":      ["openai", "groq", "gemini", "ollama"],
        "strategy":    "confidence_weighted",
        "description": "Math with OpenAI + cloud models"
    },
    "coding": {
        "models":      ["openai", "groq", "gemini", "ollama"],
        "strategy":    "confidence_weighted",
        "description": "Code with OpenAI + cloud models"
    },
    "factual": {
        "models":      ["openai", "groq","ollama"],
        "strategy":    "majority_vote",
        "description": "Facts with OpenAI + cloud models"
    },
    "creative": {
        "models":      ["openai", "groq", "ollama", "gemini"],
        "strategy":    "creative_blend",
        "description": "Creativity with OpenAI + cloud models"
    },
    "reasoning": {
        "models":      ["openai", "groq", "gemini", "ollama"],
        "strategy":    "debate_merge",
        "description": "Reasoning with OpenAI + cloud models"
    },
    "general": {
        "models":      ["openai", "ollama", "groq"],
        "strategy":    "confidence_weighted",
        "description": "Simple questions — OpenAI + local Ollama + Groq fallback"
    }
}


MODEL_ROLES = {
    "groq": {
        "name":   "Primary Analyst",
        "prompt": (
            "You are the Primary Analyst in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide accurate, complete, and properly formatted answers.\n"
            "Focus on ACCURACY and CLARITY.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.87"
        )
    },
    "gemini": {
        "name":   "Creative Synthesizer",
        "prompt": (
            "You are the Creative Synthesizer in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide complete answers with practical examples and clear structure.\n"
            "Focus on CLARITY, USEFULNESS, and PROPER FORMATTING.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.91"
        )
    },
    "ollama": {
        "name":   "Local Validator",
        "prompt": (
            "You are the Local Validator in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide complete, well-structured answers with proper validation.\n"
            "Focus on CORRECTNESS and CLEAR FORMATTING.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.85"
        )
    },
    "openai": {
        "name":   "Advanced Reasoner",
        "prompt": (
            "You are the Advanced Reasoner in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide deep, nuanced answers with comprehensive reasoning.\n"
            "Focus on DEPTH, ACCURACY, and COMPREHENSIVE EXPLANATION.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.89"
        )
    }
}


_CATEGORY_FORMAT = {
    "math":      "This is a MATH question. Show step-by-step working. Use clear notation.",
    "coding":    "This is a CODING question. Provide complete, working code with explanation.",
    "factual":   "This is a FACTUAL question. Be precise, cite key facts, stay objective.",
    "creative":  "This is a CREATIVE question. Write imaginatively with vivid detail.",
    "reasoning": "This is a REASONING question. Structure your argument with clear logic.",
    "general":   "This is a GENERAL question. Be concise and conversational.",
}


async def dispatch_parallel(question: str, category: str, complexity: str) -> dict:
    """
    Sends question to all relevant models SIMULTANEOUSLY.

    Args:
        question:   The user's question
        category:   From classifier (math, coding, etc.)
        complexity: From classifier (low, medium, high)

    Returns:
        dict with: answers, confidence_scores, models_used, strategy, latencies
    """
    routing        = ROUTING_TABLE.get(category, ROUTING_TABLE["general"])
    models_to_call = routing["models"]
    strategy       = routing["strategy"]

    print(f"[DISPATCHER] Category={category}, Strategy={strategy}, Models={models_to_call}")

    tasks = []
    for model_name in models_to_call:
        if model_name == "groq":
            tasks.append(_call_groq(question, category))
        elif model_name == "gemini":
            tasks.append(_call_gemini(question, category))
        elif model_name == "ollama":
            tasks.append(_call_ollama(question, category))
        elif model_name == "openai":
            tasks.append(_call_groq_openai(question, category))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers           = []
    confidence_scores = []
    models_used       = []
    latencies         = []

    for i, result in enumerate(results):
        model_name = models_to_call[i]

        if isinstance(result, Exception):
            print(f"[DISPATCHER] {model_name} failed: {str(result)}")
            continue

        if result and result.get("answer"):
            answers.append(result["answer"])
            confidence_scores.append(result["confidence"])
            models_used.append(model_name)
            latencies.append(result.get("latency", 0))
            print(
                f"[DISPATCHER] {model_name}: "
                f"conf={result['confidence']:.2f}, "
                f"time={result.get('latency', 0):.1f}s"
            )

    if not answers:
        raise Exception("All models failed to respond. Check your API keys.")

    return {
        "answers":           answers,
        "confidence_scores": confidence_scores,
        "models_used":       models_used,
        "strategy":          strategy,
        "latencies":         latencies
    }


# ── HELPER: DYNAMIC TOKEN LIMITS ────────────────────────────

def _get_max_tokens(category: str) -> int:
    """Returns max tokens based on question category."""
    return {
        "math":      2000,
        "coding":    3000,
        "factual":   3000,
        "creative":  3000,
        "reasoning": 3000,
        "general":   3000,
    }.get(category, 3000)


def _build_system_prompt(role_prompt: str, question: str, category: str) -> str:
    """Build system prompt with category-specific format instruction."""
    format_instruction = _CATEGORY_FORMAT.get(category, "")
    if format_instruction:
        return f"{format_instruction}\n\n{role_prompt}"
    return role_prompt

async def _call_groq(question: str, category: str) -> dict:
    """Call Groq API."""
    start      = time.time()
    role       = MODEL_ROLES["groq"]
    max_tokens = _get_max_tokens(category)
    prompt     = _build_system_prompt(role["prompt"], question, category)

    response = await groq_client.chat.completions.create(
        model=GROQ_ANSWER_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    raw    = response.choices[0].message.content.strip()
    answer, confidence = _parse_model_response(raw)

    return {
        "answer":     answer,
        "confidence": confidence,
        "latency":    round(time.time() - start, 2),
        "model":      GROQ_ANSWER_MODEL
    }


async def _call_gemini(question: str, category: str) -> dict:
    """
    Calls Google Gemini API (gemini-2.5-flash, free tier).
    """
    start      = time.time()
    role       = MODEL_ROLES["gemini"]
    max_tokens = _get_max_tokens(category)
    prompt     = _build_system_prompt(role["prompt"], question, category)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=prompt
    )

    response = await asyncio.to_thread(
        model.generate_content,
        question,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=max_tokens,
        )
    )

    raw    = response.text.strip()
    answer, confidence = _parse_model_response(raw)

    return {
        "answer":     answer,
        "confidence": confidence,
        "latency":    round(time.time() - start, 2),
        "model":      "gemini-2.5-flash"
    }


async def _call_ollama(question: str, category: str) -> dict:
    """
    Calls local Ollama model.
    """
    start      = time.time()
    role       = MODEL_ROLES.get("ollama", MODEL_ROLES["groq"])
    max_tokens = _get_max_tokens(category)
    prompt     = _build_system_prompt(role["prompt"], question, category)

    ollama_api = "http://localhost:11434/api/generate"

    try:
        response = await asyncio.to_thread(
            requests.post,
            ollama_api,
            json={
                "model":       "mistral",
                "prompt":      f"{prompt}\n\nUser question: {question}",
                "stream":      False,
                "temperature": 0.7,
                "options":     {"num_predict": max_tokens}
            },
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")

        raw    = response.json().get("response", "").strip()
        answer, confidence = _parse_model_response(raw)

        return {
            "answer":     answer,
            "confidence": confidence,
            "latency":    round(time.time() - start, 2),
            "model":      "mistral"
        }

    except requests.exceptions.ConnectionError:
        raise Exception("Ollama not running. Start with: ollama serve")
    except Exception as e:
        print(f"[DISPATCHER] Ollama unavailable ({str(e)}). Skipping.")
        raise Exception(f"Ollama error: {str(e)}")


async def _call_groq_openai(question: str, category: str) -> dict:
    """Call Groq API with OpenAI compatibility."""
    start      = time.time()
    role       = MODEL_ROLES["groq"]
    max_tokens = _get_max_tokens(category)
    prompt     = _build_system_prompt(role["prompt"], question, category)

    response = await groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    raw    = response.choices[0].message.content.strip()
    answer, confidence = _parse_model_response(raw)

    return {
        "answer":     answer,
        "confidence": confidence,
        "latency":    round(time.time() - start, 2),
        "model":      "openai/gpt-oss-120b"
    }

def _parse_model_response(raw_text: str) -> tuple:
    """Extract answer text and confidence score from model response."""
    lines          = raw_text.strip().split('\n')
    confidence     = 0.75
    answer_lines   = []

    for line in lines:
        if line.strip().startswith("CONFIDENCE:"):
            try:
                conf_str   = line.replace("CONFIDENCE:", "").strip()
                confidence = float(conf_str)
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.75
        else:
            answer_lines.append(line)

    answer = '\n'.join(answer_lines).strip()

    if not answer:
        answer     = raw_text
        confidence = 0.6   # Lower confidence — format was wrong

    return answer, confidence
