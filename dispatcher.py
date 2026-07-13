# FUSION 1.0 — PARALLEL DISPATCHER

import asyncio
import time
import requests
from typing import Any, cast
from groq import AsyncGroq
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GROQ_ANSWER_MODEL,
    GROQ_OPENAI_MODEL,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    MODEL_ROLES,
    CATEGORY_FORMAT,
    TOKEN_LIMITS,
    ROUTING_TABLE,
)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
gemini_client = None
openai_client = None


def _get_gemini_client():
    """Create a Gemini client lazily so imports remain safe without an API key."""
    global gemini_client
    if gemini_client is None:
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY is not configured.")
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client


def _get_openai_client():
    """Create an OpenAI client lazily so imports remain safe without an API key."""
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY is not configured.")
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return openai_client


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

        if isinstance(result, dict) and result.get("answer"):
            answers.append(result["answer"])
            confidence_scores.append(result["confidence"])
            models_used.append(model_name)
            latencies.append(result.get("latency", 0))
            print(
                f"[DISPATCHER] {model_name}: "
                f"conf={result['confidence']:.2f}, "
                f"time={result.get('latency', 0):.1f}s"
            )

    if len(answers) == 0:
        raise Exception("All models failed to respond. Check your API keys.")
    if len(answers) < 2:
        print(
            f"[DISPATCHER] ⚠️  WARNING: Only {len(answers)} model responded. "
            f"Fusion has no effect — answer is from a single source. "
            f"Check Ollama/Gemini availability."
        )

    return {
        "answers":          answers,
        "confidence_scores": confidence_scores,
        "models_used":      models_used,
        "latencies":        latencies,
        "strategy":         strategy,
    }


# ── HELPER: DYNAMIC TOKEN LIMITS ────────────────────────────

def _get_max_tokens(category: str) -> int:
    """Returns max tokens based on question category."""
    return TOKEN_LIMITS.get(category, 3000)


def _build_system_prompt(role_prompt: str, question: str, category: str) -> str:
    """Build system prompt with category-specific format instruction."""
    format_instruction = CATEGORY_FORMAT.get(category, "")
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

    raw = (response.choices[0].message.content or "").strip()
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

    client = _get_gemini_client()

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=max_tokens,
            system_instruction=prompt,
        ),
    )

    raw = (getattr(response, "text", "") or "").strip()
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

    ollama_api = f"{OLLAMA_BASE_URL}/api/generate"

    try:
        response = await asyncio.to_thread(
            requests.post,
            ollama_api,
            json={
                "model":       OLLAMA_MODEL,
                "prompt":      f"{prompt}\n\nUser question: {question}",
                "stream":      False,
                "temperature": 0.7,
                "options":     {"num_predict": max_tokens}
            },
            timeout=OLLAMA_TIMEOUT
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
        model=GROQ_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )

    raw = (response.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
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
