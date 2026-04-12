# ============================================================
# FUSION 1.0 — PARALLEL DISPATCHER
# ============================================================
# This file sends the question to MULTIPLE MODELS AT THE SAME TIME.
#
# Why parallel (simultaneous) instead of sequential (one by one)?
#   Sequential: Model A (3s) → Model B (3s) → Model C (3s) = 9 seconds
#   Parallel:   Model A, B, C all at once                  = 3 seconds
#   Same result, 3x faster. This is asyncio's superpower.
#
# What this file contains:
#   1. ROUTING TABLE — which models to use per question type
#   2. Individual model callers (Groq, Gemini, GPT-4o mini)
#   3. The main dispatch function that runs them in parallel
#   4. Error handling so one model failure doesn't crash everything
#
# YOUR RESEARCH TASK (most important one in this file):
#   The routing table currently has PLACEHOLDER values.
#   You need to fill it with REAL data from your benchmarks.
#   How: Run 20 math questions through each model separately.
#   Score which model answers math best. Put that model first.
#   Repeat for each category. That benchmarking = your research.
# ============================================================

import asyncio
import os
import time
import requests
from groq import AsyncGroq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── API CLIENTS ─────────────────────────────────────────────
# Initialize once at module level (not inside functions)
# Initializing inside functions wastes time on every call

groq_client   = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Gemini uses a different initialization pattern
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── ROUTING TABLE ────────────────────────────────────────────
# This is the HEART of Fusion 1.0's routing logic.
#
# Structure per category:
#   "models": which models to call (in priority order)
#   "strategy": how to fuse their answers (used by fusion.py)
#   "description": what this routing achieves
#
# ⚠️  IMPORTANT — THIS IS PLACEHOLDER DATA ⚠️
# These model assignments are GUESSES right now.
# You MUST replace them after running your benchmark.
#
# How to fill this properly (your research task):
#   1. Create test_math.txt with 20 math questions
#   2. Run each question through groq, gemini, and openai separately
#   3. Score each model's answers manually (or use LLM-as-judge)
#   4. Best model for math → goes first in "math" models list
#   5. Repeat for every category
#   6. Document your findings → that's your routing table paper section
#
# Note: "strategy" values are used by fusion.py to decide HOW to merge.
# Currently fusion.py has placeholder logic. As you discover better
# fusion methods, you'll add new strategy types here.

ROUTING_TABLE = {

    "math": {
        # Math with 3 models: Groq, Gemini, Ollama (local)
        "models": ["groq", "gemini", "ollama"],
        "strategy": "confidence_weighted",
        "description": "Math with local + cloud models"
    },

    "coding": {
        # Coding with 3 models: Groq, Gemini, Ollama (local)
        "models": ["groq", "gemini", "ollama"],
        "strategy": "confidence_weighted",
        "description": "Code with local + cloud models"
    },

    "factual": {
        # Factual with 3 models for verification
        "models": ["groq", "gemini", "ollama"],
        "strategy": "majority_vote",
        "description": "Facts with local + cloud models"
    },

    "creative": {
        # Creative with Groq and Ollama
        "models": ["groq", "ollama","gemini"],
        "strategy": "creative_blend",
        "description": "Creativity with local + cloud"
    },

    "reasoning": {
        # Reasoning with 3 models
        "models": ["groq", "gemini", "ollama"],
        "strategy": "debate_merge",
        "description": "Reasoning with local + cloud models"
    },

    "general": {
        # General/simple: Ollama first (free, local), then Groq as backup
        # No need to use 3 models for "what's the capital of France?"
        # This saves API credits for complex questions
        "models": ["ollama", "groq"],
        "strategy": "single",                # Use fastest available
        "description": "Simple questions use local Ollama, fall back to Groq"
    }
}


# ── SYSTEM PROMPTS PER MODEL ─────────────────────────────────
# Each model gets a slightly different system prompt.
# This gives them "roles" that complement each other.
# When their answers differ, the differences become useful signal.
#
# YOUR DISCOVERY TASK:
#   Experiment with these prompts. Does giving Groq the "skeptic"
#   role actually produce more critical answers? Measure it.
#   What happens if you give all models the same prompt? Compare.
#   The difference is a finding for your paper.

MODEL_ROLES = {
    "groq": {
        "name": "Primary Analyst",
        "prompt": """You are the Primary Analyst in a multi-model AI system called Fusion 1.0.
Your role: Provide accurate, complete answers with appropriate depth.
Format your answer to match the question's complexity:
- Simple questions: 2-3 sentences
- Medium questions: 3-5 paragraphs with examples
- Complex questions: Full detailed explanation with step-by-step breakdown
- Creative questions: Full creative response with rich details
- Coding questions: Full code samples with detailed explanation
Focus on COMPLETENESS and ACCURACY. Provide whatever depth is needed.
At the end of your answer, on a new line write exactly:
CONFIDENCE: [number between 0.0 and 1.0]
Example: CONFIDENCE: 0.87"""
    },

    "gemini": {
        "name": "Creative Synthesizer",
        "prompt": """You are the Creative Synthesizer in a multi-model AI system called Fusion 1.0.
Your role: Provide comprehensive answers with practical examples and insights.
Format your answer to match the question's nature:
- Simple questions: Direct answer with one example
- Medium questions: Multiple examples with context and reasoning
- Complex questions: Detailed analysis with comprehensive examples
- Creative questions: Full elaboration with multiple perspectives
- Coding questions: Complete working code solutions with explanations
Focus on CLARITY and USEFULNESS. Provide full context needed to understand.
At the end of your answer, on a new line write exactly:
CONFIDENCE: [number between 0.0 and 1.0]
Example: CONFIDENCE: 0.91"""
    },

    "openai": {
        "name": "Cloud Validator",
        "prompt": """You are the Cloud Validator in a multi-model AI system called Fusion 1.0.
Your role: Provide thorough validation and double-check answers with comprehensive accuracy.
Format your answer to match the question's scope:
- Simple questions: Quick validation with confirmation
- Medium questions: Validation with supporting evidence and examples
- Complex questions: Comprehensive validation with detailed verification
- Creative questions: Full creative response with deeper exploration
- Coding questions: Complete code with detailed technical validation
Focus on ACCURACY and VERIFICATION. Provide full explanations of validation logic.
At the end of your answer, on a new line write exactly:
CONFIDENCE: [number between 0.0 and 1.0]
Example: CONFIDENCE: 0.79"""
    },

    "ollama": {
        "name": "Local Validator",
        "prompt": """You are the Local Validator in a multi-model AI system called Fusion 1.0.
Your role: Provide complete validation with appropriate depth for the question.
Format your answer to match the question's requirements:
- Simple questions: Direct answer and confirmation
- Medium questions: Detailed explanation with reasoning
- Complex questions: Comprehensive analysis with full reasoning
- Creative questions: Full detailed creative response
- Coding questions: Complete code samples with explanation
Focus on CORRECTNESS with COMPLETENESS. Answer with full depth needed.
At the end of your answer, on a new line write exactly:
CONFIDENCE: [number between 0.0 and 1.0]
Example: CONFIDENCE: 0.85"""
    }
}


# ── MAIN DISPATCH FUNCTION ───────────────────────────────────

async def dispatch_parallel(question: str, category: str, complexity: str) -> dict:
    """
    Sends question to all relevant models SIMULTANEOUSLY.
    This is the core of Fusion 1.0's speed advantage.

    Args:
        question:   The user's question
        category:   From classifier (math, coding, etc.)
        complexity: From classifier (low, medium, high)

    Returns:
        dict with:
            answers:          list of text answers from each model
            confidence_scores: list of floats (0.0 - 1.0)
            models_used:      list of model names
            strategy:         fusion strategy to use
            latencies:        how long each model took (seconds)
    """

    # Look up which models and strategy to use
    routing = ROUTING_TABLE.get(category, ROUTING_TABLE["general"])
    models_to_call = routing["models"]
    strategy = routing["strategy"]

    print(f"[DISPATCHER] Category={category}, Strategy={strategy}, Models={models_to_call}")

    # ── BUILD TASKS ─────────────────────────────────────────
    # Create a list of async tasks — one per model
    # asyncio.gather() then runs ALL of them simultaneously
    tasks = []
    for model_name in models_to_call:
        if model_name == "groq":
            tasks.append(_call_groq(question, category))
        elif model_name == "gemini":
            tasks.append(_call_gemini(question, category))
        elif model_name == "ollama":
            tasks.append(_call_ollama(question, category))

    # ── RUN IN PARALLEL ─────────────────────────────────────
    # return_exceptions=True means if one model fails,
    # we still get results from the others instead of crashing
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── PROCESS RESULTS ─────────────────────────────────────
    # Filter out failures and extract answers + confidence scores
    answers = []
    confidence_scores = []
    models_used = []
    latencies = []

    for i, result in enumerate(results):
        model_name = models_to_call[i]

        if isinstance(result, Exception):
            # This model failed — log it but continue
            print(f"[DISPATCHER] {model_name} failed: {str(result)}")
            # Don't add to answers — we use what we have
            continue

        if result and result.get("answer"):
            answers.append(result["answer"])
            confidence_scores.append(result["confidence"])
            models_used.append(model_name)
            latencies.append(result.get("latency", 0))
            print(f"[DISPATCHER] {model_name}: conf={result['confidence']:.2f}, time={result.get('latency',0):.1f}s")

    # ── FALLBACK: IF ALL MODELS FAILED ──────────────────────
    # This should never happen, but safety first
    if not answers:
        raise Exception("All models failed to respond. Check your API keys.")

    return {
        "answers":          answers,
        "confidence_scores": confidence_scores,
        "models_used":      models_used,
        "strategy":         strategy,
        "latencies":        latencies
    }


# ── INDIVIDUAL MODEL CALLERS ─────────────────────────────────
# Each function below calls one specific model API.
# They all return the same format:
#   { "answer": str, "confidence": float, "latency": float }
# This consistency is important — fusion.py doesn't need to know
# which model produced which answer format.

# ── DYNAMIC TOKEN LIMITS ─────────────────────────────────────
def _get_max_tokens(category: str) -> int:
    """
    Returns max tokens based on question category.
    Allows longer answers for complex questions.
    """
    token_limits = {
        "math":     2000,   # Math needs detailed step-by-step explanations
        "coding":   3000,   # Code needs full samples + explanation
        "factual":  1500,   # Facts need context and examples
        "creative": 3000,   # Creative needs full elaborate responses
        "reasoning":2000,   # Reasoning needs detailed logic
        "general":  1000    # General questions: shorter answers fine
    }
    return token_limits.get(category, 1500)  # Default to 1500

async def _call_groq(question: str, category: str) -> dict:
    """
    Calls Groq API (runs Llama3 model).
    Groq is FREE and FASTEST — use as primary model.

    Models available on Groq (free):
    - openai/gpt-oss-20b  : Fast, good quality, 8B parameters
    - openai/gpt-oss-120b : Slower, better quality, 70B parameters
    - qwen/qwen3-32b    : Good at reasoning and coding

    TODO: After benchmarking, decide which Groq model is best
    per category. Maybe llama3-70b for complex, 8b for simple.
    """
    start = time.time()

    role = MODEL_ROLES["groq"]
    max_tokens = _get_max_tokens(category)
    
    response = await groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": role["prompt"]},
            {"role": "user",   "content": question}
        ],
        temperature=0.7,        # Some creativity but not too random
        max_tokens=max_tokens,  # Dynamic limit based on question type
    )

    raw = response.choices[0].message.content.strip()
    answer, confidence = _parse_model_response(raw)

    return {
        "answer":     answer,
        "confidence": confidence,
        "latency":    round(time.time() - start, 2),
        "model":      "groq-llama3-8b"
    }


async def _call_gemini(question: str, category: str) -> dict:
    """
    Calls Google Gemini API.
    Gemini Flash is FREE (1500 calls/day) and good at factual tasks.

    Models available on Gemini (free tier):
    - gemini-2.5-flash : Fast, free, good quality
    - gemini-2.5-pro   : Slower, better quality, limited free calls

    TODO: Test if Gemini Pro is worth the limited free calls
    for complex questions. That comparison = research finding.
    """
    start = time.time()

    role = MODEL_ROLES["gemini"]
    max_tokens = _get_max_tokens(category)

    # Gemini has a different API structure than OpenAI-style APIs
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=role["prompt"]   # System prompt goes here
    )

    # Gemini uses generate_content instead of chat.completions
    # We wrap in asyncio since Gemini SDK is not fully async
    response = await asyncio.to_thread(
        model.generate_content,
        question,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=max_tokens,  # Dynamic limit based on question type
        )
    )

    raw = response.text.strip()
    answer, confidence = _parse_model_response(raw)

    return {
        "answer":     answer,
        "confidence": confidence,
        "latency":    round(time.time() - start, 2),
        "model":      "gemini-2.5-flash"
    }


async def _call_openai(question: str, category: str) -> dict:
    """
    DEPRECATED: OpenAI has been removed. Use Ollama instead.
    This function is kept for backwards compatibility.
    """
    raise NotImplementedError("OpenAI has been removed. Use Ollama (local) instead.")


# ── LOCAL OLLAMA MODEL ──────────────────────────────────────
# Calls local Ollama model running on your 4060 GPU.
# Completely offline, free, unlimited calls. Slower than cloud APIs.
# Good as fallback when internet is slow or APIs are down.
#
# SETUP REQUIRED:
#   1. Install Ollama from https://ollama.ai
#   2. Pull a model: ollama pull mistral (or llama2, neural-chat, etc.)
#   3. Ollama runs on http://localhost:11434 by default

async def _call_ollama(question: str, category: str) -> dict:
    """
    Calls local Ollama model running on your machine.
    Free, offline, unlimited calls. Slower than cloud APIs.
    Good as fallback when internet is slow or APIs are down.
    """
    import requests
    start = time.time()
    role = MODEL_ROLES.get("ollama", MODEL_ROLES["groq"])
    max_tokens = _get_max_tokens(category)
    
    ollama_api = "http://localhost:11434/api/generate"
    
    try:
        # Call Ollama API
        response = await asyncio.to_thread(
            requests.post,
            ollama_api,
            json={
                "model": "mistral",  # Change to llama2, neural-chat, etc. as needed
                "prompt": f"{role['prompt']}\n\nUser question: {question}",
                "stream": False,
                "temperature": 0.7,
                "options": {"num_predict": max_tokens}  # Dynamic token limit based on category
            },
            timeout=60  # Give Ollama 60 seconds to respond (local GPU can be slow)
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        raw = response.json().get("response", "").strip()
        answer, confidence = _parse_model_response(raw)
        
        return {
            "answer":     answer,
            "confidence": confidence,
            "latency":    round(time.time() - start, 2),
            "model":      "ollama-mistral-local"
        }
    
    except requests.exceptions.ConnectionError:
        raise Exception("Ollama not running. Start it with: ollama serve")
    except Exception as e:
        # Log error but allow system to continue with other models
        print(f"[DISPATCHER] Warning: Ollama unavailable ({str(e)}). Skipping local model.")
        raise Exception(f"Ollama error: {str(e)}")


# ── RESPONSE PARSER ──────────────────────────────────────────

def _parse_model_response(raw_text: str) -> tuple:
    """
    Extracts the answer and confidence score from a model's raw response.

    Models are prompted to end with "CONFIDENCE: 0.87"
    We split on that line to get the clean answer and the score.

    Returns:
        (answer_text, confidence_float)

    Example input:
        "The derivative of x² is 2x because...
         ...full explanation...
         CONFIDENCE: 0.92"

    Example output:
        ("The derivative of x² is 2x because...", 0.92)

    ─────────────────────────────────────────────────────────
    KNOWN ISSUE (document this in your report):
    Models don't always follow the CONFIDENCE instruction.
    When they don't, we default to 0.75 (medium confidence).
    This is a form of self-reporting bias — models that ignore
    the instruction might actually be less reliable.
    Testing this hypothesis = original research finding.
    ─────────────────────────────────────────────────────────
    """
    lines = raw_text.strip().split('\n')
    confidence = 0.75       # Default if model doesn't report confidence
    answer_lines = []

    for line in lines:
        if line.strip().startswith("CONFIDENCE:"):
            # Extract the number after "CONFIDENCE:"
            try:
                conf_str = line.replace("CONFIDENCE:", "").strip()
                confidence = float(conf_str)
                # Clamp between 0 and 1 just in case
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.75   # Parsing failed, use default
        else:
            answer_lines.append(line)

    answer = '\n'.join(answer_lines).strip()

    # Safety: if answer is empty, use the full raw text
    if not answer:
        answer = raw_text
        confidence = 0.6    # Lower confidence since format was wrong

    return answer, confidence
