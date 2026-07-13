# ============================================================
# FUSION 1.0 — PROJECT STRUCTURE 
# ============================================================

## What is Fusion 1.0?
An adaptive multi-model AI meta-system that routes questions
to the best combination of free AI models and fuses their
answers into one superior response.

## Project Structure
```
fusion1/
├── main.py          ← Start here. FastAPI server + all endpoints
├── classifier.py    ← Detects question type (math/coding/etc)
├── dispatcher.py    ← Sends question to models in parallel
├── fusion.py        ← YOUR ALGORITHM — combines model answers
├── logger.py        ← Saves every query for research analysis
├── requirements.txt ← Python dependencies
├── .env.template    ← Copy to .env and add your API keys
└── README.md        ← This file
```

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/NishantKushwaha384/Fusion_1.0-prototype-?utm_source=oss&utm_medium=github&utm_campaign=NishantKushwaha384%2FFusion_1.0-prototype-&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)




BROARD DETAILS OF THIS PROJECT
      
# Fusion — Adaptive Multi-Model AI System
Fusion routes a user's question to several LLM providers in parallel (Groq/Llama 3.3, Gemini 2.5 Flash, Ollama/Mistral, OpenAI via Groq), then **fuses** their answers into a single response using a strategy chosen by question category. The repo contains three generations of the fusion engine plus the supporting infrastructure (dispatch, classification, identity handling, logging) and evaluation tooling.

## Fusion engine versions

| File | Version | Status |
|---|---|---|
| `fusion.py` | **v0 — earliest prototype** | Superseded |
| `fusion1.py` | **v1.0 — production fusion engine** | Stable |
| `fusion2.py` | **v2.0 — ESCF-based fusion engine** | Active development |

### `fusion.py` — v0 (prototype)
The original fusion implementation, since replaced by `fusion1.py`. Kept in the repo for history/reference.

### `fusion1.py` — v1.0
The current stable engine. Routes each question category to a dedicated strategy:

- **`_majority_vote`** — factual questions. Extracts sentence-level "facts" from all model answers, deduplicates, removes near-conflicts, and rebuilds a natural-language answer. Falls back to the best raw answer if fact extraction produces nothing.
- **`_confidence_weighted`** — math/coding (bypasses fact extraction entirely and returns the highest-confidence model's full answer verbatim, since structured content must not be sentence-split) and general questions (runs the fact-fusion pipeline).
- **`_creative_blend`** — creative questions. Picks the single highest-confidence answer rather than the longest.
- **`_debate_merge`** — reasoning questions. Compares facts against individual sentences (not whole answer blobs) to find real agreement; if no programmatic overlap is found, calls an LLM synthesizer (`call_llm_synthesizer_sync`) to blend viewpoints, with a raw-answer fallback if that also fails.
- **`_step_synthesis`** — procedural/how-to questions. Extracts full multi-line numbered/bulleted steps from the best answer and merges in tips/insights from secondary models.

Every strategy has a hard fallback so the pipeline never returns an empty string.

### `fusion2.py` — v2.0 (ESCF)
The engine currently under active development, built around **ESCF (Epistemic-State Conditioned Fusion)**. Instead of routing purely by question category, it first classifies the *epistemic state* of the set of model answers — how much they agree, and how confident each model is — and only then picks a fusion strategy. `dry.py` (see below) exercises all four states against this engine:

| Epistemic state | Entropy | Confidence spread | fusion2 behavior |
|---|---|---|---|
| **CONSENSUS** | low | low | Verified majority vote — confirms models actually agree before trusting the vote |
| **COLLECTIVE_DOUBT** | high | low | Overrides to `debate_merge` — synthesizes all viewpoints instead of discarding minority answers |
| **CONFIDENT_DISSENTER** | low | high | Trusts the single high-confidence outlier directly rather than letting it get outvoted |
| **EPISTEMIC_VOID** | high | high | Refuses to blindly fuse — returns the best single answer and flags it for human review |

ESCF classification is bypassed entirely for math, coding, creative, and procedural categories, where blending at the sentence level would corrupt the output.

> Note: the full source of `fusion2.py` wasn't available to read directly in this project — the description above is drawn from `dry.py`'s test harness and prior project context, not a line-by-line read of the file. Worth double-checking against the actual source before publishing.

## Core ESCF logic

### `escf_engine.py`
Holds the ESCF classification matrix itself: computes **semantic entropy** (via Dice coefficient over stemmed/stopword-filtered/AI-ML-normalized tokens, with a neural similarity path and SequenceMatcher fallback) and **confidence spread** across model answers, then classifies into one of the four states above (max confidence ≥ 0.90 acts as the tiebreaker between CONFIDENT_DISSENTER and EPISTEMIC_VOID). Locked thresholds: `ENTROPY_THRESHOLD = 0.60`, `SPREAD_THRESHOLD = 0.35`.

> Not directly read in this session — described from prior project context.

## Supporting infrastructure

### `main.py`
FastAPI entry point. Wires together: identity guard → classifier → dispatcher → fusion → output validation → logging, exposed via `/ask`, plus `/health`, `/logs`, and `/stats` endpoints. Validates fusion output post-hoc (empty/too-short answers fall back to the best individual model answer) and patches the routing table at startup if Ollama isn't running.

### `classifier.py`
Classifies each incoming question into a category (`math`, `coding`, `factual`, `creative`, `reasoning`, `procedural`, `general`) and complexity level, which both the dispatcher and fusion engine use for routing.

> Not directly read in this session — described from its role in `main.py` and `config.py`.

### `dispatcher.py`
Calls Groq, Gemini, Ollama, and OpenAI (via Groq) **in parallel** via `asyncio.gather()`, using per-category system prompts and token limits from `config.py`. Parses each model's raw response into `(answer, confidence)`, tolerates individual model failures, and raises only if every model fails.

### `identity_guard.py`
Intercepts meta-questions about the system itself ("who are you", "are you ChatGPT", "who made you", etc.) before they reach the main pipeline — first via a bank of high-precision regexes, then via an Ollama-based semantic classifier fallback. Answers directly from a fixed identity block rather than dispatching to the LLM ensemble.

### `logger.py`
Appends a structured JSONL record (question, classification, dispatch results, fusion result, latency) per query to `logger.jsonl`, plus analysis helpers: search by question/answer, filter by category/date/performance, CSV export, summary stats, and a small CLI (`python logger.py <command>`).

### `config.py`
Central configuration: API keys, model names, routing table (category → models + strategy), per-model role prompts, category format instructions, token limits, similarity/fusion thresholds, and CORS settings.

## Evaluation & testing

### `dry.py`
The manual dry-test harness for `fusion2.py`. Runs four hand-built test cases — one per ESCF quadrant (CONSENSUS, COLLECTIVE_DOUBT, CONFIDENT_DISSENTER, EPISTEMIC_VOID) — and prints, for each: the detected entropy/spread/state, the strategy fusion2 picked vs. what fusion1 would have picked, the actual fused answer, and a verdict on whether fusion2's behavior is an improvement. Also includes a standalone sanity check comparing the old `SequenceMatcher` similarity against the new `semantic_similarity` function.

### `fusion_lm_eval_adapter.py`
A custom [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness) LM wrapper (registers as `"fusion_adapter"`) that lets the Fusion pipeline be benchmarked on standard tasks like MMLU. Used together with `run_fusion_eval.py`-style runner scripts. Since logprobs aren't available from the fusion pipeline, it uses the `generate_until` interface with a regex-based letter extractor to parse prose output into A/B/C/D answers.


## Frontend

### `fusion.html`, `fusion.css`, `fusion.js`
A minimal web client for hitting the `/ask` endpoint.

> Not read in this session.

## Misc

- **`requirements.txt`** — Python dependencies (not read in this session).
- **`1.py`** — appears to be a scratch/experiment file based on its name; not read in this session, recommend confirming its purpose before including it in a public README.

---

© 2026 Fusion Team. All rights reserved. This work is protected under the Copyright Act, 2059 (2002) of Nepal.
