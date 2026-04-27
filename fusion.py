# ============================================================
# FUSION 1.0 — FUSION ENGINE
# ============================================================
#
# What fusion means:
#   Get 2-3 answers from different models.
#   Needed to combine them into ONE answer that is
#   BETTER than any individual answer.
#
# Strategies (in order of sophistication):
#   1. single()              — one model, no fusion needed
#   2. majority_vote()       — fact extraction + dedup + conflict
#                              removal; leads with best model
#   3. confidence_weighted() — same pipeline but weighted by conf;
#                              coding/math bypass to full answer
#   4. creative_blend()      — longest answer (pending eval data)
#   5. debate_merge()        — perspectives + convergence note
# ============================================================

import re
from difflib import SequenceMatcher
from typing import List, Tuple
import os
import requests


# FACT EXTRACTION PIPELINE
# 
def strip_reasoning_style(text: str) -> str:
    """Enhanced to handle tables, broken markdown, and conversational filler."""
    if not text:
        return ""

    patterns = [
        r"\|.*\|",                                  # Remove Markdown Tables
        r"[-_]{3,}",                                # Remove horizontal separators (--- or ___)
        r"\*\*step[- ]?by[- ]?step.*?\n?",           
        r"^\s*[\d\.\-\*\•]+\s+",                    # Improved: catches bullets, numbers, and dots
        r"```.*?```",                               
        r"(first|second|third|next|finally)\s*[:,]?", 
        r"(here is|this is|sure|the answer is)\s*.*?:", 
        r"(\(IDC|\(2020\)|\(2023\))",               # Remove citation artifacts
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove extra whitespace and newlines created by stripping
    text = re.sub(r'\n+', ' ', text)
    return text.strip()
def extract_facts(answer: str) -> List[str]:
    """
    Stage 1: Split an answer into individual fact-like clauses.

    Splits on punctuation boundaries (period, comma, semicolon,
    colon, newline). Filters out very short noise fragments.

    Known limitation: Abbreviations like "Dr." or "e.g." cause
    false splits. Document this in your research report.

    NOT used for coding or math answers — see _confidence_weighted()
    bypass for why.
    """
    parts = re.split(r'[.,;:\n]', answer)
    facts = []
    for p in parts:
        p = p.strip()
        if len(p) > 5:
            facts.append(p)
    return facts


def normalize_fact(fact: str) -> str:
    """Stage 2: Lowercase + strip for duplicate comparison."""
    return fact.lower().strip()


def deduplicate(facts: List[str]) -> List[str]:
    """
    Stage 3: Remove exact or near-exact duplicate facts.
    Preserves original capitalization in output.
    """
    seen   = set()
    unique = []
    for f in facts:
        nf = normalize_fact(f)
        if nf not in seen:
            seen.add(nf)
            unique.append(f)
    return unique


def validate_facts(facts: List[str]) -> Tuple[List[str], List[str]]:
    """
    Stage 4: Filter out low-quality facts using generic rules.

    Removes:
      - Fragments shorter than 3 words
      - Filler language ("something", "stuff", "etc")

    Keeps hedging terms ("typically", "usually") — valid in
    formal documentation.
    """
    FILLER_MARKERS = ["something", "anything", "stuff", "things",
                      "whatever", "etc"]
    valid  = []
    issues = []

    for f in facts:
        f_low = f.lower()
        if len(f_low.split()) < 3:
            issues.append(f"[too_short] {f}")
            continue
        if any(x in f_low for x in FILLER_MARKERS):
            issues.append(f"[filler] {f}")
            continue
        valid.append(f)

    return valid, issues


def similarity(a: str, b: str) -> float:
    """
    Character-level string similarity (SequenceMatcher ratio).
    0.0 = nothing in common, 1.0 = identical.

    Limitation: lexical, not semantic. "Paris is the capital of
    France" and "France's capital city is Paris" will score low
    even though they mean the same thing. Documented in report as
    motivation for future embedding-based similarity upgrade.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def remove_conflicts(facts: List[str], threshold: float = 0.85) -> List[str]:
    """
    Stage 5: Remove near-duplicate or conflicting facts.

    If a new fact has similarity > threshold with an already-accepted
    fact, it is skipped (likely a duplicate or minor restatement).

    Limitation: high similarity ≠ guaranteed contradiction.
    Threshold 0.85 is conservative — lower it only if your logs
    show too many valid facts being dropped.
    """
    final = []
    for f in facts:
        conflict = False
        for existing in final:
            if similarity(f, existing) > threshold:
                conflict = True
                break
        if not conflict:
            final.append(f)
    return final


def reconstruct_paragraph(
    facts: List[str],
    lead_answer: str,
    supporting_facts: List[str]
) -> str:
    """
    Stage 6: Reconstruct a coherent answer from facts.
    Uses the full lead answer as the body.
    Supporting facts from secondary models are appended only if genuinely new.

    Args:
      facts:            All validated facts (used for empty-check only)
      lead_answer:      FULL answer from highest-confidence model
      supporting_facts: Unique facts from secondary models
    """
    if not facts and not lead_answer:
        return "No reliable answer could be formed from the available model responses."

    # Use the full best-model answer as the base
    base = lead_answer.strip()
    if not base:
        base = facts[0] if facts else ""

    # Append supporting facts that are genuinely new
    additions = []
    for fact in supporting_facts:
        fact_clean = fact.strip()
        if not fact_clean:
            continue
        # Compare against the ENTIRE lead answer (not just sentence 1)
        if similarity(fact_clean, lead_answer) > 0.4:
            continue
        fact_sentence = fact_clean[0].upper() + fact_clean[1:]
        if not fact_sentence.endswith(('.', '!', '?')):
            fact_sentence += '.'
        additions.append(fact_sentence)

    if not additions:
        return base

    # Append additions as a supplementary paragraph
    supplement = ' '.join(additions)
    return f"{base}\n\n{supplement}"


# ════════════════════════════════════════════════════════════
# MAIN FUSION ENTRY POINT
# ════════════════════════════════════════════════════════════

def fuse_answers(
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
        category:    From classifier (factual, math, coding, creative, reasoning, general)

    Returns:
        dict with keys: answer, weights, strategy, notes
    """

    if len(answers) == 0:
        return {
            "answer":   "No models were able to respond. Please try again.",
            "weights":  [],
            "strategy": "none",
            "notes":    "All models failed"
        }

    if len(answers) == 1:
        return _single(answers, confidences)

    strategy_map = {
        "factual":   _majority_vote,
        "math":      _confidence_weighted,
        "coding":    _confidence_weighted,
        "creative":  _creative_blend,
        "reasoning": _debate_merge,
        "general":   _confidence_weighted,
    }

    strategy_fn = strategy_map.get(category, _confidence_weighted)
    return strategy_fn(question, answers, confidences, category)


# ════════════════════════════════════════════════════════════
# FUSION STRATEGIES
# ════════════════════════════════════════════════════════════

def _single(answers: List[str], confidences: List[float]) -> dict:
    """No fusion — only one answer available."""
    return {
        "answer":   answers[0],
        "weights":  [1.0],
        "strategy": "single",
        "notes":    "Only one model responded — no fusion applied"
    }

def _majority_vote(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "factual"
) -> dict:
    """
    MAJORITY VOTE — Refined for 'Short & Sweet' Factual output.
    ──────────────────────────────────────────────────────────
    1. Removes AI conversational filler and procedural steps.
    2. Dynamically switches between 'concise' (3 facts) and 'extended'.
    3. Reconstructs PURE factual sentences without conversational glue.
    """

    # 1. Rank by confidence
    # zip yields tuples (conf, ans); sorting descending puts highest confidence first
    sorted_pairs = sorted(zip(confidences, answers), key=lambda x: x[0], reverse=True)
    sorted_confs, sorted_answers = zip(*sorted_pairs)

    # Baseline 'best' content
    best_answer_cleaned = strip_reasoning_style(sorted_answers[0])

    # 2. Fact Extraction
    all_facts = []
    for ans in sorted_answers:
        # Clean the raw model output before extraction to prevent procedural facts
        cleaned = strip_reasoning_style(ans)
        all_facts.extend(extract_facts(cleaned))

    # Clean the fact pool
    facts = deduplicate(all_facts)
    facts, rejected = validate_facts(facts)
    facts = remove_conflicts(facts)

    # 3. Intent Detection (Verbosity)
    extended_keywords = ["explain", "detail", "long", "comprehensive", "why", "how", "list"]
    wants_extended = any(k in question.lower() for k in extended_keywords)

    # 4. Selection Logic
    if not wants_extended:
        # CONCISE MODE: Target the 3 most essential facts found in the highest-confidence answer
        primary_norm = {normalize_fact(f) for f in extract_facts(best_answer_cleaned)}
        final_facts = [f for f in facts if normalize_fact(f) in primary_norm][:3]
        
        # If the best answer was too short, supplement with other validated facts
        if len(final_facts) < 2:
            final_facts = facts[:3]
            
        strategy = "majority_vote_concise"
        supporting_count = 0
    else:
        # EXTENDED MODE: Merge baseline facts with unique data from secondary models
        best_norm = {normalize_fact(f) for f in extract_facts(best_answer_cleaned)}
        supplementary = [f for f in facts if normalize_fact(f) not in best_norm]
        final_facts = list(best_norm) + supplementary
        
        strategy = "majority_vote_extended"
        supporting_count = len(supplementary)

    # 5. PRIORITIZED RECONSTRUCTION (The Fix)
    # Define keywords that signal a "Core Definition" sentence.
    definition_signals = ["refers to", "is the systematic", "is a field", "is the development of"]
    
    definition_facts = []
    other_facts = []

    # Sort the available facts into core definitions and supporting details.
    for fact in final_facts:
        if any(s in fact.lower() for s in definition_signals):
            definition_facts.append(fact)
        else:
            other_facts.append(fact)

    if wants_extended:
        # Extended mode: Definition first, then all supplementary facts.
        ordered_facts = definition_facts + other_facts
        strategy = "majority_vote_extended"
    else:
        # Short Mode: Core definition ONLY. Fallback to others if none found.
        if definition_facts:
            # Join definitions directly into a tight paragraph.
            ordered_facts = [" ".join(definition_facts).strip()]
        else:
            ordered_facts = other_facts[:3]
        strategy = "majority_vote_concise"

    # Merge the prioritized list into the final paragraph structure.
    final_answer = " ".join(ordered_facts)

    # Calculate Weights
    total = sum(sorted_confs) or 1.0
    weights = [c / total for c in sorted_confs]

    return {
        "answer": final_answer,
        "weights": weights,
        "strategy": strategy,
        "notes": f"Mode: {'Extended' if wants_extended else 'Concise'} | Facts: {len(final_facts)} | Supp: {supporting_count}"
    }


def _confidence_weighted(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "general",
    min_support_conf: float = 0.40
) -> dict:
    """
    CONFIDENCE-WEIGHTED FUSION — Math, coding, general.
    For coding/math categories, returns the highest-confidence answer directly.
    For other categories, runs the full fact extraction and fusion pipeline.

    Args:
      question:         Original user question
      answers:          List of model answers
      confidences:      Confidence score per model
      category:         Question category
      min_support_conf: Min confidence for secondary model facts
    """

    # ── CODING / MATH BYPASS ──────────────────────────────
    if category in ("coding", "math"):
        sorted_pairs = sorted(
            zip(confidences, answers), key=lambda x: x[0], reverse=True
        )
        best_conf   = sorted_pairs[0][0]
        best_answer = sorted_pairs[0][1]
        total       = sum(confidences) or 1.0
        weights     = [c / total for c in confidences]

        return {
            "answer":   best_answer,
            "weights":  weights,
            "strategy": "confidence_weighted",
            "notes": (
                f"[{category.upper()} BYPASS] Fact extraction skipped — "
                f"structured answers must not be split by punctuation. "
                f"Returning full answer from highest-confidence model "
                f"Answer in plain text. Do NOT use LaTeX, brackets like \\[ \\], or Markdown formatting. Keep it simple."
                f"answer musn't contain boxed"
                f"(conf={best_conf:.2f}). "
                f"Other models: {[round(c,2) for c in confidences[1:]]}."
            )
        }

    # ── STANDARD PIPELINE (factual / general / reasoning) ──
    total_confidence = sum(confidences) or 1.0
    weights          = [c / total_confidence for c in confidences]

    sorted_triples = sorted(
        zip(weights, answers, confidences),
        key=lambda x: x[0],
        reverse=True
    )
    sorted_weights  = [t[0] for t in sorted_triples]
    sorted_answers  = [t[1] for t in sorted_triples]
    sorted_confs    = [t[2] for t in sorted_triples]
    best_answer     = sorted_answers[0]

    all_facts = []
    for i, ans in enumerate(sorted_answers):
        for f in extract_facts(ans):
            all_facts.append((f, sorted_confs[i]))

    seen    = set()
    deduped = []
    for fact, conf in all_facts:
        nf = normalize_fact(fact)
        if nf not in seen:
            seen.add(nf)
            deduped.append((fact, conf))

    validated = []
    for fact, conf in deduped:
        valid, _ = validate_facts([fact])
        if valid:
            validated.append((fact, conf))

    facts_only       = [f for f, _ in validated]
    clean_facts_only = remove_conflicts(facts_only)
    clean_set        = {normalize_fact(f) for f in clean_facts_only}
    best_facts_norm  = {normalize_fact(f) for f in extract_facts(best_answer)}

    supporting_with_conf = [
        (fact, conf) for fact, conf in validated
        if normalize_fact(fact) in clean_set
        and normalize_fact(fact) not in best_facts_norm
        and conf >= min_support_conf
    ]
    supporting_with_conf.sort(key=lambda x: x[1], reverse=True)
    supporting_facts = [f for f, _ in supporting_with_conf]

    final_answer = reconstruct_paragraph(clean_facts_only, best_answer, supporting_facts)

    return {
        "answer":   final_answer,
        "weights":  weights,
        "strategy": "confidence_weighted",
        "notes": (
            f"Weights: {[round(w, 2) for w in weights]}. "
            f"Raw facts: {len(all_facts)} -> clean: {len(clean_facts_only)}. "
            f"Supporting facts added (conf>={min_support_conf}): {len(supporting_facts)}."
        )
    }


def _creative_blend(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "creative"
) -> dict:
    """
    CREATIVE BLEND STRATEGY — Creative writing, poems, stories.
    Returns the longest answer.
    Fact extraction is NOT applied here for creative content.
    """
    longest_answer = max(answers, key=len)
    weights        = [1.0 if a == longest_answer else 0.0 for a in answers]

    return {
        "answer":   longest_answer,
        "weights":  weights,
        "strategy": "creative_blend_placeholder",
        "notes": (
            "Fact extraction intentionally NOT applied to creative answers. "
            "Using longest answer as proxy. Pending human-rated evaluation data."
        )
    }

def call_llm_synthesizer_sync(question: str, answers: list) -> str:
    """
    Blends conflicting AI responses into one natural narrative.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")

    # Prepare the context for the model
    context = "\n\n".join([f"MODEL {i+1}:\n{a}" for i, a in enumerate(answers)])
    
    prompt = {
        "model": "llama-3.1-8b-instant", # Use 8b for speed (fallback should be fast!)
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are a Synthesis Engine. You will receive a question and multiple conflicting AI answers. "
                    "Write a single, cohesive response that acknowledges the different viewpoints without "
                    "mentioning 'Model 1' or 'Model 2'. Blend them into a human-like, balanced summary."
                )
            },
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWERS:\n{context}"}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=prompt,
        timeout=5 # Don't let the fallback hang the whole system
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    
    raise Exception(f"Synthesis API failed with status {response.status_code}")


def _debate_merge(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "reasoning"
) -> dict:

    total_conf = sum(confidences) or 1.0
    weights = [c / total_conf for c in confidences]

    # Extract → Deduplicate → Validate
    all_facts = []
    for ans in answers:
        all_facts.extend(extract_facts(ans))

    unique_facts = deduplicate(all_facts)
    valid_facts, _ = validate_facts(unique_facts)

    core_agreements = []
    nuanced_points = []

    for fact in valid_facts:
        agreement_score = sum(
            weights[i] for i, ans in enumerate(answers)
            if similarity(fact, ans) > 0.35
        )

        if agreement_score > 0.6:
            core_agreements.append((fact, agreement_score))
        elif agreement_score > 0.3:
            nuanced_points.append((fact, agreement_score))

    # Rank by importance
    core_agreements.sort(key=lambda x: x[1], reverse=True)
    nuanced_points.sort(key=lambda x: x[1], reverse=True)

    # Natural synthesis (no robotic bullets)
    # Natural synthesis (safely handling grammar and fallbacks)
    def synthesize():
        # FALLBACK: If programmatic logic finds nothing, call the LLM
        if not core_agreements and not nuanced_points:
            print("[SYNTHESIS] No programmatic overlap found. Triggering LLM Synthesizer...")
            try:
                # We pass the original question and the raw answers from the outer scope
                return call_llm_synthesizer_sync(question, answers)
            except Exception as e:
                # The "emergency" fallback if the API is down
                return "The models provided highly divergent perspectives on this complex issue, with no clear consensus reached."

        # PROGRAMMATIC LOGIC: Standard path
        paragraphs = []

        if core_agreements:
            core_text = "The strongest consensus centers on these points: "
            core_sentences = [f[0].strip().capitalize() for f in core_agreements[:3]]
            core_sentences = [s + "." if not s.endswith(".") else s for s in core_sentences]
            core_text += " ".join(core_sentences)
            paragraphs.append(core_text)

        if nuanced_points:
            nuance_text = "Additional nuanced perspectives indicate that: "
            nuance_sentences = [f[0].strip().capitalize() for f in nuanced_points[:2]]
            nuance_sentences = [s + "." if not s.endswith(".") else s for s in nuance_sentences]
            nuance_text += " ".join(nuance_sentences)
            paragraphs.append(nuance_text)

        return "\n\n".join(paragraphs)
    final_answer = synthesize()

    return {
        "answer": final_answer,
        "weights": weights,
        "strategy": "debate_merge_weighted_v2",
        "notes": f"{len(core_agreements)} strong agreements, {len(nuanced_points)} secondary insights."
    }


# ════════════════════════════════════════════════════════════
# UTILITY
# ════════════════════════════════════════════════════════════

def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences.
    Known limitation: breaks on abbreviations like "Dr." or "U.S."
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]
