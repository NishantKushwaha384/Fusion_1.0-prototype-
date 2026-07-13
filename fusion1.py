# ============================================================
# FUSION 1.0 — FUSION ENGINE (fixed)
# ============================================================
#
# Key fixes applied:
#   L9  - extract_facts() now splits on sentence boundaries,
#         not punctuation — code/math/formulas survive intact
#   L10 - Fewer false duplicates because sentence splits produce
#         complete facts rather than fragments
#   L11 - strip_reasoning_style() no longer strips numbered lists
#         (removed the bullet/number pattern that fired on steps)
#   L12 - majority_vote always falls back to best raw answer if
#         build_natural_answer returns empty
#   L13 - _debate_merge compares facts against individual sentences
#         not the entire answer blob
#   L14 - creative_blend picks highest-confidence, not longest
#   L15 - reconstruct_paragraph similarity threshold raised to 0.55
#   L16 - _extract_steps preserves full multi-line step text
#   L17 - LLM synthesizer has hard 5s timeout + truncation guard
# ============================================================

import re
from difflib import SequenceMatcher
from typing import List, Tuple
import requests
import random
from config import GROQ_API_KEY, LLM_SYNTHESIZER_MODEL, LLM_SYNTHESIZER_MAX_TOKENS, LLM_SYNTHESIZER_TIMEOUT


# ── FACT EXTRACTION PIPELINE ────────────────────────────────

def strip_reasoning_style(text: str) -> str:
    """
    Remove conversational filler and markdown noise.

    FIX L11: Removed the pattern r"^\\s*[\\d\\.\\-\\*\\•]+\\s+" which was
    stripping numbered lists and bullet points wholesale — that deleted
    valid procedural steps from secondary-model answers before they
    could contribute as supporting facts.

    Only removes structural chrome (tables, separators, headers,
    filler phrases) — never content lines.
    """
    if not text:
        return ""

    patterns = [
        r"\|.*\|",                                    # Markdown tables
        r"[-_]{3,}",                                  # Horizontal separators
        r"\*\*step[- ]?by[- ]?step.*?\n?",            # "step-by-step" headers
        r"```.*?```",                                 # Code fences (kept as signal, not content)
        r"(here is|this is|sure|the answer is)\s*.*?:",  # Filler lead-ins
        r"(\(IDC|\(2020\)|\(2023\))",                 # Citation artifacts
    ]

    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)

    text = re.sub(r'\n+', ' ', text)
    return text.strip()


def extract_facts(answer: str) -> List[str]:
    """
    FIX L9: Split on sentence boundaries only (period/!/? followed by
    whitespace + capital letter). The old approach split on every comma,
    semicolon, and colon — that destroyed code snippets, chemical
    formulas, math expressions, and list enumerations.

    NOT used for coding or math answers — see _confidence_weighted bypass.
    """
    if not answer:
        return []

    # Sentence boundary: ./?/! followed by whitespace and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', answer.strip())
    facts = []
    for s in sentences:
        s = s.strip()
        # Require at least 5 words to filter out noise fragments
        if len(s.split()) >= 5:
            facts.append(s)
    return facts


def normalize_fact(fact: str) -> str:
    """Lowercase + strip for duplicate comparison."""
    return fact.lower().strip()


def deduplicate(facts: List[str]) -> List[str]:
    """Remove exact or near-exact duplicate facts."""
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
    Filter out low-quality facts.

    Removes:
      - Fragments shorter than 5 words (raised from 3 — 3-word facts
        are almost always sentence fragments left over from a bad split)
      - Filler language
    """
    FILLER_MARKERS = ["something", "anything", "stuff", "things",
                      "whatever", "etc"]
    valid  = []
    issues = []

    for f in facts:
        f_low = f.lower()
        if len(f_low.split()) < 5:
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

    Limitation: lexical, not semantic. Used for deduplication only.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def remove_conflicts(facts: List[str], threshold: float = 0.85) -> List[str]:
    """Remove near-duplicate or conflicting facts."""
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
    Reconstruct a coherent answer from facts.

    FIX L15: Raised similarity threshold from 0.40 to 0.55.
    At 0.40, even loosely related sentences were being dropped as
    "too similar" to the lead answer — useful supplementary facts
    from secondary models were silently discarded.
    """
    if not facts and not lead_answer:
        return "No reliable answer could be formed from the available model responses."

    base = lead_answer.strip()
    if not base:
        base = facts[0] if facts else ""

    additions = []
    for fact in supporting_facts:
        fact_clean = fact.strip()
        if not fact_clean:
            continue
        # FIX L15: raised from 0.40 → 0.55
        if similarity(fact_clean, lead_answer) > 0.55:
            continue
        fact_sentence = fact_clean[0].upper() + fact_clean[1:]
        if not fact_sentence.endswith(('.', '!', '?')):
            fact_sentence += '.'
        additions.append(fact_sentence)

    if not additions:
        return base

    supplement = ' '.join(additions)
    return f"{base}\n\n{supplement}"


def naturalize_answer(facts, extended=False):
    if not facts:
        return ""

    facts = [f.strip().rstrip(".") for f in facts if f.strip()]

    if len(facts) == 1:
        return facts[0] + "."

    connectors = [
        "Additionally,", "Also,", "In addition,",
        "Another key point is", "Notably,"
    ]

    sentence = facts[0] + "."
    for fact in facts[1:]:
        if extended:
            connector = random.choice(connectors)
            sentence += f" {connector} {fact}."
        else:
            sentence += f" {fact}."

    return sentence


def build_natural_answer(facts, extended=False):
    if not facts:
        return ""

    cleaned = []
    seen = set()
    for f in facts:
        f = f.strip().replace("\n", " ")
        if any(x in f.lower() for x in [
            "additionally", "notably", "another key point", "###"
        ]):
            continue
        if len(f.split()) < 5:
            continue
        key = " ".join(f.lower().split()[:6])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f.rstrip("."))

    if not cleaned:
        return ""

    definition, others = [], []
    for f in cleaned:
        if any(k in f.lower() for k in ["refers to", "is defined as", "is the process of"]):
            definition.append(f)
        else:
            others.append(f)

    ordered = definition + others

    merged = [ordered[0]]
    for f in ordered[1:]:
        prev = merged[-1]
        combined_len = len(prev.split()) + len(f.split())
        if combined_len <= 18:
            merged[-1] = prev + "; " + f
        else:
            merged.append(f)

    if not extended:
        return merged[0] + "."
    return " ".join(s + "." for s in merged[:5])


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
        "factual":    _majority_vote,
        "math":       _confidence_weighted,
        "coding":     _confidence_weighted,
        "creative":   _creative_blend,
        "reasoning":  _debate_merge,
        "procedural": _step_synthesis,
        "general":    _confidence_weighted,
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
    MAJORITY VOTE — for factual questions.

    FIX L12: Added hard fallback — if build_natural_answer returns an
    empty string (e.g. all facts were filtered out), the function now
    returns the best model's raw answer instead of an empty response.
    """
    sorted_pairs = sorted(zip(confidences, answers), key=lambda x: x[0], reverse=True)
    sorted_confs, sorted_answers = zip(*sorted_pairs)

    best_answer_cleaned = strip_reasoning_style(sorted_answers[0])

    all_facts = []
    for ans in sorted_answers:
        cleaned = strip_reasoning_style(ans)
        all_facts.extend(extract_facts(cleaned))

    facts = deduplicate(all_facts)
    facts, rejected = validate_facts(facts)
    facts = remove_conflicts(facts)

    extended_keywords = ["explain", "detail", "long", "comprehensive", "why", "how", "list"]
    wants_extended = any(k in question.lower() for k in extended_keywords)

    if not wants_extended:
        primary_norm = {normalize_fact(f) for f in extract_facts(best_answer_cleaned)}
        final_facts  = [f for f in facts if normalize_fact(f) in primary_norm][:3]
        if len(final_facts) < 2:
            final_facts = facts[:3]
        strategy          = "majority_vote_concise"
        supporting_count  = 0
    else:
        best_norm         = {normalize_fact(f) for f in extract_facts(best_answer_cleaned)}
        supplementary     = [f for f in facts if normalize_fact(f) not in best_norm]
        final_facts       = list(best_norm) + supplementary
        strategy          = "majority_vote_extended"
        supporting_count  = len(supplementary)

    definition_signals = ["refers to", "is the systematic", "is a field", "is the development of"]
    definition_facts   = []
    other_facts        = []

    for fact in final_facts:
        if any(s in fact.lower() for s in definition_signals):
            definition_facts.append(fact)
        else:
            other_facts.append(fact)

    if wants_extended:
        ordered_facts = definition_facts + other_facts
    else:
        if definition_facts:
            ordered_facts = [" ".join(definition_facts).strip()]
        else:
            ordered_facts = other_facts[:3]

    final_answer = build_natural_answer(ordered_facts, extended=wants_extended)

    # FIX L12: Never return empty — fall back to best raw answer
    if not final_answer or not final_answer.strip():
        print("[FUSION] majority_vote produced empty output — falling back to best raw answer")
        final_answer = sorted_answers[0]
        strategy    += "_fallback"

    total   = sum(sorted_confs) or 1.0
    weights = [c / total for c in sorted_confs]

    return {
        "answer":   final_answer,
        "weights":  weights,
        "strategy": strategy,
        "notes":    f"Mode: {'Extended' if wants_extended else 'Concise'} | Facts: {len(final_facts)} | Supp: {supporting_count}"
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

    For coding/math: returns the highest-confidence answer directly.
    For others: runs the full fact extraction and fusion pipeline.
    """
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
                f"structured answers must not be sentence-split. "
                f"Returning full answer from highest-confidence model "
                f"(conf={best_conf:.2f}). "
                f"Other models: {[round(c, 2) for c in confidences[1:]]}."
            )
        }

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

    # Safety net: never return empty
    if not final_answer or not final_answer.strip():
        print("[FUSION] confidence_weighted produced empty output — falling back to best raw answer")
        final_answer = best_answer

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
    CREATIVE BLEND STRATEGY.

    FIX L14: Old code picked the longest answer as proxy for quality.
    Length ≠ quality — a verbose mediocre answer beats a tight excellent one.
    Now picks the highest-confidence answer, which is the model's own
    assessment of how well it answered.
    Fact extraction intentionally NOT applied to creative content.
    """
    # FIX L14: highest confidence, not longest length
    best_idx    = confidences.index(max(confidences))
    best_answer = answers[best_idx]
    weights     = [1.0 if i == best_idx else 0.0 for i in range(len(answers))]

    return {
        "answer":   best_answer,
        "weights":  weights,
        "strategy": "creative_blend",
        "notes": (
            "Fact extraction intentionally NOT applied to creative answers. "
            "Using highest-confidence answer. "
            f"Selected model index {best_idx} with confidence {confidences[best_idx]:.2f}."
        )
    }


def call_llm_synthesizer_sync(question: str, answers: list) -> str:
    """
    Blends conflicting AI responses into one natural narrative.

    FIX L17: Added response validation — checks the answer is not
    truncated (ends mid-sentence) and falls back gracefully on timeout.
    Timeout is kept at 5s to avoid hanging the system.
    """
    api_key = GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not configured.")

    context = "\n\n".join([f"MODEL {i+1}:\n{a}" for i, a in enumerate(answers)])

    payload = {
        "model": LLM_SYNTHESIZER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Synthesis Engine. You will receive a question and multiple conflicting AI answers. "
                    "Write a single, cohesive response that acknowledges the different viewpoints without "
                    "mentioning 'Model 1' or 'Model 2'. Blend them into a human-like, balanced summary. "
                    "Always end with a complete sentence."
                )
            },
            {"role": "user", "content": f"QUESTION: {question}\n\nANSWERS:\n{context}"}
        ],
        "temperature": 0.2,
        "max_tokens": LLM_SYNTHESIZER_MAX_TOKENS
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=LLM_SYNTHESIZER_TIMEOUT
        )

        if response.status_code != 200:
            raise Exception(f"Synthesis API failed with status {response.status_code}")

        text = response.json()['choices'][0]['message']['content']

        # FIX L17: Guard against truncated mid-sentence response
        text = text.strip()
        if text and not text[-1] in '.!?':
            # Find the last complete sentence
            last_terminal = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_terminal > len(text) // 2:
                text = text[:last_terminal + 1]
            else:
                # Truncation too severe — fall back to best raw answer
                raise Exception("LLM synthesizer response was too truncated to use")

        return text

    except requests.exceptions.Timeout:
        raise Exception("LLM synthesizer timed out after 5s")


def _debate_merge(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "reasoning"
) -> dict:
    """
    DEBATE MERGE — for reasoning questions.

    FIX L13: Old code compared each fact against the full answer blob
    using similarity(fact, ans) > 0.35. A short specific fact almost
    never scores > 0.35 against a 500-word essay, so core_agreements
    was always empty and the LLM synthesizer always fired.

    Now compares each fact against individual sentences of each answer,
    which gives accurate agreement detection.
    """
    total_conf = sum(confidences) or 1.0
    weights    = [c / total_conf for c in confidences]

    # Pre-split each answer into sentences for accurate comparison
    answer_sentences = [_split_into_sentences(ans) for ans in answers]

    all_facts = []
    for ans in answers:
        all_facts.extend(extract_facts(ans))

    unique_facts          = deduplicate(all_facts)
    valid_facts, _        = validate_facts(unique_facts)

    core_agreements = []
    nuanced_points  = []

    for fact in valid_facts:
        agreement_score = 0.0
        for i, sentences in enumerate(answer_sentences):
            # FIX L13: compare fact against individual sentences, not full answer
            max_sent_sim = max(
                (similarity(fact, sent) for sent in sentences),
                default=0.0
            )
            if max_sent_sim > 0.35:
                agreement_score += weights[i]

        if agreement_score > 0.6:
            core_agreements.append((fact, agreement_score))
        elif agreement_score > 0.3:
            nuanced_points.append((fact, agreement_score))

    core_agreements.sort(key=lambda x: x[1], reverse=True)
    nuanced_points.sort(key=lambda x: x[1], reverse=True)

    def synthesize():
        if not core_agreements and not nuanced_points:
            print("[SYNTHESIS] No programmatic overlap found. Triggering LLM Synthesizer...")
            try:
                return call_llm_synthesizer_sync(question, answers)
            except Exception as e:
                print(f"[SYNTHESIS] LLM synthesizer failed: {e} — using best raw answer")
                # FIX: fall back to best-confidence raw answer, not a vague message
                best_idx = confidences.index(max(confidences))
                return answers[best_idx]

        paragraphs = []

        if core_agreements:
            core_text      = "The strongest consensus centers on these points: "
            core_sentences = [f[0].strip().capitalize() for f in core_agreements[:3]]
            core_sentences = [s + "." if not s.endswith(".") else s for s in core_sentences]
            core_text     += " ".join(core_sentences)
            paragraphs.append(core_text)

        if nuanced_points:
            nuance_text      = "Additional nuanced perspectives indicate that: "
            nuance_sentences = [f[0].strip().capitalize() for f in nuanced_points[:2]]
            nuance_sentences = [s + "." if not s.endswith(".") else s for s in nuance_sentences]
            nuance_text     += " ".join(nuance_sentences)
            paragraphs.append(nuance_text)

        return "\n\n".join(paragraphs)

    final_answer = synthesize()

    # Safety net
    if not final_answer or not final_answer.strip():
        best_idx     = confidences.index(max(confidences))
        final_answer = answers[best_idx]

    return {
        "answer":   final_answer,
        "weights":  weights,
        "strategy": "debate_merge_weighted_v2",
        "notes":    f"{len(core_agreements)} strong agreements, {len(nuanced_points)} secondary insights."
    }


def _step_synthesis(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str = "procedural"
) -> dict:
    """
    STEP SYNTHESIS STRATEGY — Procedural (how-to, recipes, tutorials).

    FIX L16: Old _extract_steps() took only the first line of each step
    via step.strip().split('\\n')[0]. Multi-line steps (e.g. a step with
    a code snippet or sub-explanation below it) were silently truncated.
    Now the full step text is preserved up to a reasonable character cap.
    """
    if not answers:
        return {
            "answer":   "No answer available.",
            "weights":  [],
            "strategy": "step_synthesis",
            "notes":    "No answers provided"
        }

    sorted_pairs = sorted(zip(confidences, answers), key=lambda x: x[0], reverse=True)
    sorted_confs, sorted_answers = zip(*sorted_pairs)
    best_answer = sorted_answers[0]
    best_conf   = sorted_confs[0]

    primary_steps = _extract_steps(best_answer)

    if not primary_steps:
        final_answer = best_answer
        fusion_notes = f"No numbered steps detected. Using highest-confidence answer (conf={best_conf:.2f})"
    else:
        secondary_insights = []
        for ans in sorted_answers[1:]:
            insights = _extract_procedural_insights(ans, primary_steps)
            if insights:
                secondary_insights.extend(insights)

        final_answer = _merge_steps_with_insights(
            primary_steps,
            secondary_insights,
            best_answer
        )
        insight_count = len(secondary_insights)
        fusion_notes  = f"Primary: {len(primary_steps)} steps | Secondary insights: {insight_count} | Best conf: {best_conf:.2f}"

    # Safety net
    if not final_answer or not final_answer.strip():
        final_answer = best_answer
        fusion_notes += " [fallback: merge produced empty output]"

    total   = sum(sorted_confs) or 1.0
    weights = [c / total for c in sorted_confs]

    return {
        "answer":   final_answer,
        "weights":  weights,
        "strategy": "step_synthesis",
        "notes":    fusion_notes
    }


def _extract_steps(text: str) -> List[str]:
    """
    Extract numbered/bulleted steps from procedural text.

    FIX L16: Now preserves the full step text instead of taking only
    the first line. Steps are capped at 500 characters to prevent runaway
    content while preserving multi-sentence step descriptions.
    """
    MAX_STEP_CHARS = 500
    steps = []

    # Pattern 1: Numbered with period/colon/parenthesis
    numbered = re.findall(
        r'^\s*\d+[\.\:\)]\s+(.+?)(?=^\s*\d+[\.\:\)]|\Z)',
        text,
        re.MULTILINE | re.DOTALL
    )
    if numbered:
        for step in numbered:
            # FIX L16: preserve full text, cap at 500 chars
            step = step.strip()[:MAX_STEP_CHARS]
            if len(step.split()) >= 3:
                steps.append(step)
        return steps

    # Pattern 2: Bullet points (-, •, *)
    bulleted = re.findall(
        r'^\s*[-•\*]\s+(.+?)(?=^\s*[-•\*]|\Z)',
        text,
        re.MULTILINE | re.DOTALL
    )
    if bulleted:
        for step in bulleted:
            step = step.strip()[:MAX_STEP_CHARS]
            if len(step.split()) >= 3:
                steps.append(step)
        return steps

    # Pattern 3: "Step 1:", "Step 2:" format
    step_labels = re.findall(
        r'Step\s+\d+:?\s+(.+?)(?=Step\s+\d+:?|\Z)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if step_labels:
        for step in step_labels:
            step = step.strip()[:MAX_STEP_CHARS]
            if len(step.split()) >= 3:
                steps.append(step)
        return steps

    return []


def _extract_procedural_insights(text: str, primary_steps: List[str]) -> List[str]:
    """Extract useful tips, warnings, or alternative approaches."""
    insights = []

    tip_patterns = [
        r'(?:tip|pro tip)[\:\-]?\s*(.+?)(?=\n|tip|note|warning|\Z)',
        r'(?:note|important)[\:\-]?\s*(.+?)(?=\n|tip|note|warning|\Z)',
        r'(?:warning|caution)[\:\-]?\s*(.+?)(?=\n|tip|note|warning|\Z)',
        r'(?:alternative|you can also)[\:\-]?\s*(.+?)(?=\n|alternative|step|\Z)',
    ]

    for pattern in tip_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            insight = match.strip().split('\n')[0][:150]
            if len(insight.split()) >= 5 and insight not in insights:
                insights.append(insight)

    return insights


def _merge_steps_with_insights(
    primary_steps: List[str],
    secondary_insights: List[str],
    best_answer: str
) -> str:
    """Merge primary steps (numbered) with secondary insights."""
    output_lines = []

    prereq_patterns = [
        r'(?:before|first|prerequisite|requirements?|you need)[\:\-]?\s*(.+?)(?=\n|step|procedure)',
    ]
    for pattern in prereq_patterns:
        match = re.search(pattern, best_answer, re.IGNORECASE | re.DOTALL)
        if match:
            prereq = match.group(1).strip().split('\n')[0]
            if len(prereq.split()) >= 3:
                output_lines.append(f"Prerequisites: {prereq}\n")
                break

    output_lines.append("Steps:\n")
    for i, step in enumerate(primary_steps, 1):
        output_lines.append(f"{i}. {step}\n")

    if secondary_insights:
        output_lines.append("\nTips & Insights:\n")
        for insight in secondary_insights[:3]:
            output_lines.append(f"• {insight}\n")

    return "".join(output_lines).strip()


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using sentence-boundary detection.
    Used by _debate_merge for per-sentence similarity comparison (FIX L13).
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]