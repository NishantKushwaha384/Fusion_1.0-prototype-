# ============================================================
# FUSION 1.0 — FUSION ENGINE (fixed · revision 2 + matrix batching)
# ============================================================

import re
from difflib import SequenceMatcher
from typing import List, Tuple, Optional,Dict
import os
import requests
import random
from config import GROQ_API_KEY, LLM_SYNTHESIZER_MODEL, LLM_SYNTHESIZER_MAX_TOKENS, LLM_SYNTHESIZER_TIMEOUT

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sentence_transformers import SentenceTransformer #type ig
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


# ── FACT EXTRACTION PIPELINE ────────────────────────────────

def strip_reasoning_style(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"\|.*?\|",                                   # Markdown tables (patched for lazy match)
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
    if not answer:
        return []

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', answer.strip())
    facts = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 5:
            facts.append(s)
    return facts


def normalize_fact(fact: str) -> str:
    return fact.lower().strip()


def deduplicate(facts: List[str]) -> List[str]:
    seen   = set()
    unique = []
    for f in facts:
        nf = normalize_fact(f)
        if nf not in seen:
            seen.add(nf)
            unique.append(f)
    return unique


def validate_facts(facts: List[str]) -> Tuple[List[str], List[str]]:
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
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ── SEMANTIC SIMILARITY ──────────────────────────────────────

_semantic_model: Optional[SentenceTransformer] = None
_embedding_cache: Dict[str, np.ndarray] = {}  # PERF FIX: Cache embeddings
MAX_CACHE_SIZE = 1000


def get_semantic_model() -> SentenceTransformer:
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def _get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """PERF FIX: Retrieve cached embedding or None."""
    return _embedding_cache.get(text)


def _cache_embedding(text: str, embedding: np.ndarray) -> None:
    """PERF FIX: Cache embedding with size limit."""
    global _embedding_cache
    if len(_embedding_cache) >= MAX_CACHE_SIZE:
        keys_to_remove = list(_embedding_cache.keys())[:100]
        for k in keys_to_remove:
            del _embedding_cache[k]
    _embedding_cache[text] = embedding


def _batch_encode_with_cache(texts: List[str]) -> np.ndarray:
    """PERF FIX: Batch encode texts, using cache for hits."""
    model = get_semantic_model()
    to_encode = []
    indices_to_encode = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        cached = _get_cached_embedding(text)
        if cached is not None:
            results[i] = cached
        else:
            to_encode.append(text)
            indices_to_encode.append(i)
    
    if to_encode:
        try:
            new_embeddings = model.encode(to_encode, convert_to_numpy=True)
            for j, idx in enumerate(indices_to_encode):
                results[idx] = new_embeddings[j]
                _cache_embedding(texts[idx], new_embeddings[j])
        except Exception:
            pass
    
    return np.array([r for r in results if r is not None])


def semantic_similarity(a: str, b: str) -> float:
    try:
        model = get_semantic_model()
        embeddings = model.encode([a, b], convert_to_numpy=True)
        score = cos_sim(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        return float(score)
    except Exception:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def remove_conflicts(facts: List[str], threshold: float = 0.85) -> List[str]:
    """Hybrid conflict removal: string similarity for exact dupes, semantic for paraphrases.
    
    Previous version only ran semantic encoding for >10 facts, causing paraphrases
    (e.g., 'Paris is the capital of France' vs 'The capital of France is Paris')
    to leak through in small fact sets. Now semantic path runs for all sets.
    """
    if not facts:
        return []
    
    # PASS 1: Fast string dedup for exact/near-exact duplicates (catches 80% cheaply)
    string_filtered = []
    for f in facts:
        conflict = False
        for existing in string_filtered:
            ratio = SequenceMatcher(None, f.lower(), existing.lower()).ratio()
            if ratio > 0.92:  # Exact duplicate threshold (slightly higher)
                conflict = True
                break
        if not conflict:
            string_filtered.append(f)
    
    # PASS 2: Semantic dedup for ALL remaining facts (paraphrase detection)
    # This catches word-reordered sentences that string similarity misses.
    if len(string_filtered) < 2:
        return string_filtered
    
    try:
        embeddings = _batch_encode_with_cache(string_filtered)
        sim_matrix = cos_sim(embeddings, embeddings)
        
        final_semantic = []
        final_indices = []
        for i, f in enumerate(string_filtered):
            conflict = False
            for j in final_indices:
                if sim_matrix[i, j] > threshold:
                    conflict = True
                    break
            if not conflict:
                final_semantic.append(f)
                final_indices.append(i)
        return final_semantic
    except Exception:
        # Fallback: string similarity with lower threshold
        final_fallback = []
        for f in string_filtered:
            conflict = False
            for existing in final_fallback:
                ratio = SequenceMatcher(None, f.lower(), existing.lower()).ratio()
                if ratio > threshold:
                    conflict = True
                    break
            if not conflict:
                final_fallback.append(f)
        return final_fallback


def reconstruct_paragraph(
    facts: List[str],
    lead_answer: str,
    supporting_facts: List[str]
) -> str:
    """Reconstruct a coherent answer from facts using vectorized embeddings.
    
    PERF FIX: Batch encode base + supporting facts together instead of separate calls.
    """
    if not facts and not lead_answer:
        return "No reliable answer could be formed from the available model responses."

    base = lead_answer.strip()
    if not base:
        base = facts[0] if facts else ""

    additions = []
    valid_supp = [f.strip() for f in supporting_facts if f.strip()]
    
    if valid_supp:
        try:
            # PERF FIX: Batch encode base + supporting facts together
            all_texts = [base] + valid_supp
            embeddings = _batch_encode_with_cache(all_texts)
            base_emb = embeddings[0].reshape(1, -1)
            supp_embs = embeddings[1:]
            
            sims = cos_sim(supp_embs, base_emb).flatten()
            
            for idx, fact_clean in enumerate(valid_supp):
                if sims[idx] > 0.70:
                    continue
                fact_sentence = fact_clean[0].upper() + fact_clean[1:] if fact_clean else ""
                if fact_sentence and not fact_sentence.endswith(('.', '!', '?')):
                    fact_sentence += '.'
                if fact_sentence:
                    additions.append(fact_sentence)
        except Exception:
            # PERF FIX: Fallback still uses fast string similarity
            for fact_clean in valid_supp:
                if SequenceMatcher(None, fact_clean.lower(), base.lower()).ratio() > 0.70:
                    continue
                fact_sentence = fact_clean[0].upper() + fact_clean[1:] if fact_clean else ""
                if fact_sentence and not fact_sentence.endswith(('.', '!', '?')):
                    fact_sentence += '.'
                if fact_sentence:
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
        if combined_len <= 30:
            merged[-1] = prev + "; " + f
        else:
            merged.append(f)

    if not extended:
        return merged[0] + "."
    return " ".join(s + "." for s in merged[:5])


# ════════════════════════════════════════════════════════════
# HYBRID ESCF — EPISTEMIC-STATE CONDITIONED FUSION (BEST OF BOTH)
# ════════════════════════════════════════════════════════════
#
# Combines:
#   1. Neural embeddings (fusion2) for semantic understanding
#   2. Token-based Dice coefficient (escf_engine) for paraphrases
#   3. Category bypass logic (escf_engine) for structured answers
#   4. Max-min spread (escf_engine) for outlier detection
#   5. Performance optimizations (fusion2) with caching
#
# ── HYBRID TOKEN-BASED SIMILARITY (DICE COEFFICIENT) ──────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "or", "but", "if", "because", "as",
    "until", "while", "this", "that", "these", "those", "it", "its",
    "it's", "they", "them", "their", "which", "who", "whom", "what",
    "where", "when", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now", "i", "we", "you", "he", "she", "do",
    "does", "did", "having", "has", "have", "had", "also", "may", "might",
    "must", "shall", "would", "could", "being",
}

# AI/ML domain-specific synonym mapping
_AI_ML_SYNONYMS = {
    "ml": "machinelearning",
    "machine": "machinelearning",
    "learning": "machinelearning",
    "learn": "machinelearning",
    "ai": "artificialintelligence",
    "artificial": "artificialintelligence",
    "intelligence": "artificialintelligence",
    "algorithms": "algorithm",
    "algorithmic": "algorithm",
    "statistics": "statistical",
    "statistically": "statistical",
    "programmed": "program",
    "programming": "program",
    "systems": "system",
}

_SUFFIXES = (
    "ational", "tional", "alize", "icate", "iciti", "ative", "ical",
    "ness", "ment", "able", "ible", "ing", "ies", "ied", "ion", "ers",
    "er", "ed", "ly", "es", "s",
)


def _stem_token(token: str) -> str:
    """Lightweight suffix-stripping stemmer."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[:-len(suf)]
    return token


def _tokenize(text: str) -> List[str]:
    """HYBRID: Tokenize, stem, and normalize text for Dice similarity."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for tok in text.split():
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        tok = _AI_ML_SYNONYMS.get(tok, tok)
        tok = _stem_token(tok)
        if tok:
            tokens.append(tok)
    return tokens


def _dice_coefficient(a: str, b: str) -> float:
    """HYBRID: Semantic similarity via embeddings (replaces broken token Dice).
    
    Token-based Dice fails on natural paraphrases because words like
    'process' and 'analyze' share no surface-form tokens after stemming.
    We use the existing SentenceTransformer infrastructure
    (_batch_encode_with_cache) which is already imported and cached.
    
    0.0 = no shared meaning
    1.0 = identical meaning
    """
    # Fast path: identical or trivial
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Very short texts: token dice is still fine (no paraphrase complexity)
    if len(a) < 50 and len(b) < 50:
        set_a = set(_tokenize(a))
        set_b = set(_tokenize(b))
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        return (2.0 * intersection) / (len(set_a) + len(set_b))
    
    # Semantic path: use the already-implemented embedding cache
    try:
        embs = _batch_encode_with_cache([a, b])
        if embs.shape[0] < 2:
            return 0.0
        # Compute cosine similarity manually (np is already imported)
        a_vec = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)
        b_vec = embs[1] / (np.linalg.norm(embs[1]) + 1e-8)
        return float(np.dot(a_vec, b_vec))
    except Exception:
        # Fallback to token Dice if embeddings fail
        set_a = set(_tokenize(a))
        set_b = set(_tokenize(b))
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        return (2.0 * intersection) / (len(set_a) + len(set_b))
# ── CATEGORY BYPASS ──────────────────────────────────────────

BYPASS_CATEGORIES = {"math", "coding", "creative", "procedural"}


class ESCFDetector:

    ENTROPY_THRESHOLD  = 0.5   
    CONF_STD_THRESHOLD = 0.25
    CONF_MAXMIN_THRESHOLD = 0.35  # For max-min outlier detection
    CONFIDENT_TIEBREAKER = 0.90   # High-confidence override
    _cache = {}  # Cache entropy calculations

    def compute_semantic_entropy(self, responses: List[str], use_dice: bool = False) -> float:
        """HYBRID: Compute semantic entropy with fallback to Dice.
        
        Optimization:
        - Use cache to avoid recalculation
        - Fast path for short responses (string similarity)
        - Semantic encoding for complex cases
        - Dice coefficient fallback for paraphrase detection
        """
        if len(responses) < 2:
            return 0.0
        
        cache_key = tuple(sorted([hash(r) for r in responses]))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Fast path for short responses or few models
        if len(responses) <= 2 or any(len(r) < 100 for r in responses):
            try:
                n = len(responses)
                sim_matrix = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            sim_matrix[i, j] = 1.0
                        else:
                            # Try Dice first (better paraphrase detection)
                            if use_dice:
                                sim_matrix[i, j] = _dice_coefficient(responses[i][:200], responses[j][:200])
                            else:
                                sim_matrix[i, j] = SequenceMatcher(None, responses[i][:200].lower(), responses[j][:200].lower()).ratio()
            except Exception:
                return 0.5
        else:
            # Use semantic encoding for long, complex cases
            try:
                embeddings = _batch_encode_with_cache(responses)
                sim_matrix = cos_sim(embeddings, embeddings)
            except Exception:
                # Fallback to Dice coefficient
                n = len(responses)
                sim_matrix = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            sim_matrix[i, j] = 1.0
                        else:
                            sim_matrix[i, j] = _dice_coefficient(responses[i][:200], responses[j][:200])

        # === FIX: Cluster-based entropy (proposal-compliant) ===
        # Cluster responses into meaning-equivalent groups using similarity threshold,
        # then compute Shannon entropy over the cluster-size distribution.
        # High entropy = many small clusters (disagreement).
        # Low entropy  = one large cluster (consensus).
        threshold = 0.75
        n = len(responses)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j] and sim_matrix[i, j] >= threshold:
                    cluster.append(j)
                    visited[j] = True
            clusters.append(cluster)

        cluster_sizes = [len(c) for c in clusters]
        total = sum(cluster_sizes)
        probs = np.array([size / total for size in cluster_sizes])
        raw_entropy = scipy_entropy(probs)
        max_entropy = np.log(n)

        result = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Cache result
        self._cache[cache_key] = result
        if len(self._cache) > 500:  # Limit cache size
            keys = list(self._cache.keys())[100:]
            for k in keys:
                del self._cache[k]
        
        return result

    def compute_confidence_spread(self, confidences: List[float], use_maxmin: bool = True) -> float:
        """HYBRID: Support both std-dev and max-min spread.
        
        Max-min is better for detecting outliers (one confident model vs. guesses).
        Std-dev measures overall variance.
        """
        if len(confidences) < 2:
            return 0.0
        
        if use_maxmin:
            # HYBRID: Max-min spread (from escf_engine) - better outlier detection
            return float(max(confidences) - min(confidences))
        else:
            # Original std-dev
            return float(np.std(confidences))

    def detect(
        self,
        responses: List[str],
        confidences: List[float],
        category: str = "factual"
    ) -> Optional[dict]:
        """HYBRID: Full ESCF detection with category awareness.
        
        Returns None if category should bypass ESCF (math, coding, etc.)
        Otherwise returns epistemic state classification.
        """
        if category in BYPASS_CATEGORIES:
            print(f"[ESCF] Bypassing for {category.upper()} category (structured answers)")
            return None

        if len(responses) < 2 or len(confidences) < 2:
            return {
                "state":             "Consensus",
                "semantic_entropy":  0.0,
                "confidence_avg":    0.0,
                "confidence_spread": 0.0,
                "high_entropy":      False,
                "low_confidence":    False,
            }

        # HYBRID: Use Dice coefficient for factual (better paraphrase detection)
        use_dice = category == "factual"
        sem_entropy = self.compute_semantic_entropy(responses, use_dice=use_dice)
        
        # FIX: Use average confidence for the 2x2 matrix (proposal Table 3.8)
        avg_conf = float(np.mean(confidences))
        conf_spread = self.compute_confidence_spread(confidences, use_maxmin=True)

        # HYBRID: High-confidence tiebreaker from escf_engine
        if max(confidences) >= self.CONFIDENT_TIEBREAKER:
            return {
                "state":             "Confident_Dissenter",
                "semantic_entropy":  round(sem_entropy, 4),
                "confidence_avg":    round(avg_conf, 4),
                "confidence_spread": round(conf_spread, 4),
                "high_entropy":      False,
                "low_confidence":    False,
                "tiebreaker_applied": True,
            }

        # Standard 2x2 quadrant classification (proposal Table 3.8)
        # X-axis: Semantic Entropy (Low = agree, High = disagree)
        # Y-axis: Average Confidence (High = confident, Low = uncertain)
        high_entropy = sem_entropy > self.ENTROPY_THRESHOLD
        low_confidence = avg_conf < 0.5

        if not high_entropy and not low_confidence:
            state = "Consensus"
        elif not high_entropy and low_confidence:
            state = "Collective_Doubt"
        elif high_entropy and not low_confidence:
            state = "Confident_Dissenter"
        else:
            state = "Epistemic_Void"

        return {
            "state":             state,
            "semantic_entropy":  round(sem_entropy, 4),
            "confidence_avg":    round(avg_conf, 4),
            "confidence_spread": round(conf_spread, 4),
            "high_entropy":      high_entropy,
            "low_confidence":    low_confidence,
            "tiebreaker_applied": False,
        }


_escf = ESCFDetector()


# ════════════════════════════════════════════════════════════
# MAIN FUSION ENTRY POINT
# How ESCF controls answer fusion
# ════════════════════════════════════════════════════════════
#
# WORKFLOW (called from main.py:/ask endpoint):
#
#   main.py:
#   ├─ question → dispatcher.dispatch_parallel()
#   ├─ [runs groq, gemini, ollama in parallel]
#   └─ answers[], confidences[] → fuse_answers()
#
#   fuse_answers() FLOW:
#   ├─ Step 1: Analyze epistemic state via ESCF
#   │   └─ _escf.detect(answers, confidences)
#   │      → semantic_entropy (do models disagree?)
#   │      → confidence_spread (is trust split?)
#   │
#   ├─ Step 2: If state == "Confident_Dissenter"
#   │   └─ Trust the highest-confidence model
#   │      (one model is clearly more confident than others)
#   │
#   ├─ Step 3: If state == "Collective_Doubt"
#   │   └─ Use debate_merge() to extract & synthesize facts
#   │      (all models disagree but equally uncertain)
#   │
#   ├─ Step 4: If state == "Epistemic_Void"
#   │   └─ Fallback to highest confidence + flag for review
#   │      (total disagreement + split confidence = too risky)
#   │
#   └─ Step 5: Otherwise (Consensus)
#       └─ Use strategy_map[category]
#          (e.g., majority_vote for factual, synthesis for procedural)
#
#   RETURN to main.py:
#   ├─ answer (final merged answer)
#   ├─ epistemic_state (e.g., "Confident_Dissenter")
#   ├─ escf_metrics (entropy, spread, thresholds crossed)
#   ├─ strategy (which fusion strategy was used)
#   └─ notes (explanation for the client)
#

def fuse_answers(
    question: str,
    answers: List[str],
    confidences: List[float],
    category: str
) -> dict:
    if len(answers) == 0:
        return {
            "answer":          "No models were able to respond. Please try again.",
            "weights":         [],
            "strategy":        "none",
            "epistemic_state": None,
            "escf_metrics":    None,
            "notes":           "All models failed",
        }

    if len(answers) == 1:
        result = _single(answers, confidences)
        result["epistemic_state"] = None
        result["escf_metrics"]    = None
        return result

    # HYBRID: Call ESCF detector with category awareness
    # Returns None if category should bypass ESCF (math, coding, creative, procedural)
    escf_result = _escf.detect(answers, confidences, category=category)
    
    if escf_result is None:
        # HYBRID: Category bypass - use domain-specific strategy directly
        print(f"[ESCF] Category '{category}' bypassed — using domain-specific strategy")
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
        result = strategy_fn(question, answers, confidences, category)
        result["epistemic_state"] = None
        result["escf_metrics"]    = None
        return result
    
    escf_state = escf_result["state"]
    best_idx = confidences.index(max(confidences)) if confidences else 0
    best_conf = confidences[best_idx] if confidences else 0.0

    print(
        f"[ESCF] State: {escf_state} | "
        f"Entropy: {escf_result['semantic_entropy']} | "
        f"Conf Spread: {escf_result['confidence_spread']}"
    )

    if escf_state == "Epistemic_Void":
        return {
            "answer":          answers[best_idx],
            "weights":         [1.0 if i == best_idx else 0.0
                                for i in range(len(answers))],
            "strategy":        "escf_epistemic_void_fallback",
            "epistemic_state": escf_state,
            "escf_metrics":    escf_result,
            "notes": (
                "High semantic disagreement AND unequal confidence across models. "
                "Returning highest-confidence single model. Human review recommended."
            ),
        }

    if escf_state == "Confident_Dissenter":
        return {
            "answer":          answers[best_idx],
            "weights":         [1.0 if i == best_idx else 0.0
                                for i in range(len(answers))],
            "strategy":        "escf_confident_dissenter",
            "epistemic_state": escf_state,
            "escf_metrics":    escf_result,
            "notes": (
                f"Confident dissenter detected. "
                f"Trusting highest-confidence model "
                f"(conf={best_conf:.2f})."
            ),
        }

    strategy_map = {
        "factual":    _majority_vote,
        "math":       _confidence_weighted,
        "coding":     _confidence_weighted,
        "creative":   _creative_blend,
        "reasoning":  _debate_merge,
        "procedural": _step_synthesis,
        "general":    _confidence_weighted,
    }

    if escf_state == "Collective_Doubt":
        strategy_fn = _debate_merge
    else:
        strategy_fn = strategy_map.get(category, _confidence_weighted)

    result = strategy_fn(question, answers, confidences, category)

    result["epistemic_state"] = escf_state
    result["escf_metrics"]    = escf_result
    return result


# ════════════════════════════════════════════════════════════
# FUSION STRATEGIES
# ════════════════════════════════════════════════════════════

def _single(answers: List[str], confidences: List[float]) -> dict:
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

        text = text.strip()
        if text and not text[-1] in '.!?':
            last_terminal = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_terminal > len(text) // 2:
                text = text[:last_terminal + 1]
            else:
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
    """PERF FIX: Optimize nested loop + limit fact/sentence processing.
    
    Changes:
    - Limit to top N facts (most confident/relevant)
    - Batch encode all facts + sentences together
    - Reduce sentence processing overhead
    """
    total_conf = sum(confidences) or 1.0
    weights    = [c / total_conf for c in confidences]

    answer_sentences = [_split_into_sentences(ans)[:50] for ans in answers]  # PERF FIX: Limit sentences

    all_facts = []
    for ans in answers:
        all_facts.extend(extract_facts(ans))

    unique_facts          = deduplicate(all_facts)
    valid_facts, _        = validate_facts(unique_facts)
    
    # PERF FIX: Limit facts to top 20 (avoid O(n*m*k) complexity)
    if len(valid_facts) > 20:
        valid_facts = valid_facts[:20]

    core_agreements = []
    nuanced_points  = []

    if valid_facts:
        try:
            # PERF FIX: Batch encode all facts + all sentences together
            all_texts = valid_facts + [s for ss in answer_sentences for s in ss]
            embeddings = _batch_encode_with_cache(all_texts)
            
            fact_embeddings = embeddings[:len(valid_facts)]
            
            ans_embeddings_list = []
            idx = len(valid_facts)
            for sentences in answer_sentences:
                if sentences:
                    ans_embeddings_list.append(embeddings[idx:idx+len(sentences)])
                    idx += len(sentences)
                else:
                    ans_embeddings_list.append(np.array([]))

            for fact_idx, fact in enumerate(valid_facts):
                agreement_score = 0.0
                fact_emb = fact_embeddings[fact_idx].reshape(1, -1)

                for i, sentences in enumerate(answer_sentences):
                    ans_emb = ans_embeddings_list[i]
                    max_sent_sim = 0.0
                    
                    if ans_emb.size > 0:
                        sims = cos_sim(fact_emb, ans_emb)[0]
                        max_sent_sim = float(np.max(sims))

                    if max_sent_sim > 0.55:
                        agreement_score += weights[i]

                if agreement_score > 0.6:
                    core_agreements.append((fact, agreement_score))
                elif agreement_score > 0.3:
                    nuanced_points.append((fact, agreement_score))

        except Exception:
            for fact in valid_facts:
                agreement_score = 0.0
                for i, sentences in enumerate(answer_sentences):
                    max_sent_sim = max(
                        (SequenceMatcher(None, fact.lower(), sent.lower()).ratio() for sent in sentences),
                        default=0.0
                    )
                    if max_sent_sim > 0.55:
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
    MAX_STEP_CHARS = 500
    steps = []

    numbered = re.findall(
        r'^\s*\d+[\.\:\)]\s+(.+?)(?=^\s*\d+[\.\:\)]|\Z)',
        text,
        re.MULTILINE | re.DOTALL
    )
    if numbered:
        for step in numbered:
            step = step.strip()[:MAX_STEP_CHARS]
            if len(step.split()) >= 3:
                steps.append(step)
        return steps

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
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]