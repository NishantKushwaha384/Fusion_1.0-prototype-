# ============================================================
# FUSION 1.0 — BACKEND ENTRY POINT (fixed)
# ============================================================
#
# Fixes applied:
#   L18 - Math format instruction is now in the category format
#         string inside dispatcher.py (where it belongs), not
#         buried in fusion_notes that no model ever reads.
#   L19 - FusionResponse now includes the `answerer` field so
#         the return statement in /ask doesn't cause a Pydantic
#         ValidationError on every non-identity request.
#   L20 - Added post-fusion output validation. Empty string,
#         whitespace-only, or sub-3-word answers are caught and
#         replaced with the best individual model answer before
#         the response is returned to the client.
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
import requests
from collections import defaultdict
from config import CORS_ORIGINS, MAX_QUESTION_LENGTH, SERVER_HOST, SERVER_PORT, OLLAMA_BASE_URL, ROUTING_TABLE

from classifier  import classify_question
from dispatcher  import dispatch_parallel
from fusion2     import fuse_answers
from logger      import log_query
from identity_guard import check_identity_guard

app = FastAPI(
    title="Fusion 1.0",
    description="Adaptive multi-model AI meta-system",
    version="1.1.0"
)


@app.get("/health")
def health_check():
    ollama_up = _check_ollama_available()
    return {
        "status":         "healthy",
        "version":        "1.1.0",
        "ollama_running": ollama_up,
        "routing_table":  {k: v["models"] for k, v in ROUTING_TABLE.items()}
    }


# ── CORS MIDDLEWARE ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"]
)


# ── OLLAMA AVAILABILITY CHECK ────────────────────────────────

def _check_ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _patch_routing_table_if_ollama_missing():
    ollama_up = _check_ollama_available()
    if ollama_up:
        print("[STARTUP] Ollama is running — local model available")
        return

    print("[STARTUP] Ollama not detected — removing from routing table")
    for category, config in ROUTING_TABLE.items():
        original = config["models"]
        patched  = [m for m in original if m != "ollama"]
        if len(patched) == 0:
            patched = ["groq"]
        if patched != original:
            config["models"] = patched
            print(f"[STARTUP]   {category}: {original} -> {patched}")


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("  FUSION 1.0 v1.0 — Starting up...")
    print("=" * 50)
    _patch_routing_table_if_ollama_missing()
    print("[STARTUP] Ready.")


# ── REQUEST / RESPONSE MODELS ───────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    user_id:  str = "anonymous"


# FIX L19: Added `answerer` field. Without it, the return statement
# in /ask passed answerer=... to FusionResponse, which raised a
# Pydantic ValidationError on every non-identity-guard response.
class FusionResponse(BaseModel):
    final_answer:       str
    category:           str
    complexity:         str
    strategy:           str
    models_used:        list
    individual_answers: list
    confidence_scores:  list
    fusion_weights:     list
    fusion_notes:       str
    latency_seconds:    float
    cost_estimate:      str
    answerer:           str = ""   # FIX L19


# ── COST ESTIMATION ─────────────────────────────────────────
_COST_TABLE = {
    "groq":            "free tier",
    "gemini":          "free tier",
    "gemini_creative": "free tier",
    "ollama":          "local / free",
}

def _estimate_cost(models_used: list) -> str:
    if not models_used:
        return "₹0"
    paid = [m for m in models_used if m == "openai"]
    if paid:
        return f"~₹{0.08 * len(paid):.2f} (OpenAI calls)"
    return "₹0 (all free-tier / local models)"


# ── OUTPUT VALIDATION ────────────────────────────────────────

def _validate_fusion_output(
    fusion_result: dict,
    dispatch_result: dict
) -> dict:
    """
    FIX L20: Post-fusion output guard.

    Catches three failure modes that previously reached the client:
    1. Empty string answer
    2. Whitespace-only answer
    3. Answer shorter than 3 words (almost certainly a pipeline artifact)

    In any of these cases, falls back to the highest-confidence
    individual model answer and appends a note to fusion_notes.
    """
    answer = fusion_result.get("answer", "")

    is_empty     = not answer or not answer.strip()
    is_too_short = len(answer.strip().split()) < 3

    if is_empty or is_too_short:
        print(
            f"[VALIDATION] ⚠️  Fusion produced {'empty' if is_empty else 'too-short'} output. "
            f"Falling back to best individual model answer."
        )

        answers     = dispatch_result.get("answers", [])
        confidences = dispatch_result.get("confidence_scores", [])

        if answers and confidences:
            best_idx = confidences.index(max(confidences))
            fusion_result["answer"] = answers[best_idx]
            fusion_result["notes"]  = (
                fusion_result.get("notes", "") +
                " [⚠️ Fallback: fusion produced empty/short output — using best individual answer]"
            )
        else:
            fusion_result["answer"] = "Unable to generate a response. Please try again."
            fusion_result["notes"]  = "All fallback sources were empty."

    return fusion_result


# ── ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Fusion 1.0 v1.1 is running",
        "status":  "online",
        "docs":    "Visit /docs for API documentation",
    }


@app.post("/ask", response_model=FusionResponse)
async def ask_fusion(request: QuestionRequest):
    """
    MAIN ENDPOINT — classify -> dispatch -> fuse -> validate -> return.

    v1.1 changes:
    - Polish step removed (was overwriting fusion output with single model).
    - Post-fusion output validation added (FIX L20).
    - FusionResponse schema fixed to include `answerer` (FIX L19).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Question too long (max {MAX_QUESTION_LENGTH} chars)")

    start_time = time.time()

    try:
        print(f"\n[FUSION 1.0] New query: {request.question[:60]}...")

        # ── STEP 0: IDENTITY GUARD ─────────────────────────
        print("[STEP 0] Checking identity guard...")
        guard_response = check_identity_guard(request.question)
        if guard_response:
            elapsed = round(time.time() - start_time, 2)
            log_query(
                question=request.question,
                classification={"category": "system_meta", "confidence": 1.0},
                dispatch_result={
                    "models_used":       ["identity_guard"],
                    "answers":           [guard_response["final_answer"]],
                    "confidence_scores": [1.0],
                    "strategy":          "identity_guard"
                },
                fusion_result={
                    "answer":   guard_response["final_answer"],
                    "strategy": "identity_guard",
                    "notes":    "Intercepted by identity guard",
                    "weights":  [1.0]
                },
                latency=elapsed
            )
            return FusionResponse(
                final_answer=guard_response["final_answer"],
                category="system_meta",
                complexity="low",
                strategy="identity_guard",
                models_used=["identity_guard"],
                individual_answers=[guard_response["final_answer"]],
                confidence_scores=[1.0],
                fusion_weights=[1.0],
                fusion_notes="Identity guard intercepted meta-question",
                latency_seconds=elapsed,
                cost_estimate="₹0 (no API call)",
                answerer="Fusion Identity Guard"
            )

        # ── STEP 1: CLASSIFY ───────────────────────────────
        print("[STEP 1] Classifying question...")
        classification = await classify_question(request.question)
        print(
            f"[STEP 1] Result: {classification['category']} / "
            f"{classification['complexity']} / conf={classification['confidence']:.2f}"
        )

        # ── STEP 2: DISPATCH ───────────────────────────────
        print("[STEP 2] Dispatching to models in parallel...")
        dispatch_result = await dispatch_parallel(
            question=request.question,
            category=classification["category"],
            complexity=classification["complexity"]
        )

        n_answers = len(dispatch_result["answers"])
        print(f"[STEP 2] Got {n_answers} answer(s) from: {dispatch_result['models_used']}")

        if n_answers == 0:
            raise HTTPException(
                status_code=503,
                detail="All models failed to respond. Check your API keys and Ollama status."
            )

        # ── STEP 3: FUSE ───────────────────────────────────
        # HOW ESCF INTEGRATES HERE:
        #
        # fuse_answers() (in fusion2.py) uses escf_engine to:
        #   1. Compute semantic_entropy(answers)
        #      → Low = models agree semantically
        #      → High = models say different things
        #
        #   2. Compute confidence_spread(confidences)
        #      → Low = models equally confident/uncertain
        #      → High = one model far more confident than others
        #
        #   3. Classify into epistemic state:
        #      CONSENSUS          → all agree + confident      → use majority_vote()
        #      CONFIDENT_DISSENTER → all agree but one trusts it more → trust that one
        #      COLLECTIVE_DOUBT    → all disagree equally        → use debate_merge()
        #      EPISTEMIC_VOID      → chaos on both fronts        → fallback + flag review
        #
        # The fusion_result dict includes:
        #   - answer: final merged answer (from selected strategy)
        #   - epistemic_state: which quadrant the models fell into
        #   - escf_metrics: raw entropy/spread values (for debugging)
        #   - strategy: which fusion strategy was applied
        #   - weights: how much each model contributed
        #   - notes: explanation of what happened
        #
        # EXAMPLE CLIENT-FACING RESPONSE:
        # If models say conflicting things with split confidence:
        #   → epistemic_state: "Epistemic_Void"
        #   → fusion_notes: "High disagreement AND unequal confidence. 
        #                    Returning highest-confidence model. Human review recommended."
        #   → final_answer: [that model's answer]
        #
        print("[STEP 3] Fusing answers...")
        fusion_result = fuse_answers(
            question=request.question,
            answers=dispatch_result["answers"],
            confidences=dispatch_result["confidence_scores"],
            category=classification["category"]
        )
        print(f"[STEP 3] Strategy: {fusion_result['strategy']}")
        print(f"[STEP 3] Epistemic State: {fusion_result.get('epistemic_state', 'N/A')}")
        print(f"[STEP 3] Notes: {fusion_result.get('notes', '')}")

        # ── STEP 3b: VALIDATE OUTPUT ───────────────────────
        # FIX L20: Catch empty/garbage fusion output before it reaches the client
        fusion_result = _validate_fusion_output(fusion_result, dispatch_result)

        # ── STEP 4: LOG ────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        log_query(
            question=request.question,
            classification=classification,
            dispatch_result=dispatch_result,
            fusion_result=fusion_result,
            latency=elapsed
        )
        print(f"[DONE] Total latency: {elapsed}s")

        # ── STEP 5: RETURN ─────────────────────────────────
        return FusionResponse(
            final_answer=fusion_result["answer"],
            category=classification["category"],
            complexity=classification["complexity"],
            strategy=dispatch_result["strategy"],
            models_used=dispatch_result["models_used"],
            individual_answers=dispatch_result["answers"],
            confidence_scores=dispatch_result["confidence_scores"],
            fusion_weights=fusion_result.get("weights", []),
            fusion_notes=fusion_result.get("notes", ""),
            latency_seconds=elapsed,
            cost_estimate=_estimate_cost(dispatch_result["models_used"]),
            answerer=f"Fusion Dispatcher ({', '.join(dispatch_result['models_used'])})"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
def get_logs(limit: int = 20):
    """Returns recent query logs from logger.jsonl."""
    logs = []
    try:
        with open("logger.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
    except FileNotFoundError:
        pass
    return {"count": len(logs), "logs": logs}


@app.get("/stats")
def get_stats():
    """Aggregate performance stats from all logged queries."""
    stats      = defaultdict(list)
    total      = 0
    models     = defaultdict(int)
    strategies = defaultdict(int)

    try:
        with open("logger.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                total += 1
                cat   = entry.get("category", "unknown")
                stats[cat].append(entry.get("total_latency_seconds", 0))
                for m in entry.get("models_used", []):
                    models[m] += 1
                strat = entry.get("fusion_strategy", "unknown")
                strategies[strat] += 1
    except FileNotFoundError:
        pass

    return {
        "total_queries": total,
        "model_usage":   dict(models),
        "strategies":    dict(strategies),
        "by_category": {
            cat: {
                "count":       len(times),
                "avg_latency": round(sum(times) / len(times), 2) if times else 0,
            }
            for cat, times in stats.items()
        }
    }


# ── RUN SERVER ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  FUSION 1.0 v1.1 — Starting server...")
    print(f"  Open http://{SERVER_HOST}:{SERVER_PORT} in your browser")
    print(f"  API docs at http://{SERVER_HOST}:{SERVER_PORT}/docs")
    print("=" * 50)

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True
    )