# ============================================================
# FUSION 1.0 — BACKEND ENTRY POINT
# ============================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
import requests
from collections import defaultdict

from classifier import classify_question
from dispatcher import dispatch_parallel
from fusion import fuse_answers
from logger import log_query
from identity_guard import check_identity_guard

# ── CREATE THE APP ──────────────────────────────────────────
app = FastAPI(
    title="Fusion 1.0",
    description="Adaptive multi-model AI meta-system",
    version="1.0.0"
)
@app.get("/health")
def health_check():
    from dispatcher import ROUTING_TABLE
    ollama_up = _check_ollama_available()
    return {
        "status":          "healthy",
        "version":         "1.1.0",
        "ollama_running":  ollama_up,
        "routing_table":   {k: v["models"] for k, v in ROUTING_TABLE.items()}
    }


# ── CORS MIDDLEWARE ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nishantkushwaha384.github.io",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*","ngrok-skip-browser-warning"]
)


# ── OLLAMA AVAILABILITY CHECK ────────────────────────────────
# Check once at startup whether Ollama is running locally.
# If not, patch the routing table to remove "ollama" from all
# model lists so it never silently drops during dispatch.

def _check_ollama_available() -> bool:
    """Ping Ollama's local server. Returns True if it responds."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _patch_routing_table_if_ollama_missing():
    """
    Remove 'ollama' from all routing table entries if Ollama
    is not running. Prevents silent 2-model degradation where
    you think you have 3 models but only get 2.
    """
    from dispatcher import ROUTING_TABLE
    ollama_up = _check_ollama_available()
    if ollama_up:
        print("[STARTUP] Ollama is running — local model available")
        return

    print("[STARTUP] Ollama not detected — removing from routing table")
    for category, config in ROUTING_TABLE.items():
        original = config["models"]
        patched  = [m for m in original if m != "ollama"]
        if len(patched) == 0:
            patched = ["groq"]   # always keep at least one model
        if patched != original:
            config["models"] = patched
            print(f"[STARTUP]   {category}: {original} -> {patched}")


@app.on_event("startup")
async def startup_event():
    """Runs once when the server boots."""
    print("=" * 50)
    print("  FUSION 1.0 v1.0 — Starting up...")
    print("=" * 50)
    _patch_routing_table_if_ollama_missing()
    print("[STARTUP] Ready.")


# ── REQUEST / RESPONSE MODELS ───────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    user_id: str = "anonymous"


class FusionResponse(BaseModel):
    final_answer: str
    category: str
    complexity: str
    strategy: str
    models_used: list
    individual_answers: list
    confidence_scores: list
    fusion_weights: list
    fusion_notes: str
    latency_seconds: float
    cost_estimate: str
    

# ── COST ESTIMATION ─────────────────────────────────────────
_COST_TABLE = {
    "groq":   "free tier",
    "gemini": "free tier",
    "ollama": "local / free",
    "openai": "free tier",
}

def _estimate_cost(models_used: list) -> str:
    if not models_used:
        return "₹0"
    paid = [m for m in models_used if m == "openai"]
    if paid:
        return f"~₹{0.08 * len(paid):.2f} (OpenAI calls)"
    return "₹0 (all free-tier / local models)"


# ── ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Fusion 1.0 v1.0 is running",
        "status":  "online",
        "docs":    "Visit /docs for API documentation",
    }




@app.post("/ask", response_model=FusionResponse)
async def ask_fusion(request: QuestionRequest):
    """
    MAIN ENDPOINT — classify -> dispatch -> fuse -> return.

    v1.1 change: Polish step REMOVED.
    ─────────────────────────────────────────────────────────
    The old code did:
        fusion_result = fuse_answers(...)
        polish_result = await polish_answer_with_groq(...)   # single Groq call
        fusion_result["answer"] = polish_result["answer"]   # OVERWROTE fusion

    This was the single biggest logical error in the system.
    The entire multi-model fusion pipeline was being discarded
    and replaced by one model's rewrite. Removed entirely.

    If you want polishing back, gate it behind a quality check:
        if fusion_result["strategy"] == "single": apply_polish()
    That way it only runs when fusion genuinely didn't happen.
    ─────────────────────────────────────────────────────────
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 chars)")

    start_time = time.time()

    try:
        print(f"\n[FUSION 1.0] New query: {request.question[:60]}...")

        # ── STEP 0: IDENTITY GUARD ─────────────────────────
        # Intercept system-identity meta-questions before they
        # reach the classifier and dispatcher. Return immediately
        # if matched.
        print("[STEP 0] Checking identity guard...")
        guard_response = check_identity_guard(request.question)
        if guard_response:
            elapsed = round(time.time() - start_time, 2)
            guard_response["latency_seconds"] = elapsed
            log_query(
                question=request.question,
                classification={"category": "system_meta", "confidence": 1.0},
                dispatch_result={"models_used": ["identity_guard"], "answers": [guard_response["final_answer"]], "confidence_scores": [1.0], "strategy": "identity_guard"},
                fusion_result={"answer": guard_response["final_answer"], "strategy": "identity_guard", "notes": "Intercepted by identity guard", "weights": [1.0]},
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
                detail=(
                    "All models failed to respond. "
                    "Check your API keys in .env and Ollama status."
                )
            )

        # ── STEP 3: FUSE ───────────────────────────────────
        # THIS IS THE FINAL STEP. No overwriting after this.
        print("[STEP 3] Fusing answers...")

        fusion_result = fuse_answers(
            question=request.question,
            answers=dispatch_result["answers"],
            confidences=dispatch_result["confidence_scores"],
            category=classification["category"]
        )

        print(f"[STEP 3] Strategy: {fusion_result['strategy']}")
        print(f"[STEP 3] Notes: {fusion_result.get('notes', '')}")

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
                cat = entry.get("category", "unknown")
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
    print("  Open http://localhost:8000 in your browser")
    print("  API docs at http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
