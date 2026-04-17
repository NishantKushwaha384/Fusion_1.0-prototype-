# ============================================================
# FUSION 1.0 — BACKEND ENTRY POINT
# ============================================================
# This is the main file. Run this to start the backend server.
# Command: python main.py
#
# What this file does:
#   - Creates the FastAPI web server
#   - Defines all API endpoints (routes)
#   - Connects frontend HTML to the backend Python
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import time

# Import our custom modules (files we build)
from classifier import classify_question
from dispatcher import dispatch_parallel
from fusion import fuse_answers
from logger import log_query

# ── CREATE THE APP ──────────────────────────────────────────
# FastAPI is a Python web framework. Think of it as the
# "bridge" between your HTML frontend and your Python logic.
app = FastAPI(
    title="Fusion 1.0",
    description="Adaptive multi-model AI meta-system",
    version="1.0.0"
)
@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Used to verify the server is running before sending real requests.
    Frontend calls this on startup.
    """
    return {
        "status": "healthy",
        "models_available": ["groq-llama3", "gemini-flash", "gpt4o-mini"],
        "version": "1.0.0"
    }
# ── CORS MIDDLEWARE ─────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# This allows your HTML frontend (running on a different port)
# to talk to this Python backend. Without this, the browser
# blocks the connection for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nishantkushwaha384.github.io/Fusion_1.0-prototype-/fusion.html"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SERVE FRONTEND ──────────────────────────────────────────
# This serves your HTML/CSS/JS files as static files.
# So when someone opens http://localhost:8000, they see your UI.
# Comment this out if you're using Streamlit instead of HTML.
# app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── REQUEST / RESPONSE MODELS ───────────────────────────────
# Pydantic models define the shape of data going in and out.
# FastAPI automatically validates these for you.

class QuestionRequest(BaseModel):
    """What the frontend sends to us"""
    question: str                    # The user's question (required)
    user_id: str = "anonymous"       # Optional: track different users

class FusionResponse(BaseModel):
    """What we send back to the frontend"""
    final_answer: str                # The fused answer
    category: str                    # Question type (math, coding etc)
    complexity: str                  # low / medium / high
    strategy: str                    # Which fusion strategy was used
    models_used: list                # Which models answered
    individual_answers: list         # Raw answers from each model
    confidence_scores: list          # Confidence per model (0.0 - 1.0)
    fusion_weights: list             # How much each model contributed
    latency_seconds: float           # Total time taken
    cost_estimate: str               # Approximate cost


# ── ENDPOINTS (API ROUTES) ──────────────────────────────────
# An endpoint is a URL that the frontend can call.
# @app.get("/url") = browser can GET this URL
# @app.post("/url") = frontend can POST data to this URL

@app.get("/")
def home():
    """
    Home page — just returns a welcome message.
    Open http://localhost:8000 in your browser to see this.
    """
    return {
        "message": "Fusion 1.0 is running",
        "status": "online",
        "docs": "Visit /docs for API documentation"
    }





@app.post("/ask", response_model=FusionResponse)
async def ask_fusion(request: QuestionRequest):
    """
    MAIN ENDPOINT — This is where all the magic happens.

    Flow:
    1. Receive question from frontend
    2. Classify the question (what type is it?)
    3. Route to the right models based on type
    4. Dispatch to all models in parallel
    5. Fuse the answers
    6. Return everything to frontend

    The frontend calls this like:
        fetch('/ask', { method: 'POST', body: JSON.stringify({ question: "..." }) })
    """

    # Validate input
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 chars)")

    start_time = time.time()

    try:
        # ── STEP 1: CLASSIFY ───────────────────────────────
        # Send question to cheapest model to figure out what type it is
        print(f"\n[FUSION 1.0] New query: {request.question[:60]}...")
        print("[STEP 1] Classifying question...")

        classification = await classify_question(request.question)

        print(f"[STEP 1] Result: {classification['category']} / {classification['complexity']} / conf={classification['confidence']}")

        # ── STEP 2: DISPATCH ───────────────────────────────
        # Send to all relevant models in PARALLEL (at the same time)
        print("[STEP 2] Dispatching to models in parallel...")

        dispatch_result = await dispatch_parallel(
            question=request.question,
            category=classification["category"],
            complexity=classification["complexity"]
        )

        print(f"[STEP 2] Got {len(dispatch_result['answers'])} answers")

        # ── STEP 3: FUSE ───────────────────────────────────
        # YOUR ALGORITHM GOES HERE eventually.
        # For now this is a placeholder fusion.
        print("[STEP 3] Fusing answers...")

        fusion_result = await fuse_answers(
            question=request.question,
            answers=dispatch_result["answers"],
            confidences=dispatch_result["confidence_scores"],
            category=classification["category"]
        )

        print("[STEP 3] Fusion complete")

        # ── STEP 4: LOG ────────────────────────────────────
        # Save everything to a log file for later analysis.
        # This log becomes your research dataset!
        elapsed = round(time.time() - start_time, 2)

        log_query(
            question=request.question,
            classification=classification,
            dispatch_result=dispatch_result,
            fusion_result=fusion_result,
            latency=elapsed
        )

        # ── STEP 5: RETURN ─────────────────────────────────
        return FusionResponse(
            final_answer=fusion_result["answer"],
            category=classification["category"],
            complexity=classification["complexity"],
            strategy=dispatch_result["strategy"],
            models_used=dispatch_result["models_used"],
            individual_answers=dispatch_result["answers"],
            confidence_scores=dispatch_result["confidence_scores"],
            fusion_weights=fusion_result["weights"],
            latency_seconds=elapsed,
            cost_estimate="~₹0"   # Update this when you track real costs
        )

    except Exception as e:
        # If anything goes wrong, return a proper error
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
def get_logs(limit: int = 20):
    """
    Returns recent query logs.
    Useful for studying your system's behavior during development.
    Open http://localhost:8000/logs in browser to see recent queries.
    """
    import json
    logs = []
    try:
        with open("logger.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Return last `limit` entries
            for line in lines[-limit:]:
                logs.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass  # No logs yet, that's fine
    return {"count": len(logs), "logs": logs}


@app.get("/stats")
def get_stats():
    """
    Returns aggregate statistics about your system's performance.
    Use this to understand which models are performing best.

    ─────────────────────────────────────────────────────────
    TODO (your research task):
    After you have 100+ logs, analyze:
    - Which category has the highest average confidence?
    - Which model contributes most to the fusion?
    - What is the average latency per category?
    This data will shape your fusion algorithm!
    ─────────────────────────────────────────────────────────
    """
    import json
    from collections import defaultdict

    stats = defaultdict(list)
    total = 0

    try:
        with open("logger.txt", "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                total += 1
                cat = entry.get("category", "unknown")
                stats[cat].append(entry.get("latency", 0))
    except FileNotFoundError:
        pass

    return {
        "total_queries": total,
        "by_category": {
            cat: {
                "count": len(times),
                "avg_latency": round(sum(times) / len(times), 2) if times else 0
            }
            for cat, times in stats.items()
        }
    }


# ── RUN SERVER ──────────────────────────────────────────────
# This block runs when you do: python main.py
# uvicorn is a fast web server for Python async apps
if __name__ == "__main__":
    print("=" * 50)
    print("  FUSION 1.0 — Starting server...")
    print("  Open http://localhost:8000 in your browser")
    print("  API docs at http://localhost:8000/docs")
    print("=" * 50)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # Accept connections from any device on network
        port=8000,
        reload=True        # Auto-restart when you change code (dev mode)
    )
