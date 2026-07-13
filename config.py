import os
from dotenv import load_dotenv

load_dotenv()

# ── Server ───────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# ── Input Limits ─────────────────────────────────────────────
MAX_QUESTION_LENGTH = 2000
MIN_ANSWER_WORDS = 3

# ── Similarity & Fusion Thresholds ───────────────────────────
SIMILARITY_THRESHOLD = 0.85          # remove_conflicts near-dup cutoff
SUPPORTING_FACT_SIMILARITY = 0.55    # reconstruct_paragraph inclusion cutoff
MIN_SUPPORT_CONFIDENCE = 0.40        # min confidence for secondary facts
DEBATE_AGREEMENT_THRESHOLD = 0.35    # debate_merge sentence similarity
DEBATE_CORE_WEIGHT = 0.6             # weight sum for core agreement
DEBATE_NUANCE_WEIGHT = 0.3           # weight sum for nuanced point
MIN_FACT_WORDS = 5                   # minimum words for a valid fact
MAX_STEP_CHARS = 500                 # max chars per procedural step
MERGE_MAX_WORDS = 18                 # max combined words for merging facts

# ── Classifier ───────────────────────────────────────────────
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.75  # below this triggers double-pass

# ── Model Configuration ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GROQ_CLASSIFIER_MODEL = "llama-3.3-70b-versatile"
GROQ_ANSWER_MODEL = "llama-3.3-70b-versatile"
GROQ_OPENAI_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
IDENTITY_GUARD_OLLAMA_URL = os.getenv("IDENTITY_GUARD_OLLAMA_URL", "http://localhost:11434/api/generate")
IDENTITY_GUARD_OLLAMA_MODEL = os.getenv("IDENTITY_GUARD_OLLAMA_MODEL", "mistral")

LLM_SYNTHESIZER_MODEL = "llama-3.1-8b-instant"
LLM_SYNTHESIZER_TIMEOUT = 5
LLM_SYNTHESIZER_MAX_TOKENS = 512

# ── Logging ──────────────────────────────────────────────────
LOG_FILE = "logger.jsonl"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ── Categories ───────────────────────────────────────────────
CATEGORIES = {
    "math":       "Mathematical calculations, equations, proofs, statistics",
    "coding":     "Writing code, debugging, explaining algorithms",
    "factual":    "Facts, history, science, definitions, general knowledge",
    "creative":   "Stories, poems, brainstorming, creative writing",
    "reasoning":  "Logic puzzles, analysis, 'why' questions, opinion",
    "procedural": "How-to, tutorials, step-by-step instructions, recipes, processes",
    "general":    "Casual questions, greetings, simple everyday queries",
}

# ── Routing Table ────────────────────────────────────────────
ROUTING_TABLE = {
    "math":       {"models": ["openai", "groq", "gemini", "ollama"], "strategy": "confidence_weighted"},
    "coding":     {"models": ["openai", "groq", "gemini", "ollama"], "strategy": "confidence_weighted"},
    "factual":    {"models": ["openai", "groq", "ollama", "gemini"], "strategy": "majority_vote"},
    "creative":   {"models": ["openai", "groq", "ollama", "gemini"], "strategy": "creative_blend"},
    "reasoning":  {"models": ["openai", "groq", "gemini", "ollama"], "strategy": "debate_merge"},
    "procedural": {"models": ["openai", "groq", "gemini", "ollama"], "strategy": "step_synthesis"},
    "general":    {"models": ["openai", "ollama", "groq"],           "strategy": "confidence_weighted"},
}

# ── Model Roles ──────────────────────────────────────────────
MODEL_ROLES = {
    "groq": {
        "name": "Primary Analyst",
        "prompt": (
            "You are the Primary Analyst in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide accurate, complete, and properly formatted answers.\n"
            "Focus on ACCURACY and CLARITY.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.87"
        ),
    },
    "gemini": {
        "name": "Creative Synthesizer",
        "prompt": (
            "You are the Creative Synthesizer in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide complete answers with practical examples and clear structure.\n"
            "Focus on CLARITY, USEFULNESS, and PROPER FORMATTING.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.91"
        ),
    },
    "ollama": {
        "name": "Local Validator",
        "prompt": (
            "You are the Local Validator in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide complete, well-structured answers with proper validation.\n"
            "Focus on CORRECTNESS and CLEAR FORMATTING.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.85"
        ),
    },
    "openai": {
        "name": "Advanced Reasoner",
        "prompt": (
            "You are the Advanced Reasoner in a multi-model AI system called Fusion 1.0.\n"
            "Your role: Provide deep, nuanced answers with comprehensive reasoning.\n"
            "Focus on DEPTH, ACCURACY, and COMPREHENSIVE EXPLANATION.\n"
            "At the end of your answer, on a new line write exactly:\n"
            "CONFIDENCE: [number between 0.0 and 1.0]\n"
            "Example: CONFIDENCE: 0.89"
        ),
    },
}

# ── Category-Specific Format Instructions ────────────────────
CATEGORY_FORMAT = {
    "math":       "This is a MATH question. Show step-by-step working. Use plain text notation only — no LaTeX, no \\[ \\], no \\boxed{}.",
    "coding":     "This is a CODING question. Provide complete, working code with explanation.",
    "factual":    "This is a FACTUAL question. Be precise, cite key facts, stay objective.",
    "creative":   "This is a CREATIVE question. Write imaginatively with vivid detail.",
    "reasoning":  "This is a REASONING question. Structure your argument with clear logic.",
    "procedural": "This is a PROCEDURAL question. Provide clear, numbered steps. Start with prerequisites.",
    "general":    "This is a GENERAL question. Be concise and conversational.",
}

# ── Token Limits Per Category ────────────────────────────────
TOKEN_LIMITS = {
    "math": 2000,
    "coding": 3000,
    "factual": 3000,
    "creative": 3000,
    "reasoning": 3000,
    "procedural": 3500,
    "general": 3000,
}

# ── Fusion Helpers ───────────────────────────────────────────
FILLER_MARKERS = ["something", "anything", "stuff", "things", "whatever", "etc"]
DEFINITION_SIGNALS = ["refers to", "is the systematic", "is a field", "is the development of", "is defined as", "is the process of"]
EXTENDED_KEYWORDS = ["explain", "detail", "long", "comprehensive", "why", "how", "list"]
CONNECTORS = ["Additionally,", "Also,", "In addition,", "Another key point is", "Notably,"]

# ── Identity Guard ──────────────────────────────────────────
SYSTEM_IDENTITY = {
    "name":    "Fusion 1.0",
    "creator": "Fusion Team",
    "models":  ["Groq (Llama3)", "Google Gemini Flash", "Ollama (Mistral)", "OpenAI (via Groq)"],
}

IDENTITY_GUARD_OLLAMA_TIMEOUT = float(os.getenv("IDENTITY_GUARD_OLLAMA_TIMEOUT", "6"))
IDENTITY_GUARD_MIN_CONFIDENCE = float(os.getenv("IDENTITY_GUARD_MIN_CONFIDENCE", "0.80"))
IDENTITY_GUARD_MAX_LENGTH = int(os.getenv("IDENTITY_GUARD_MAX_LENGTH", "1000"))

# ── CORS ─────────────────────────────────────────────────────
CORS_ORIGINS = [
    "https://nishantkushwaha384.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ── Cost Table ──────────────────────────────────────────────
COST_TABLE = {
    "groq":   "free tier",
    "gemini": "free tier",
    "ollama": "local / free",
    "openai": "free tier (via Groq)",
}

