# ============================================================
# FUSION 1.0 — PROJECT STRUCTURE & SETUP GUIDE
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

## Setup (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add API keys
```bash
cp .env.template .env
# Open .env and fill in your keys
```

### 3. Run the server
```bash
python main.py
```

### 4. Test it
Open browser: http://localhost:8000
API docs:     http://localhost:8000/docs
View logs:    http://localhost:8000/logs

## API Usage

### Ask a question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the derivative of x squared?"}'
```

### Check system health
```bash
curl http://localhost:8000/health
```

## Development Status

| Component      | Status      | Notes                              |
|----------------|-------------|-------------------------------------|
| Classifier     | ✅ Working  | Improve prompts after testing       |
| Dispatcher     | ✅ Working  | Fill routing table from benchmarks  |
| Fusion Engine  | ⚠️ Placeholder | YOUR MAIN RESEARCH TASK            |
| Logger         | ✅ Working  | Collect data for 2 weeks first      |
| API Server     | ✅ Working  | Production-ready structure          |

## Your Research Tasks (in order)

1. Get all 3 API keys → test each model individually
2. Run 50 questions → study the logs
3. Find where models agree vs disagree
4. Discover your fusion algorithm pattern
5. Implement it in fusion.py
6. Benchmark: your system vs single models
7. Write up findings

## .gitignore
```
.env
logger.txt
__pycache__/
*.pyc
.DS_Store
```
