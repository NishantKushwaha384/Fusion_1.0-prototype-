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



*.pyc
.DS_Store
```
