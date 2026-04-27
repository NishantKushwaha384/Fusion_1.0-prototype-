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
