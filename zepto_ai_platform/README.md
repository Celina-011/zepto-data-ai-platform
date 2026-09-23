# Zepto Data & AI Platform

A single repository containing the three capstone modules:

- `data_pipeline/` — scraping, cleaning, currency conversion, SQLite, SQL and pandas.
- `analytics/` — Titanic EDA, preprocessing, classification, imbalance handling, tuning and regression.
- `support_assistant/` — local embeddings, ChromaDB, LangGraph, FastAPI and deterministic mock LLM mode.

## Setup

Create a virtual environment and install each module's requirements:

```bash
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```

## Run

### Data pipeline

```bash
cd data_pipeline
python main.py
```

The required currency baseline is **1 GBP = 105.50 INR**. No external currency API is required.

### Analytics

```bash
cd analytics
python main.py
```

The raw Titanic dataset is loaded once through Seaborn and saved as `titanic.csv` for offline reuse.

### Support assistant

```bash
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then POST to `/ask` with:

```json
{"query": "How much is standard delivery?"}
```

The graded baseline leaves `MOCK_LLM` unset (equivalent to mock mode).

## Git workflow

Create a feature branch, make at least two commits on it, then merge it into `main`:

```bash
git checkout -b feature/zepto-capstone
git add .
git commit -m "Build capstone modules"
git add .
git commit -m "Add documentation and validation"
git checkout main
git merge --no-ff feature/zepto-capstone -m "Merge Zepto capstone feature"
```
## Project Status

The three capstone modules are organized as separate folders:
data_pipeline, analytics, and support_assistant.
## Module Structure

- `data_pipeline/` — web scraping, cleaning, SQLite storage, and SQL analysis.
- `analytics/` — Titanic EDA, classification, and regression analysis.
- `support_assistant/` — document retrieval and support-question answering.