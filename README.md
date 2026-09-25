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

<CodeBlock language="markdown">

#### Support Assistant Architecture

The Zepto Policy Support Assistant implements a local retrieval-augmented generation (RAG) workflow.

- **Document ingestion:** Loads eight policy documents from `support_assistant/docs/` and splits them into overlapping text chunks.
- **Embeddings:** Uses the local `all-MiniLM-L6-v2` SentenceTransformer model to generate normalized embeddings.
- **Vector database:** ChromaDB stores document chunks, embeddings, and source metadata.
- **Workflow:** LangGraph uses three nodes: `classify_intent`, `retrieve_and_answer`, and `direct_answer`, with conditional routing.
- **Response validation:** Pydantic validates the response fields `answer`, `sources`, and `confidence`.

#### API Examples

The API uses the `POST /ask` endpoint with a JSON request containing the `query` field.

Example requests and responses are saved in `support_assistant/api_examples.txt`.

#### Mock Mode

The application defaults to deterministic mock mode with `MOCK_LLM=1`. No paid LLM API key is required for the graded baseline.

#### Docker

From the `support_assistant` directory, build and run the application:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
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