# VideoRAG

VideoRAG is a YouTube question-answering application that produces grounded answers with clickable timestamp citations. It combines transcript ingestion, hybrid retrieval, reranking, query planning, evidence grading, corrective retrieval, and grounding verification behind a Streamlit interface and FastAPI backend.

## Features

- Process a YouTube video from its URL.
- Ask questions and maintain conversational context.
- Cite supporting transcript sections with clickable timestamps.
- Generate summaries, study notes, and multiple-choice quizzes.
- Retry retrieval when the initial evidence is weak.
- Decline unsupported questions when reliable evidence is unavailable.
- Record structured traces for retrieval and generation operations.

## Architecture

The request pipeline consists of:

1. Fetching timestamped YouTube transcripts.
2. Semantic and timestamp-aware transcript chunking.
3. Dense retrieval with BGE embeddings and local Qdrant storage.
4. Sparse BM25 retrieval.
5. Reciprocal Rank Fusion and cross-encoder reranking.
6. Query planning, rewriting, and optional timestamp filtering.
7. Evidence grading and one corrective-retrieval attempt.
8. Grounded answer generation and one verification retry.
9. Deterministic timestamp citation construction.

LLM operations use `openai/gpt-oss-120b` through Groq. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation details.

## Requirements

- Python 3.11 or newer
- A Groq API key
- Internet access to retrieve YouTube transcripts and download embedding and reranking models

Qdrant runs in local persistent mode. Redis is optional; when `REDIS_URL` is empty or unavailable, the application uses an in-memory cache.

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set at least:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Other settings, including ports, model names, storage paths, and Redis, can be configured through `.env.example`.

## Run the application

Start the API and interface together:

```bash
python run_app.py
```

Open `http://localhost:8501`. The API runs at `http://127.0.0.1:8000` by default.

To run the services separately:

```bash
uvicorn app.main:app --reload
streamlit run frontend/app.py
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/videos/process` | Ingest and index a video |
| `GET` | `/videos/{video_id}` | Read stored video metadata |
| `POST` | `/chat` | Answer a grounded question |
| `POST` | `/summary` | Generate a video summary |
| `POST` | `/notes` | Generate study notes |
| `POST` | `/quiz` | Generate a quiz |

## Evaluation

The evaluation workflow uses 120 timestamp-grounded cases across 10 technical videos: 100 answerable questions and 20 unsupported questions.

```bash
python -m evaluation.prepare_ground_truth
python -m evaluation.review_dataset
python -m evaluation.run_benchmark
```

The benchmark reports Recall@1/3/5, MRR, faithfulness, answer relevance, timestamp citation hit rate, abstention accuracy, corrective-retrieval recovery rate, and P50/P95 latency. Completed case IDs are preserved in `evaluation/results/raw_results.jsonl`, so rerunning the benchmark skips successful cases.

See [evaluation/README.md](evaluation/README.md) for methodology and output details.

## Tests

```bash
pytest -q
```

The tests cover deterministic logic without requiring Groq or YouTube network calls.

## Project structure

```text
app/
  api/              FastAPI routes
  database/         Local Qdrant and optional Redis integrations
  evaluation/       Reusable retrieval metrics and evaluator
  generation/       Answers, citations, summaries, notes, and quizzes
  ingestion/        YouTube metadata, transcripts, and chunking
  llm/              Groq client
  observability/    Structured tracing and metrics
  reasoning/        Query planning, evidence grading, and grounding
  retrieval/        Dense, sparse, fusion, and reranking stages
  schemas/          API and internal data models
frontend/           Streamlit interface
evaluation/         Benchmark preparation, execution, and results
tests/              Automated tests
data/               Generated video, vector, and trace data
```

## Runtime data

- `data/videos/`: video metadata and transcript chunks
- `data/qdrant/`: persistent vector collections
- `data/logs/traces.jsonl`: structured execution traces

Do not delete `data/videos/` or `data/qdrant/` while an evaluation is in progress because the benchmark reads the indexed videos from these locations.
