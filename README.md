# VideoRAG(Youtube Video Q&A System)

VideoRAG is an advanced, user-focused YouTube question-answering system built as an AI engineering project rather than a basic vector-search demo.

It uses timestamp-aware semantic chunking, dense + sparse hybrid retrieval, cross-encoder reranking, agentic query planning, self-corrective retrieval, grounded generation, evaluation, and internal tracing. The frontend stays intentionally simple and hides backend implementation details from users.

## Product features

- Paste a YouTube URL and prepare the video for questions.
- Ask natural follow-up questions in a chat interface.
- Receive grounded answers with clickable timestamp sources.
- Generate a structured summary.
- Generate study notes.
- Take an interactive multiple-choice quiz.
- Automatically retry retrieval once when evidence is weak.
- Refuse unsupported answers instead of hallucinating.

## AI engineering architecture

1. YouTube transcript ingestion with timestamps.
2. Semantic + timestamp-aware chunking.
3. BGE dense embeddings stored in persistent local Qdrant.
4. BM25 sparse retrieval.
5. Reciprocal Rank Fusion.
6. Cross-encoder reranking.
7. GPT-OSS 120B query planning and rewriting.
8. Evidence grading and corrective retrieval.
9. GPT-OSS 120B grounded answer generation.
10. Grounding verification with one controlled regeneration.
11. Retrieval evaluation and JSONL tracing.

Only one LLM is used throughout the project:

```text
openai/gpt-oss-120b via Groq
```

## Requirements

- Python 3.11+
- A Groq API key
- Internet access on first run to download embedding/reranker models and fetch YouTube transcripts

Qdrant runs in persistent local mode, so the project does not need a separate vector database service.

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Copy `.env.example` to `.env` and add your Groq key:

```env
GROQ_API_KEY=your_key_here
```

### 4. Start the complete product

```bash
python run_app.py
```

Then open:

```text
http://localhost:8501
```

You can also start the services separately:

```bash
uvicorn app.main:app --reload
streamlit run frontend/app.py
```

## API endpoints

```text
GET  /health
POST /videos/process
GET  /videos/{video_id}
POST /chat
POST /summary
POST /notes
POST /quiz
```

## Evaluation

An example dataset is included at `evaluation/datasets/sample.jsonl`.

Run retrieval evaluation after a video has been indexed:

```bash
python -m app.evaluation.evaluator --dataset evaluation/datasets/sample.jsonl
```

The evaluator reports Recall@K, Precision@K, MRR, and NDCG. Expected relevant chunk IDs in the dataset should be updated for the video you evaluate.

## Tests

```bash
pytest -q
```

Tests are designed to cover pure logic without requiring Groq or YouTube network calls.

## Project structure

```text
app/
  api/              FastAPI routes
  ingestion/        YouTube, transcript and chunking
  retrieval/        Dense, sparse, fusion and reranking
  reasoning/        Query planning, evidence grading, grounding
  generation/       Answers, summaries, notes, quizzes and citations
  database/         Qdrant and cache integrations
  evaluation/       Retrieval metrics and evaluator
  observability/    Structured tracing
frontend/           Streamlit product UI
evaluation/         Evaluation datasets/results
tests/              Unit tests
```

## Important design choices

- The Streamlit UI never shows vector scores, model names, databases, chunk IDs, agent states, token counts, or internal prompts.
- Corrective RAG is capped at one retrieval retry and one answer regeneration.
- Deterministic tasks such as timestamp formatting, RRF, filtering, and citation URL creation are done in Python, not with the LLM.
- Redis support is optional. If `REDIS_URL` is empty or Redis is unavailable, the application falls back to an in-memory cache.
- This version intentionally supports one active video experience at a time in the frontend. Multi-video knowledge bases and playlists are out of scope for now.

See `docs/ARCHITECTURE.md` for the finalized design specification.
