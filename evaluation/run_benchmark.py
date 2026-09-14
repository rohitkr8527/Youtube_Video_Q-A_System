from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from groq import RateLimitError
from app.config import get_settings
from app.generation.answer import AnswerGenerator
from app.generation.citations import build_citations
from app.llm.groq_client import GroqLLM
from app.reasoning.evidence_grader import EvidenceGrader
from app.reasoning.grounding_checker import GroundingChecker
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import get_reranker
from app.schemas.chat import ChatRequest
from app.services import RAGService
from evaluation.benchmark_models import BenchmarkCase, JudgeScores
from evaluation.metric_utils import (
    citation_hit_counts,
    hit_at_k,
    percentile,
    reciprocal_rank_by_time,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "benchmark_dataset.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"
RAW_OUTPUT = RESULTS_DIR / "raw_results.jsonl"
METRICS_OUTPUT = RESULTS_DIR / "metrics.json"
REPORT_OUTPUT = RESULTS_DIR / "evaluation_report.md"

JUDGE_SYSTEM = """You are an evaluator for a grounded YouTube question-answering system.
Return two scores between 0 and 1.

faithfulness: how completely the generated answer is supported by the supplied retrieved transcript evidence. Unsupported factual claims lower this score. A correct abstention on an unanswerable question is fully faithful.

answer_relevance: how directly and completely the generated answer addresses the user's question, using the reference answer only as a benchmark of what a good response should cover. A correct abstention on an unanswerable question is fully relevant.

Do not reward outside knowledge. Judge only the supplied material.
"""


def _load_cases() -> list[BenchmarkCase]:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET} does not exist. Run: python evaluation/prepare_ground_truth.py"
        )
    cases: list[BenchmarkCase] = []
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(BenchmarkCase.model_validate_json(line))
    return cases


def _judge(llm: GroqLLM, case: BenchmarkCase, answer: str, evidence) -> JudgeScores:
    evidence_text = "\n\n".join(
        f"[{c.start_time:.1f}s-{c.end_time:.1f}s] {c.text}" for c in evidence
    ) or "No transcript evidence was used."
    return llm.structured(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question:\n{case.question}\n\n"
                    f"Reference answer:\n{case.reference_answer}\n\n"
                    f"Generated answer:\n{answer}\n\n"
                    f"Retrieved evidence:\n{evidence_text}"
                ),
            },
        ],
        JudgeScores,
        name="benchmark_judge",
        reasoning_effort="low",
        max_tokens=500,
    )


def _run_case(case: BenchmarkCase, rag: RAGService, llm: GroqLLM) -> dict:
    settings = get_settings()
    request = ChatRequest(video_id=case.video_id, question=case.question, history=[])
    router, grader, answerer, grounding, _summaryer, _noteser, _quizzer = rag._llm_components()

    started = time.perf_counter()
    plan = router.plan(case.question, [])

    # Benchmark questions are designed for Q&A. If the router selects a content mode,
    # force a standalone retrieval query instead of evaluating summary/notes generation.
    if plan.route in {"SUMMARY", "NOTES", "QUIZ"}:
        plan.route = "EXPLANATION"

    initial_evidence = rag._retrieve_plan(case.video_id, plan)
    initial_hit5 = hit_at_k(initial_evidence, case.relevant_timestamps, 5) if case.answerable else 0.0

    initial_grade = grader.grade(plan.standalone_query, initial_evidence)
    final_grade = initial_grade
    evidence = initial_evidence
    correction_attempted = False

    if initial_grade.label == "PARTIALLY_RELEVANT" and settings.max_retrieval_retries > 0:
        correction_attempted = True
        corrected = rag.retriever.retrieve(
            video_id=case.video_id,
            query=initial_grade.improved_query,
            start_time=plan.start_time if plan.use_timestamp_filter else None,
            end_time=plan.end_time if plan.use_timestamp_filter else None,
        )
        combined = reciprocal_rank_fusion([initial_evidence, corrected], limit=16)
        evidence = get_reranker().rerank(plan.standalone_query, combined, settings.rerank_top_k)
        final_grade = grader.grade(plan.standalone_query, evidence)

    insufficient = final_grade.label == "IRRELEVANT" or not evidence
    grounded = False
    generation_retry = False

    if insufficient:
        answer = "I couldn't find enough information in this video to answer that reliably."
        citations = []
    else:
        answer = answerer.generate(case.question, evidence, [])
        check = grounding.check(case.question, answer, evidence)
        grounded = check.grounded
        if not check.grounded and settings.max_generation_retries > 0:
            generation_retry = True
            instruction = "Remove or correct these unsupported claims: " + "; ".join(check.unsupported_claims[:6])
            answer = answerer.generate(case.question, evidence, [], instruction)
            check = grounding.check(case.question, answer, evidence)
            grounded = check.grounded
        if not grounded:
            answer = "I found related parts of the video, but I couldn't verify a reliable answer from them."
            citations = []
            insufficient = True
        else:
            citations = build_citations(evidence, 4)

    elapsed_ms = (time.perf_counter() - started) * 1000
    final_hit1 = hit_at_k(evidence, case.relevant_timestamps, 1) if case.answerable else 0.0
    final_hit3 = hit_at_k(evidence, case.relevant_timestamps, 3) if case.answerable else 0.0
    final_hit5 = hit_at_k(evidence, case.relevant_timestamps, 5) if case.answerable else 0.0
    mrr = reciprocal_rank_by_time(evidence, case.relevant_timestamps) if case.answerable else 0.0
    citation_correct, citation_total = citation_hit_counts(citations, case.relevant_timestamps) if case.answerable else (0, 0)

    judge = _judge(llm, case, answer, evidence)

    # Recovery is deliberately strict: the first retrieval missed the ground truth,
    # correction was triggered, and the corrected retrieval then found it.
    correction_recovered = bool(correction_attempted and initial_hit5 == 0 and final_hit5 > 0)

    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "video_title": case.video_title,
        "question": case.question,
        "category": case.category,
        "difficulty": case.difficulty,
        "answerable": case.answerable,
        "reference_answer": case.reference_answer,
        "relevant_timestamps": [r.model_dump() for r in case.relevant_timestamps],
        "route": plan.route,
        "standalone_query": plan.standalone_query,
        "initial_grade": initial_grade.label,
        "final_grade": final_grade.label,
        "correction_attempted": correction_attempted,
        "correction_recovered": correction_recovered,
        "generation_retry": generation_retry,
        "insufficient_evidence": insufficient,
        "grounded": grounded,
        "answer": answer,
        "sources": [c.model_dump() for c in citations],
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": c.chunk_id,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "topic": c.topic,
                "rerank_score": c.rerank_score,
            }
            for rank, c in enumerate(evidence, start=1)
        ],
        "initial_recall_at_5": initial_hit5,
        "recall_at_1": final_hit1,
        "recall_at_3": final_hit3,
        "recall_at_5": final_hit5,
        "mrr": mrr,
        "citation_correct": citation_correct,
        "citation_total": citation_total,
        "faithfulness": judge.faithfulness,
        "answer_relevance": judge.answer_relevance,
        "judge_reason": judge.reason,
        "latency_ms": round(elapsed_ms, 2),
    }


def _aggregate(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    corrections = [r for r in answerable if r["correction_attempted"] and r["initial_recall_at_5"] == 0]

    citation_correct = sum(r["citation_correct"] for r in answerable)
    citation_total = sum(r["citation_total"] for r in answerable)
    correct_abstentions = sum(bool(r["insufficient_evidence"]) for r in unanswerable)
    recovered = sum(bool(r["correction_recovered"]) for r in corrections)

    return {
        "videos": len({r["video_id"] for r in rows}),
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "recall_at_1": mean(r["recall_at_1"] for r in answerable) if answerable else 0.0,
        "recall_at_3": mean(r["recall_at_3"] for r in answerable) if answerable else 0.0,
        "recall_at_5": mean(r["recall_at_5"] for r in answerable) if answerable else 0.0,
        "mrr": mean(r["mrr"] for r in answerable) if answerable else 0.0,
        "faithfulness": mean(r["faithfulness"] for r in answerable) if answerable else 0.0,
        "answer_relevance": mean(r["answer_relevance"] for r in answerable) if answerable else 0.0,
        "citation_hit_rate": citation_correct / citation_total if citation_total else 0.0,
        "citation_correct": citation_correct,
        "citation_total": citation_total,
        "abstention_accuracy": correct_abstentions / len(unanswerable) if unanswerable else 0.0,
        "correct_abstentions": correct_abstentions,
        "correction_attempts_after_initial_miss": len(corrections),
        "correction_recoveries": recovered,
        "correction_recovery_rate": recovered / len(corrections) if corrections else 0.0,
        "p50_latency_ms": percentile((r["latency_ms"] for r in rows), 50),
        "p95_latency_ms": percentile((r["latency_ms"] for r in rows), 95),
    }


def _report(metrics: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    sec = lambda ms: f"{ms / 1000:.2f} s"
    return f"""# VideoRAG Evaluation Report

## Benchmark

- Videos: **{metrics['videos']}**
- Questions: **{metrics['questions']}**
- Answerable: **{metrics['answerable_questions']}**
- Unanswerable: **{metrics['unanswerable_questions']}**

## Retrieval

| Metric | Result |
|---|---:|
| Recall@1 | **{pct(metrics['recall_at_1'])}** |
| Recall@3 | **{pct(metrics['recall_at_3'])}** |
| Recall@5 | **{pct(metrics['recall_at_5'])}** |
| MRR | **{metrics['mrr']:.3f}** |

## Answer Quality

| Metric | Result |
|---|---:|
| Faithfulness | **{metrics['faithfulness']:.3f}** |
| Answer Relevance | **{metrics['answer_relevance']:.3f}** |

Faithfulness and answer relevance are judged by GPT-OSS 120B from the question, generated answer, reference answer, and retrieved transcript evidence. Spot-check a random sample before publishing these numbers.

## Grounding & Safety

| Metric | Result |
|---|---:|
| Timestamp Citation Hit Rate | **{pct(metrics['citation_hit_rate'])}** ({metrics['citation_correct']}/{metrics['citation_total']}) |
| Abstention Accuracy | **{pct(metrics['abstention_accuracy'])}** ({metrics['correct_abstentions']}/{metrics['unanswerable_questions']}) |

## Corrective RAG

- Correction attempts after an initial Recall@5 miss: **{metrics['correction_attempts_after_initial_miss']}**
- Successful recoveries: **{metrics['correction_recoveries']}**
- Correction Recovery Rate: **{pct(metrics['correction_recovery_rate'])}**

A recovery counts only when the first retrieval misses the ground-truth timestamp, corrective retrieval is triggered, and the corrected top-5 then contains the ground-truth section.

## Latency

| Metric | Result |
|---|---:|
| P50 end-to-end latency | **{sec(metrics['p50_latency_ms'])}** |
| P95 end-to-end latency | **{sec(metrics['p95_latency_ms'])}** |

## Key Metrics

Use these only after reviewing the benchmark dataset and spot-checking evaluation outputs:

> Evaluated VideoRAG across **{metrics['questions']} questions from {metrics['videos']} technical videos**, achieving **{pct(metrics['recall_at_5'])} Recall@5**, **{metrics['faithfulness']:.2f} faithfulness**, and **{pct(metrics['citation_hit_rate'])} timestamp citation hit rate**.

> The grounded abstention pipeline achieved **{pct(metrics['abstention_accuracy'])} abstention accuracy** on unsupported questions, with **{pct(metrics['correction_recovery_rate'])} corrective-retrieval recovery rate** on initial retrieval misses.

## Important Methodology Note

Ground-truth answerable questions are generated from stored transcript windows rather than from retrieval results. Review the generated benchmark and spot-check the evaluation output before publishing metrics.
"""


def _run_case_with_retry(case: BenchmarkCase, rag: RAGService, llm: GroqLLM) -> dict:
    for attempt in range(4):
        try:
            return _run_case(case, rag, llm)
        except RateLimitError as error:
            retry_after = error.response.headers.get("retry-after") if hasattr(error, "response") and error.response else None
            try:
                wait_seconds = float(retry_after) + 1.0 if retry_after is not None else 60.0
            except ValueError:
                wait_seconds = 60.0
            if attempt >= 3:
                raise
            print(f"  Groq rate limit hit. Waiting {wait_seconds:.0f}s before retry ({3 - attempt} left)...", flush=True)
            time.sleep(wait_seconds)
    return _run_case(case, rag, llm)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    if not cases:
        raise ValueError("Benchmark dataset is empty.")

    rag = RAGService()
    llm = GroqLLM()
    rows: list[dict] = []
    completed_case_ids: set[str] = set()

    if RAW_OUTPUT.exists():
        for line in RAW_OUTPUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "error" not in data and "case_id" in data:
                        completed_case_ids.add(data["case_id"])
                        rows.append(data)
                except Exception:
                    pass

    if completed_case_ids:
        print(f"Loaded {len(completed_case_ids)} completed cases from {RAW_OUTPUT}")

    with RAW_OUTPUT.open("a" if completed_case_ids else "w", encoding="utf-8") as raw_handle:
        for index, case in enumerate(cases, start=1):
            if case.case_id in completed_case_ids:
                continue
            print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}")
            try:
                row = _run_case_with_retry(case, rag, llm)
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "video_id": case.video_id,
                    "video_title": case.video_title,
                    "question": case.question,
                    "answerable": case.answerable,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {row['error']}")
            rows.append(row)
            raw_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            raw_handle.flush()

    successful = [r for r in rows if "error" not in r]
    if not successful:
        raise RuntimeError("All benchmark cases failed. Check raw_results.jsonl for errors.")

    metrics = _aggregate(successful)
    metrics["failed_cases"] = len(rows) - len(successful)
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_OUTPUT.write_text(_report(metrics), encoding="utf-8")

    print("\nBenchmark complete.")
    print(f"Metrics: {METRICS_OUTPUT}")
    print(f"Report:  {REPORT_OUTPUT}\n")
    print(_report(metrics))


if __name__ == "__main__":
    main()
