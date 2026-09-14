# VideoRAG Evaluation

This directory contains the timestamp-grounded evaluation workflow for VideoRAG.

## Benchmark design

The supplied manifest contains 10 technical YouTube videos. Ground-truth generation creates 12 cases for each video:

- 10 answerable questions grounded in transcript windows distributed across the video
- 2 plausible but unsupported questions used to test abstention

The completed dataset contains 120 cases: 100 answerable and 20 unsupported.

Ground truth is generated directly from stored transcript chunks, independently of the retriever being evaluated. Each answerable case includes a reference answer and the timestamp range that must support it.

## Metrics

- Recall@1, Recall@3, and Recall@5
- Mean Reciprocal Rank (MRR)
- Answer faithfulness and relevance
- Timestamp citation hit rate
- Abstention accuracy
- Corrective-retrieval recovery rate
- P50 and P95 end-to-end latency

Faithfulness and relevance are judged by `openai/gpt-oss-120b` using the question, reference answer, generated answer, and retrieved transcript evidence.

## 1. Prepare ground truth

From the project root, run:

```bash
python -m evaluation.prepare_ground_truth
```

The command ingests videos that are not already stored and writes `evaluation/benchmark_dataset.jsonl`. Progress is saved after each generated case. If the command stops or reaches an API quota, run it again; existing case IDs are skipped.

## 2. Review the dataset

Display every case with its reference answer and supporting timestamp:

```bash
python -m evaluation.review_dataset
```

Before publishing results, verify that:

- Each answerable question is clear and fully supported by its timestamp range.
- Reference answers contain no information outside the cited transcript window.
- Unsupported questions are related to the video topic but not answered in the video.
- Categories and difficulty labels are reasonable.

## 3. Run the benchmark

```bash
python -m evaluation.run_benchmark
```

Successful case IDs are read from the raw output when the command starts. Running it again skips those cases and continues with failed or unfinished cases.

The benchmark writes:

```text
evaluation/results/raw_results.jsonl
evaluation/results/metrics.json
evaluation/results/evaluation_report.md
```

`raw_results.jsonl` contains per-case retrieval, answer, citation, correction, grounding, judge, and latency data. `metrics.json` contains aggregate values, while `evaluation_report.md` provides a readable summary.

## Validate the results

A complete `metrics.json` should contain:

```text
videos: 10
questions: 120
answerable_questions: 100
unanswerable_questions: 20
failed_cases: 0
```

Spot-check generated answers, retrieved timestamps, citations, abstentions, and judge explanations in `raw_results.jsonl` before publishing the aggregate results.

## API limits

Ground-truth generation and benchmark execution call Groq and may stop when the account reaches its quota. Successful cases do not need to be regenerated. Wait for the quota window to reset, then run the same command again.
