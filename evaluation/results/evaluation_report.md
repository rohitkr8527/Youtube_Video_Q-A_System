# VideoRAG Evaluation Report

## Benchmark

- Videos: **10**
- Questions: **120**
- Answerable: **100**
- Unanswerable: **20**

## Retrieval

| Metric | Result |
|---|---:|
| Recall@1 | **88.0%** |
| Recall@3 | **100.0%** |
| Recall@5 | **100.0%** |
| MRR | **0.933** |

## Answer Quality

| Metric | Result |
|---|---:|
| Faithfulness | **0.985** |
| Answer Relevance | **0.989** |

Faithfulness and answer relevance are judged by GPT-OSS 120B from the question, generated answer, reference answer, and retrieved transcript evidence. Spot-check a random sample before publishing these numbers.

## Grounding & Safety

| Metric | Result |
|---|---:|
| Timestamp Citation Hit Rate | **52.7%** (198/376) |
| Abstention Accuracy | **90.0%** (18/20) |

## Corrective RAG

- Correction attempts after an initial Recall@5 miss: **0**
- Successful recoveries: **0**
- Correction Recovery Rate: **0.0%**

A recovery counts only when the first retrieval misses the ground-truth timestamp, corrective retrieval is triggered, and the corrected top-5 then contains the ground-truth section.

## Latency

| Metric | Result |
|---|---:|
| P50 end-to-end latency | **51.95 s** |
| P95 end-to-end latency | **85.75 s** |

## Key Metrics

Use these only after reviewing the benchmark dataset and spot-checking evaluation outputs:

> Evaluated VideoRAG across **120 questions from 10 technical videos**, achieving **100.0% Recall@5**, **0.99 faithfulness**, and **52.7% timestamp citation hit rate**.

> The grounded abstention pipeline achieved **90.0% abstention accuracy** on unsupported questions, with **0.0% corrective-retrieval recovery rate** on initial retrieval misses.

## Important Methodology Note

Ground-truth answerable questions are generated from stored transcript windows rather than from retrieval results. Review the generated benchmark and spot-check the evaluation output before publishing metrics.
