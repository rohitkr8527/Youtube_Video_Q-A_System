from __future__ import annotations

from pathlib import Path

from evaluation.benchmark_models import BenchmarkCase

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "benchmark_dataset.jsonl"


def _fmt(seconds: float) -> str:
    total = max(0, int(seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError("Run prepare_ground_truth.py first.")
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = BenchmarkCase.model_validate_json(line)
        ranges = ", ".join(f"{_fmt(r.start)}-{_fmt(r.end)}" for r in case.relevant_timestamps) or "N/A"
        print("=" * 88)
        print(f"{case.case_id} | {case.video_title}")
        print(f"Category: {case.category} | Difficulty: {case.difficulty} | Answerable: {case.answerable}")
        print(f"Question: {case.question}")
        print(f"Reference: {case.reference_answer}")
        print(f"Ground-truth time: {ranges}")
    print("\nTip: open the timestamp in YouTube while spot-checking the draft dataset.")


if __name__ == "__main__":
    main()
