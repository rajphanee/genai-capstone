"""
run_evals.py - golden-question evaluation of the whole pipeline.

Unit tests prove the code paths work. Evals prove the SYSTEM gives the right
answers on realistic questions. Run after changing prompts, chunk size,
embeddings, relevance threshold or the documents.

    python evals/run_evals.py                       # uses your .env settings
    EMBEDDING_BACKEND=hashing python evals/run_evals.py   # fully offline, as in CI

It indexes sample_docs/ into a throwaway folder, asks every golden question,
prints PASS/FAIL per case plus accuracy, groundedness and latency, and exits
0 only if every case passes (CI-friendly).
"""

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.service import DocQAService  # noqa: E402


def main() -> int:
    spec = json.loads((Path(__file__).parent / "golden_questions.json").read_text())
    # ignore_cleanup_errors: on Windows ChromaDB may still hold a file open when the temp folder is deleted
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = load_settings(env=dict(os.environ, DATA_DIR=tmp))
        svc = DocQAService.create(settings)
        for path in sorted((ROOT / "sample_docs").glob("*.*")):
            svc.ingest(path.name, path.read_bytes())
        info = svc.info()
        print(f"LLM={info['llm']} ({info['model']}) | embeddings={info['embeddings']} | "
              f"chunks={info['chunks']} | min_relevance={info['min_relevance']}\n")

        failures, latencies, answered, grounded = [], [], 0, 0
        for case in spec["cases"]:
            r = svc.ask(case["q"])
            latencies.append(r["total_ms"])
            ok = r["status"] == case["status"]
            if ok and case["status"] == "answered":
                answered += 1
                files = {c["filename"] for c in r["citations"]}
                ok = case["contains"].lower() in r["answer"].lower() and case["file"] in files
                grounded += bool(r["citations"])
            if any(bad in r["answer"] for bad in spec["must_never_appear"]):
                ok = False
            print(f"  {'PASS' if ok else 'FAIL'}  {case['q'][:62]:62} -> {r['status']}")
            if not ok:
                failures.append((case, r["status"], r["answer"][:200]))

    total = len(spec["cases"])
    print(f"\nAccuracy:     {total - len(failures)}/{total}")
    print(f"Groundedness: {grounded}/{answered} answered questions carry citations")
    print(f"Latency:      median {statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms")
    for case, status, answer in failures:
        print(f"\nFAILED: {case['q']}\n  expected {case['status']} / got {status}\n  answer: {answer}")
    print("\nRESULT:", "ALL PASSED" if not failures else f"{len(failures)} FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
