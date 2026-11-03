"""登録済みの文献に語句リストを当てて、語句ごとの出現文書数を数える。

    venv/Scripts/python.exe evaluation/integrity/scan_corpus.py
"""
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simcheck.integrity import TORTURED_PATH, find_tortured, parse_tortured
from simcheck.store import Store

BATCH = 2000
OUT = Path(__file__).parent

_fingerprints = None


def _scan(texts: list[str]) -> list[list[str]]:
    global _fingerprints
    if _fingerprints is None:
        _fingerprints = parse_tortured(TORTURED_PATH)
    return [[hit.phrase for hit in find_tortured(text, _fingerprints)] for text in texts]


def _batches(store, ids):
    for i in range(0, len(ids), BATCH):
        docs = store.get_docs(ids[i:i + BATCH])
        yield [doc.text for doc in docs.values()]


def main():
    fingerprints = parse_tortured(TORTURED_PATH)
    print(f"語句 {fingerprints.n_loaded:,} / 解釈できない式 {fingerprints.n_skipped:,}")
    summary = {}
    with Store() as store, ProcessPoolExecutor() as pool:
        for source in ("library", "pubmed"):
            ids = store.iter_doc_ids(source)
            df, per_doc = Counter(), Counter()
            for results in pool.map(_scan, _batches(store, ids)):
                for phrases in results:
                    df.update(phrases)
                    per_doc[min(len(phrases), 5)] += 1
            with open(OUT / f"df_{source}.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["phrase", "documents"])
                writer.writerows(df.most_common())
            summary[source] = {
                "documents": len(ids),
                "phrases_seen": len(df),
                "documents_by_hits_capped_at_5": dict(sorted(per_doc.items())),
                "top": df.most_common(40),
            }
    (OUT / "scan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
