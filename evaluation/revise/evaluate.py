"""公開済みの抄録の文で修正案を作り、検査の結果を集計する。未発表の原稿は使わない。

使い方: evaluate.py [モデル名]（LM Studio のサーバーを起動しておく）
抄録は data/ai_threshold/human.jsonl の PubMed の層だけを使う。文を出典の一致として扱う。
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import httpx
import numpy as np

from simcheck import DATA_DIR
from simcheck.revise import (
    DEFAULT_URL, _PLACEHOLDER, Target, available_model, check_candidate, is_common_wording, request_suggestion,
    sentence_ranges, similarities,
)
from simcheck.semantic import encode
from simcheck.store import Doc, Store

# 20261103 は問題点を洗い出すのに使った。検証には別の文を使う
SEED = int(os.environ.get("REVISE_EVAL_SEED", "20261104"))
N_SENTENCES = 60
STRATA = {"ophth_japan", "ophth_other", "med_japan", "med_other"}
OUT = Path(__file__).resolve().parent / "results.json"
# 公開抄録の文だが、照合先のデータと同じくコミットしない
DETAILS = DATA_DIR.parent / "revise_eval" / "details.jsonl"
_HAS_NUMBER = re.compile(r"\d")


def sample() -> list[tuple[dict, str, str, str]]:
    rows = [json.loads(line) for line in open(DATA_DIR.parent / "ai_threshold" / "human.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["stratum"] in STRATA]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    picked = []
    for row in rows:
        text = row["text"]
        ranges = sentence_ranges(text)
        options = [i for i, (s, e) in enumerate(ranges) if 15 <= len(text[s:e].split()) <= 50 and _HAS_NUMBER.search(text[s:e])]
        if not options:
            continue
        i = rng.choice(options)
        before = text[slice(*ranges[i - 1])] if i else ""
        after = text[slice(*ranges[i + 1])] if i + 1 < len(ranges) else ""
        picked.append((row, text[slice(*ranges[i])], before, after))
        if len(picked) == N_SENTENCES:
            break
    return picked


def main():
    url = DEFAULT_URL
    records = []
    store = Store()
    with httpx.Client(timeout=600) as client:
        model = sys.argv[1] if len(sys.argv) > 1 else available_model(url, client)
        print("model:", model, flush=True)
        for n, (row, sentence, before, after) in enumerate(sample(), 1):
            doc = Doc(0, row["id"], "pubmed", row.get("title") or "", None, None, False, {}, row["text"])
            target = Target(0, len(sentence), 0, sentence, before, after, doc, sentence, 0.0)
            if is_common_wording(sentence, store):
                records.append({"id": row["id"], "sentence": sentence, "action": "keep", "n": 0, "candidates": []})
                print(n, "keep (common)", flush=True)
                continue
            started = time.time()
            try:
                parsed = request_suggestion(client, url, model, target)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
                records.append({"id": row["id"], "error": type(e).__name__})
                print(n, "error", type(e).__name__, flush=True)
                continue
            seconds = time.time() - started
            texts = [" ".join(_PLACEHOLDER.sub("", str(c)).split()) for c in parsed.get("candidates") or []]
            texts = [t for t in dict.fromkeys(texts) if t and t != sentence][:3]
            scores = similarities(sentence, texts)
            checks = [check_candidate(sentence, t, None, doc, s) for t, s in zip(texts, scores)]
            records.append({
                "id": row["id"], "sentence": sentence, "reason": parsed.get("reason", ""),
                "action": parsed["action"], "seconds": seconds, "n": len(texts),
                "candidates": [{
                    "text": t, "problems": c.problems(), "passed": c.passed, "numbers_ok": not (c.missing_numbers or c.added_numbers),
                    "terms_ok": not c.missing_terms, "negation_ok": not c.negation_changed,
                    "direction_ok": not c.direction_changed, "significance_ok": not c.significance_changed,
                    "language_ok": not c.language_changed, "similarity": c.similarity,
                    "longest_source_run": c.longest_source_run,
                } for t, c in zip(texts, checks)],
            })
            print(n, parsed["action"], len(texts), f"{seconds:.0f}s", flush=True)

    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    with open(DETAILS, "w", encoding="utf-8") as f:
        for record in records:
            print(json.dumps(record, ensure_ascii=False), file=f)

    # 意味の近さの閾値の参考として、無関係な文どうしの値も見る
    sentences = [s for _, s, _, _ in sample()]
    vectors = encode(sentences)
    matrix = vectors @ vectors.T
    unrelated = matrix[np.triu_indices(len(sentences), 1)]
    candidates = [c for r in records for c in r.get("candidates", [])]
    similarity = np.array([c["similarity"] for c in candidates if c["similarity"] is not None])
    summary = {
        "model": model,
        "n_sentences": len(records),
        "n_errors": sum("error" in r for r in records),
        "actions": {a: sum(r.get("action") == a for r in records) for a in ("keep", "cite", "quote", "rewrite")},
        "n_candidates": len(candidates),
        "sentences_with_candidates": sum(1 for r in records if r.get("n")),
        "sentences_with_passing_candidate": sum(1 for r in records if any(c["passed"] for c in r.get("candidates", []))),
        "numbers_ok": sum(c["numbers_ok"] for c in candidates),
        "terms_ok": sum(c["terms_ok"] for c in candidates),
        "negation_ok": sum(c["negation_ok"] for c in candidates),
        "direction_ok": sum(c["direction_ok"] for c in candidates),
        "significance_ok": sum(c["significance_ok"] for c in candidates),
        "language_ok": sum(c["language_ok"] for c in candidates),
        "seed": SEED,
        "passed": sum(c["passed"] for c in candidates),
        "similarity_percentiles": {str(p): float(np.percentile(similarity, p)) for p in (1, 5, 25, 50, 75, 95)} if similarity.size else {},
        "unrelated_percentiles": {str(p): float(np.percentile(unrelated, p)) for p in (50, 95, 99, 100)},
        "longest_source_run_median": float(np.median([c["longest_source_run"] for c in candidates])) if candidates else None,
        "seconds_median": float(np.median([r["seconds"] for r in records if "seconds" in r])) if records else None,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
