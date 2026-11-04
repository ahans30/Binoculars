"""言い換え検索の類似度を集め、閾値ごとの検出率と誤検出を求める。dev の結果だけを使って閾値を決める。

    venv/Scripts/python.exe evaluation/detection/semantic_scores.py --split dev
    venv/Scripts/python.exe evaluation/detection/semantic_scores.py --split dev --scdg
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import prepare_cases
from paraphrase import load_paraphrases
from simcheck.rerank import GainScorer
from simcheck.semantic import SemanticIndex, encode, sentence_spans
from simcheck.store import Store
from simcheck.textnorm import tokenize

TOP_K = 20
SAME_THRESHOLDS = (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90)
CROSS_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SCDG_MIN_SCORE = 0.70
SCDG_CONTEXT_CHARS = 400
SCDG_PAIRS = 400


def sentences_of(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def best_hits(index, chunks, vectors, ext_of, exclude=frozenset()):
    """各文について [(類似度, 出典のID, 文の位置)] を類似度順に返す。"""
    scores, ids = index.search(vectors, TOP_K)
    rows = []
    for row_scores, row_ids in zip(scores, ids):
        hits = []
        for score, idx in zip(row_scores, row_ids):
            if idx < 0:
                break
            doc_id, s, e = (int(x) for x in chunks[idx])
            ext = ext_of.get(doc_id)
            if ext is None or ext in exclude:
                continue
            hits.append((float(score), ext, doc_id, s, e))
        rows.append(hits)
    return rows


def rate(passages: list[list[float]], threshold: float) -> float:
    return sum(any(s >= threshold for s in p) for p in passages) / max(len(passages), 1)


def false_hits_per_1000(best_scores: list[float], words: int, threshold: float) -> float:
    return 1000 * sum(s >= threshold for s in best_scores) / max(words, 1)


class Scdg:
    """平均のほかに、上位30%の語の平均も比べる。"""

    def __init__(self):
        self.scorer = GainScorer()

    def score(self, source: str, target: str) -> dict:
        gains = self.scorer.gains(source, target)
        top = np.sort(gains)[::-1][:max(1, int(np.ceil(len(gains) * 0.3)))]
        return {"mean": float(gains.mean()), "top30": float(top.mean())}


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--scdg", action="store_true", help="再順位付けの効果も測る")
    args = parser.parse_args()

    with Store() as store:
        data = prepare_cases(store, args.split)
        paraphrases = load_paraphrases(args.split)
        rows = store.conn.execute("SELECT doc_id, ext_id FROM docs").fetchall()
        ext_of = dict(rows)
        index = SemanticIndex()
        results = {"split": args.split, "parts": len(index.parts), "same_language": {}, "cross_language": {}}

        filler_sentences = [s for text in data["fillers"] for s in sentences_of(text)]
        filler_words = sum(len(tokenize(s).tokens) for s in filler_sentences)
        filler_vectors = encode(filler_sentences)
        human_rows = [(row["ext_id"], s) for row in data["human"] for s in sentences_of(row["text"])]
        human_words = sum(len(tokenize(s).tokens) for _, s in human_rows)
        human_vectors = encode([s for _, s in human_rows])

        def control_best(part_index, part_chunks):
            filler = [hits[0][0] if hits else 0.0 for hits in best_hits(part_index, part_chunks, filler_vectors, ext_of)]
            human = []
            for hits, (ext, _) in zip(best_hits(part_index, part_chunks, human_vectors, ext_of), human_rows):
                others = [h for h in hits if h[1] != ext]
                human.append(others[0][0] if others else 0.0)
            return filler, human

        part_names = ["library", "pubmed"][:len(index.parts)]
        controls = {}
        for name, (part_index, part_chunks) in zip(part_names, index.parts):
            controls[name] = control_best(part_index, part_chunks)

        scdg_pairs = {"positive": [], "negative": []}
        for mode in ("light", "heavy"):
            for stratum, part_name in (("library", "library"), ("pubmed", "pubmed")):
                if part_name not in part_names:
                    continue
                part_index, part_chunks = index.parts[part_names.index(part_name)]
                passages = []
                for i, passage in enumerate(data["passages"][stratum]):
                    text = paraphrases.get((f"{stratum}:{i}", mode))
                    if not text:
                        continue
                    sentences = sentences_of(text)
                    if not sentences:
                        continue
                    hits = best_hits(part_index, part_chunks, encode(sentences), ext_of)
                    passages.append([
                        max((h[0] for h in row if h[1] == passage["ext_id"]), default=0.0) for row in hits
                    ])
                    if args.scdg and mode == "heavy":
                        for sentence, row in zip(sentences, hits):
                            for h in row:
                                if h[0] >= SCDG_MIN_SCORE and h[1] == passage["ext_id"]:
                                    scdg_pairs["positive"].append((sentence, h))
                                    break
                filler_best, human_best = controls[part_name]
                results["same_language"][f"{mode}/{stratum}"] = {
                    "n": len(passages),
                    "by_threshold": {
                        str(t): {
                            "detected": rate(passages, t),
                            "filler_false_per_1000": false_hits_per_1000(filler_best, filler_words, t),
                            "human_false_per_1000": false_hits_per_1000(human_best, human_words, t),
                        } for t in SAME_THRESHOLDS
                    },
                }

        if index.ja_index is not None:
            translated = []
            for i, passage in enumerate(data["passages"]["library_ja"]):
                text = paraphrases.get((f"library_ja:{i}", "translate_en"))
                if not text or not sentences_of(text):
                    continue
                hits = best_hits(index.ja_index, index.ja_chunks, encode(sentences_of(text)), ext_of)
                translated.append([max((h[0] for h in row if h[1] == passage["ext_id"]), default=0.0) for row in hits])
            filler_ja, human_ja = control_best(index.ja_index, index.ja_chunks)
            results["cross_language"] = {
                "n": len(translated),
                "by_threshold": {
                    str(t): {
                        "detected": rate(translated, t),
                        "filler_false_per_1000": false_hits_per_1000(filler_ja, filler_words, t),
                        "human_false_per_1000": false_hits_per_1000(human_ja, human_words, t),
                    } for t in CROSS_THRESHOLDS
                },
            }

        if args.scdg:
            docs_needed = {h[2] for _, h in scdg_pairs["positive"]}
            for name, (part_index, part_chunks) in zip(part_names, index.parts):
                rows_ = best_hits(part_index, part_chunks, filler_vectors, ext_of)
                for sentence, row in zip(filler_sentences, rows_):
                    if row and row[0][0] >= SCDG_MIN_SCORE:
                        scdg_pairs["negative"].append((sentence, row[0]))
                        docs_needed.add(row[0][2])
            rng = np.random.default_rng(20261103)
            for key in scdg_pairs:
                pairs = scdg_pairs[key]
                order = rng.permutation(len(pairs))[:SCDG_PAIRS]
                scdg_pairs[key] = [pairs[i] for i in order]
            docs = store.get_docs(sorted({h[2] for pairs in scdg_pairs.values() for _, h in pairs}))
            model = Scdg()

            def measure(pairs):
                measured = []
                for sentence, (score, _, doc_id, s, e) in pairs:
                    doc = docs.get(doc_id)
                    if doc is None:
                        continue
                    context = doc.text[max(0, s - SCDG_CONTEXT_CHARS):e + SCDG_CONTEXT_CHARS]
                    measured.append({"cosine": score, **model.score(context, sentence)})
                return measured

            positive, negative = measure(scdg_pairs["positive"]), measure(scdg_pairs["negative"])
            results["scdg"] = {
                "n_positive": len(positive), "n_negative": len(negative),
                "auc": {
                    key: auc([p[key] for p in positive], [n[key] for n in negative])
                    for key in ("cosine", "mean", "top30")
                },
                "quantiles": {
                    key: {
                        "positive": [float(np.quantile([p[key] for p in positive], q)) for q in (0.1, 0.5, 0.9)],
                        "negative": [float(np.quantile([n[key] for n in negative], q)) for q in (0.1, 0.5, 0.9)],
                    } for key in ("cosine", "mean", "top30")
                },
            }

    out = HERE / f"semantic_scores_{args.split}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
