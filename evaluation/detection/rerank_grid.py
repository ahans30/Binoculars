"""言い換えの候補の再評価について、類似度と利得の閾値の組み合わせを比べる。dev だけで決める。

    venv/Scripts/python.exe evaluation/detection/rerank_grid.py --split dev
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import prepare_cases
from paraphrase import load_paraphrases
from simcheck.rerank import GainScorer, source_context
from simcheck.semantic import SemanticIndex, sentence_spans
from simcheck.store import Store
from simcheck.textnorm import tokenize

MIN_COSINE = 0.75
COSINES = (0.75, 0.78, 0.80, 0.82, 0.85)
GAINS = (None, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
CELLS = (("light", "pubmed"), ("light", "library"), ("heavy", "pubmed"), ("heavy", "library"))


def sentences_of(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def hits_for(sentences, index, ids_of, exclude=frozenset()):
    return [hit for hit in index.search(sentences, MIN_COSINE, set(exclude), cross_threshold=None)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    scorer = GainScorer()
    with Store() as store:
        data = prepare_cases(store, args.split)
        paraphrases = load_paraphrases(args.split)
        doc_of = dict(store.conn.execute("SELECT ext_id, doc_id FROM docs").fetchall())
        index = SemanticIndex()
        cache: dict[tuple, float] = {}
        docs: dict[int, object] = {}
        started = time.time()

        def gain(sentence, hit) -> float:
            doc_id, s, e, _ = hit
            key = (sentence, doc_id, s)
            if key not in cache:
                if doc_id not in docs:
                    docs.update(store.get_docs([doc_id]))
                doc = docs.get(doc_id)
                cache[key] = scorer.mean_gain(source_context(doc.text, s, e), sentence) if doc else -99.0
            return cache[key]

        positives: dict[str, list[list[tuple[float, float, bool]]]] = {}
        for mode, stratum in CELLS:
            passages = []
            for i, passage in enumerate(data["passages"][stratum]):
                text = paraphrases.get((f"{stratum}:{i}", mode))
                sentences = sentences_of(text) if text else []
                if not sentences:
                    continue
                target = doc_of.get(passage["ext_id"])
                rows = []
                for sentence, hit in zip(sentences, hits_for(sentences, index, doc_of)):
                    if hit is not None:
                        rows.append((hit[3], gain(sentence, hit), hit[0] == target))
                passages.append(rows)
            positives[f"{mode}/{stratum}"] = passages
            print(f"{mode}/{stratum}: {len(passages)} 文章, {len(cache)} 組, {time.time() - started:.0f}秒", flush=True)

        def control(rows_of_text):
            scores, words = [], 0
            for text, exclude in rows_of_text:
                sentences = sentences_of(text)
                words += sum(len(tokenize(s).tokens) for s in sentences)
                for sentence, hit in zip(sentences, hits_for(sentences, index, doc_of, exclude)):
                    if hit is not None:
                        scores.append((hit[3], gain(sentence, hit)))
            return scores, words

        filler_scores, filler_words = control([(t, ()) for t in data["fillers"]])
        print(f"地の文: {len(filler_scores)} 組, {time.time() - started:.0f}秒", flush=True)
        human_scores, human_words = control([
            (row["text"], (doc_of[row["ext_id"]],)) for row in data["human"] if row["ext_id"] in doc_of
        ])
        print(f"人間の抄録: {len(human_scores)} 組, {time.time() - started:.0f}秒", flush=True)

    grid = []
    for cosine in COSINES:
        for min_gain in GAINS:
            def kept(c, g):
                return c >= cosine and (min_gain is None or g >= min_gain)

            row = {"cosine": cosine, "gain": min_gain}
            for cell, passages in positives.items():
                row[cell] = sum(any(kept(c, g) and right for c, g, right in p) for p in passages) / max(len(passages), 1)
            row["filler_per_1000"] = 1000 * sum(kept(c, g) for c, g in filler_scores) / filler_words
            row["human_per_1000"] = 1000 * sum(kept(c, g) for c, g in human_scores) / human_words
            grid.append(row)
    out = HERE / f"rerank_grid_{args.split}.json"
    out.write_text(json.dumps({
        "pairs": len(cache), "seconds": time.time() - started, "grid": grid,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in grid:
        print(" ".join(f"{k}={v if not isinstance(v, float) else round(v, 3)}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
