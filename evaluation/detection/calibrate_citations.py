"""登録済みの文献どうしの引用の重なりを調べ、閾値の根拠にする。

    venv/Scripts/python.exe evaluation/detection/calibrate_citations.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")
from simcheck.citations import MIN_MANUSCRIPT_REFS, _canonical_refs, _resolve_pmid, coupled_documents
from simcheck.store import Store

N_SAMPLE = 3000
COUPLINGS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_SHARED = (5, 8, 10)
OUT = Path(__file__).parent / "calibrate_citations.json"


def main():
    with Store() as store:
        # 手元のPDFの引用は本文から読み直す必要があるので、PubMedの文献だけで試す
        eligible = [row[0] for row in store.conn.execute(
            "SELECT c.doc_id FROM citation_counts c JOIN docs d ON d.doc_id=c.doc_id "
            "WHERE c.n>=? AND d.source='pubmed'", (MIN_MANUSCRIPT_REFS,)
        )]
        sample = random.Random(20261103).sample(eligible, min(N_SAMPLE, len(eligible)))
        total = store.conn.execute("SELECT COUNT(*), SUM(n) FROM citation_counts").fetchone()
        summary = {"docs_with_citations": total[0], "eligible": len(eligible), "sampled": len(sample), "citations": total[1]}
        refs_of = dict(store.conn.execute(
            f"SELECT doc_id, json_extract(meta, '$.refs') FROM docs WHERE doc_id IN ({','.join(map(str, sample))})"
        ))
        tops = []
        for doc_id in sample:
            keys = _canonical_refs(json.loads(refs_of[doc_id]), lambda pmid: _resolve_pmid(pmid, store))
            for found in coupled_documents(keys, store, {doc_id})[:1]:
                tops.append({"doc": doc_id, "n_keys": len(keys), **found})
        grid = {}
        for min_shared in MIN_SHARED:
            for coupling in COUPLINGS:
                hit = sum(t["coupling"] >= coupling and t["shared"] >= min_shared for t in tops)
                grid[f"shared>={min_shared},coupling>={coupling}"] = round(hit / max(len(sample), 1), 4)
        summary["share_of_docs_flagged"] = grid
        # 上位の組は、同じ論文の重複か本当に近い論文かを読んで確かめる
        tops.sort(key=lambda t: (-t["coupling"], -t["shared"]))
        docs = store.get_docs(sorted({t["doc"] for t in tops[:40]} | {t["doc_id"] for t in tops[:40]}))
        summary["top_pairs"] = [
            {
                "a": (docs[t["doc"]].title or "")[:90], "b": (docs[t["doc_id"]].title or "")[:90],
                "coupling": round(t["coupling"], 3), "shared": t["shared"], "a_refs": t["n_keys"],
                "b_refs": t["source_refs"], "order": round(t["order_share"], 2),
            } for t in tops[:40]
        ]
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_pairs"}, ensure_ascii=False, indent=1))
    for p in summary["top_pairs"]:
        print(p["coupling"], p["shared"], p["a_refs"], p["b_refs"], p["order"], "|", p["a"][:60], "|", p["b"][:60])


if __name__ == "__main__":
    main()
