"""手元のPDFで、表示と抽出文字の一致率の分布を調べ、閾値の根拠にする。

    venv/Scripts/python.exe evaluation/detection/calibrate_pdf.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simcheck.pdfcheck import page_agreements
from simcheck.store import Store

N_PDFS = 120
OUT = Path(__file__).parent / "calibrate_pdf.json"


def main():
    with Store() as store:
        docs = store.get_docs(store.iter_doc_ids("library"))
    paths = sorted(d.meta["path"] for d in docs.values() if d.meta.get("path") and Path(d.meta["path"]).suffix.lower() == ".pdf")
    sample = random.Random(20261103).sample(paths, min(N_PDFS, len(paths)))
    shares, low = [], []
    for i, path in enumerate(sample, 1):
        try:
            for page, share in page_agreements(path):
                shares.append(share)
                if share < 0.7:
                    low.append((Path(path).name[:60], page, round(share, 3)))
        except Exception as e:
            print(f"失敗: {Path(path).name[:60]}: {e}", flush=True)
        if i % 20 == 0:
            print(f"{i}/{len(sample)}", flush=True)
    shares.sort()
    n = len(shares)
    summary = {
        "pdfs": len(sample), "pages": n,
        "min": shares[0], "p01": shares[n // 100], "p05": shares[n // 20], "median": shares[n // 2],
        "below_0.5": sum(s < 0.5 for s in shares), "below_0.7": sum(s < 0.7 for s in shares), "low_pages": low,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
