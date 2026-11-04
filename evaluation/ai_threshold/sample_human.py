"""英語の人間の文章を層別に集めて data/ai_threshold/human.jsonl に保存する。"""
import io
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import pymupdf
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simcheck import DATA_DIR
from simcheck.pubmed import EUTILS_URL, _get_with_retry, parse_article
from simcheck.sections import strip_references
from simcheck.store import Store

OUT_DIR = DATA_DIR.parent / "ai_threshold"
SEED = 20261102
N_PER_STRATUM = 120
N_LONG = 60
LONG_CHARS = 6000
FETCH_POOL = 200
_CJK = re.compile(r"[぀-鿿]")

EYE = '("Eye Diseases"[MeSH Terms] OR "Ophthalmology"[MeSH Terms])'
EPI = '"Epidemiologic Studies"[MeSH Terms]'
BASE = 'english[la] AND hasabstract AND 2010:2018[dp] NOT "Retracted Publication"[pt]'
STRATA = {
    "ophth_japan": f"{EYE} AND Japan[ad] AND {BASE}",
    "ophth_other": f"{EYE} NOT Japan[ad] AND {BASE}",
    "med_japan": f"{EPI} AND Japan[ad] AND {BASE}",
    "med_other": f"{EPI} NOT Japan[ad] AND {BASE}",
}


def fetch_random(client, term: str, rng: random.Random, batches: int = 6) -> list[dict]:
    found = _get_with_retry(client, EUTILS_URL + "esearch.fcgi", {
        "db": "pubmed", "term": term, "usehistory": "y", "retmax": 0, "retmode": "json",
    }).json()["esearchresult"]
    count = min(int(found["count"]), 9999)
    print(f"  検索結果 {found['count']} 件", flush=True)
    records = []
    for _ in range(batches):
        start = rng.randrange(max(1, count - FETCH_POOL))
        response = _get_with_retry(client, EUTILS_URL + "efetch.fcgi", {
            "db": "pubmed", "query_key": found["querykey"], "WebEnv": found["webenv"],
            "retstart": start, "retmax": FETCH_POOL, "retmode": "xml",
        })
        for _, element in etree.iterparse(io.BytesIO(response.content), tag="PubmedArticle", huge_tree=True):
            record = parse_article(element)
            element.clear()
            if record:
                records.append(record)
        time.sleep(0.4)
    return list({r["pmid"]: r for r in records}.values())


def abstract_only(record: dict) -> str:
    # 先頭行は題名。AI文に題名は入らないので比較をそろえるために除く
    return record["text"].split("\n", 1)[1].strip() if "\n" in record["text"] else ""


def pdf_year(path: str) -> int | None:
    try:
        with pymupdf.open(path) as doc:
            m = re.search(r"(19|20)\d{2}", (doc.metadata or {}).get("creationDate") or "")
    except Exception:
        return None
    return int(m.group()) if m else None


def body_excerpt(text: str, rng: random.Random) -> str | None:
    # 冒頭の書誌情報と末尾を避け、本文の中ほどから取る
    body = re.sub(r"\s*\n\s*", " ", strip_references(text).replace("\f", "\n"))
    body = re.sub(r"-\s(?=[a-z])", "", re.sub(r"[ \t]+", " ", body))
    low, high = int(len(body) * 0.15), int(len(body) * 0.85) - LONG_CHARS
    if high <= low:
        return None
    start = rng.randrange(low, high)
    return body[start:start + LONG_CHARS]


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(timeout=120, follow_redirects=True, transport=transport) as client:
        for stratum, term in STRATA.items():
            print(stratum, flush=True)
            usable = [
                (r, a) for r in fetch_random(client, term, rng)
                if (a := abstract_only(r)) and 800 <= len(a) <= 3000 and not _CJK.search(a)
            ]
            rng.shuffle(usable)
            for r, a in usable[:N_PER_STRATUM]:
                rows.append({"id": f"pmid:{r['pmid']}", "stratum": stratum, "lang": "en", "title": r["title"],
                             "year": r["year"], "text": a})
            print(f"  採用 {min(len(usable), N_PER_STRATUM)} 件", flush=True)

    with Store() as store:
        docs = store.get_docs(store.iter_doc_ids("library"))
    long_rows = []
    for doc in sorted(docs.values(), key=lambda d: d.doc_id):
        if len(_CJK.findall(doc.text)) >= 0.01 * len(doc.text):
            continue
        year = pdf_year(doc.meta["path"])
        if year is None or year > 2018:
            continue
        excerpt = body_excerpt(doc.text, rng)
        if excerpt:
            long_rows.append({"id": doc.ext_id[:20], "stratum": "en_long", "lang": "en",
                              "title": doc.title, "year": str(year), "text": excerpt})
    rng.shuffle(long_rows)
    rows += long_rows[:N_LONG]
    print(f"en_long {min(len(long_rows), N_LONG)} 件（候補 {len(long_rows)}）")

    with open(OUT_DIR / "human.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"合計 {len(rows)} 件 → {OUT_DIR / 'human.jsonl'}")


if __name__ == "__main__":
    main()
