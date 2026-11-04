import gzip
import hashlib
import io
import os
import re
import shutil
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
from lxml import etree

from . import DATA_DIR
from .store import DocRow, Store, compress_text
from .textnorm import fingerprints

BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
QUERY_BATCH = 200
MAX_QUERY_RECORDS = 9999
FETCH_ATTEMPTS = 5
_FILE_RE = re.compile(r'href="(pubmed\d+n\d+\.xml\.gz)"')
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_MD5_RE = re.compile(r"[0-9a-f]{32}")
# NCBI の負荷を考えて同時接続は控えめにする
DEFAULT_WORKERS = 4
# 取り込みでディスクを使い切らないよう、空きがこれを下回ったら止める（続きは再実行で再開できる）
MIN_FREE_GB = 40


def list_baseline_files() -> list[str]:
    response = httpx.get(BASELINE_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    return sorted(set(_FILE_RE.findall(response.text)))


def _download(name: str, dest: Path) -> Path:
    path = dest / name
    expected = _MD5_RE.search(
        httpx.get(BASELINE_URL + name + ".md5", timeout=60, follow_redirects=True).text
    ).group()
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.md5()
    with httpx.stream("GET", BASELINE_URL + name, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"MD5不一致: {name}")
    os.replace(tmp, path)
    return path


def _text_of(element) -> str:
    # AbstractText は <i> や <sup> を含むので itertext で全文を取る
    return "".join(element.itertext()).strip()


def reference_ids(article) -> list[dict]:
    """PubMedが持つ引用文献のIDを、並び順に返す。"""
    refs = []
    for reference in article.iterfind("PubmedData/ReferenceList/Reference"):
        ids = {aid.get("IdType"): (aid.text or "").strip() for aid in reference.iterfind("ArticleIdList/ArticleId")}
        ref = {key: ids[source] for key, source in (("pmid", "pubmed"), ("doi", "doi")) if ids.get(source)}
        if ref:
            refs.append(ref)
    return refs


def parse_article(article) -> dict | None:
    citation = article.find("MedlineCitation")
    art = citation.find("Article") if citation is not None else None
    if art is None:
        return None
    pmid = citation.findtext("PMID")

    parts = []
    for node in art.findall("Abstract/AbstractText"):
        text = _text_of(node)
        if text:
            label = node.get("Label")
            parts.append(f"{label}: {text}" if label else text)
    # 日本語抄録は OtherAbstract に入っていることがある
    for other in citation.findall("OtherAbstract"):
        text = " ".join(_text_of(n) for n in other.findall("AbstractText"))
        if text.strip():
            parts.append(text.strip())
    if not pmid or not parts:
        return None

    title_node = art.find("ArticleTitle")
    title = _text_of(title_node) if title_node is not None else ""
    vernacular = art.find("VernacularTitle")
    if vernacular is not None:
        parts.insert(0, _text_of(vernacular))

    pub_date = art.find("Journal/JournalIssue/PubDate")
    year = None
    if pub_date is not None:
        year = pub_date.findtext("Year")
        if not year:
            m = _YEAR_RE.search(pub_date.findtext("MedlineDate") or "")
            year = m.group() if m else None

    doi = None
    for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text
    pub_types = {t.text for t in art.findall("PublicationTypeList/PublicationType")}

    return {
        "pmid": pmid,
        "title": title,
        "text": unicodedata.normalize("NFC", "\n".join([title, *parts])),
        "year": year,
        "doi": doi,
        "journal": art.findtext("Journal/ISOAbbreviation") or art.findtext("Journal/Title"),
        "is_preprint": "Preprint" in pub_types,
        "retracted": "Retracted Publication" in pub_types,
        "refs": reference_ids(article),
    }


def iter_articles(path: Path):
    with gzip.open(path, "rb") as f:
        for _, element in etree.iterparse(f, tag="PubmedArticle", huge_tree=True):
            record = parse_article(element)
            if record:
                yield record
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]


def _to_row(record: dict) -> tuple[DocRow, np.ndarray]:
    fps = fingerprints(record["text"])
    row = DocRow(
        ext_id=f"pmid:{record['pmid']}",
        source="pubmed",
        title=record["title"],
        year=record["year"],
        url=f"https://pubmed.ncbi.nlm.nih.gov/{record['pmid']}/",
        is_preprint=record["is_preprint"],
        meta={
            "doi": record["doi"], "journal": record["journal"], "retracted": record["retracted"],
            **({"refs": record["refs"]} if record["refs"] else {}),
        },
        ztext=compress_text(record["text"]),
    )
    return row, np.fromiter(fps, dtype=np.uint64, count=len(fps))


def _process_file(name: str, dest: str, keep_gz: bool):
    dest_path = Path(dest)
    path = _download(name, dest_path)
    rows, hash_parts, id_parts = [], [], []
    for record in iter_articles(path):
        row, fps = _to_row(record)
        if not fps.size:
            continue
        hash_parts.append(fps)
        id_parts.append(np.full(fps.size, len(rows), dtype=np.uint32))
        rows.append(row)
    if not keep_gz:
        path.unlink(missing_ok=True)
    # 抄録のある論文を1件も含まないファイルもある
    if not rows:
        return name, rows, np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32)
    return name, rows, np.concatenate(hash_parts), np.concatenate(id_parts)


def _get_with_retry(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    # NCBI は大きな応答の途中で接続を切ることがあるので、バッチ単位でやり直す
    for attempt in range(FETCH_ATTEMPTS):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == FETCH_ATTEMPTS - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def add_pubmed_query(store: Store, term: str, progress=print) -> dict:
    """検索式に合う抄録だけを E-utilities で取り込む。ベースライン全件を入れる前の絞り込み用。"""
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(timeout=120, follow_redirects=True, transport=transport) as client:
        found = client.get(EUTILS_URL + "esearch.fcgi", params={
            "db": "pubmed", "term": term, "usehistory": "y", "retmax": 0, "retmode": "json",
        })
        found.raise_for_status()
        result = found.json()["esearchresult"]
        count = int(result["count"])
        progress(f"検索結果 {count:,} 件")
        limit = min(count, MAX_QUERY_RECORDS)
        if count > limit:
            progress(f"E-utilities の上限のため先頭 {limit:,} 件のみ取得します")

        rows, hash_parts, id_parts = [], [], []
        for start in range(0, limit, QUERY_BATCH):
            response = _get_with_retry(client, EUTILS_URL + "efetch.fcgi", {
                "db": "pubmed", "query_key": result["querykey"], "WebEnv": result["webenv"],
                # retstart+retmax が 9,999 を超えると 400 が返る
                "retstart": start, "retmax": min(QUERY_BATCH, limit - start), "retmode": "xml",
            })
            for _, element in etree.iterparse(io.BytesIO(response.content), tag="PubmedArticle", huge_tree=True):
                record = parse_article(element)
                element.clear()
                if record:
                    row, fps = _to_row(record)
                    if fps.size:
                        hash_parts.append(fps)
                        id_parts.append(np.full(fps.size, len(rows), dtype=np.uint32))
                        rows.append(row)
            progress(f"  {min(start + QUERY_BATCH, limit):,}/{limit:,} 件")
            time.sleep(0.4)  # APIキーなしは毎秒3リクエストまで
    added = 0
    if rows:
        name = "query_" + hashlib.sha256(term.encode("utf-8")).hexdigest()[:16]
        added = store.add_batch(name, "pubmed", rows, np.concatenate(hash_parts), np.concatenate(id_parts))
    return {"added": added, "found": count}


def add_pubmed(store: Store, first: int | None = None, last: int | None = None,
               workers: int = DEFAULT_WORKERS, keep_gz: bool = False, progress=print) -> dict:
    """ベースラインを取り込む。first/last は 1 始まりのファイル番号で、未指定なら全件。"""
    names = list_baseline_files()
    selected = names[(first - 1) if first else 0:last if last else None]
    pending = [n for n in selected if not store.is_ingested(n.removesuffix(".xml.gz"))]
    progress(f"ベースライン {len(names)} ファイル中 {len(selected)} を選択、未取り込み {len(pending)}")

    dest = DATA_DIR / "pubmed_gz"
    dest.mkdir(parents=True, exist_ok=True)
    total_docs, failed = 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_file, n, str(dest), keep_gz): n for n in pending}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                _, rows, hashes, ids = future.result()
            except Exception as e:
                failed.append((name, str(e)))
                progress(f"  失敗: {name}: {e}")
                continue
            added = store.add_batch(name.removesuffix(".xml.gz"), "pubmed", rows, hashes, ids)
            total_docs += added
            progress(f"  [{done}/{len(pending)}] {name}: {added} 件")
            if shutil.disk_usage(DATA_DIR).free < MIN_FREE_GB * 1024**3:
                pool.shutdown(wait=False, cancel_futures=True)
                failed.append(("(中断)", f"空き容量が {MIN_FREE_GB} GB を下回りました"))
                progress(f"  空き容量が {MIN_FREE_GB} GB を下回ったので中断します")
                break
    return {"added": total_docs, "failed": failed}
