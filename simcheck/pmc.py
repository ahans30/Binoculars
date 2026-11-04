import hashlib
import time
import unicodedata

import httpx
import numpy as np
from lxml import etree

from .pubmed import EUTILS_URL, MAX_QUERY_RECORDS, _get_with_retry
from .store import DocRow, Store, compress_text
from .textnorm import fingerprints

# 全文は抄録より大きいので、1回の取得件数を抑える
QUERY_BATCH = 50
# 出版日が古い文献や、先の日付になっている文献も拾う
EARLIEST_YEAR = 1000
LATEST_YEAR = 2100
# 再利用条件がはっきりしているCCライセンスの論文だけを照合先にする
LICENSE_FILTER = '"cc license"[filter]'
_ALI_LICENSE_REF = "{http://www.niso.org/schemas/ali/1.0/}license_ref"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_SKIP_TAGS = {"table-wrap", "fig", "supplementary-material", "ref-list", "alternatives"}


def _flatten(node) -> str:
    parts = [node.text or ""]
    for child in node:
        # 文献番号は本文と無関係な数字になり、語の一致を分断するので落とす
        skip = not isinstance(child.tag, str) or child.tag in _SKIP_TAGS or (
            child.tag == "xref" and child.get("ref-type") == "bibr"
        )
        if not skip:
            parts.append(_flatten(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _blocks(node) -> list[str]:
    lines = []
    for child in node:
        if not isinstance(child.tag, str) or child.tag in _SKIP_TAGS:
            continue
        if child.tag in ("sec", "abstract", "body"):
            lines += _blocks(child)
        else:
            text = " ".join(_flatten(child).split())
            if text:
                lines.append(text)
    return lines


def _license_of(meta) -> str | None:
    node = meta.find("permissions/license")
    if node is None:
        return None
    ref = node.find(_ALI_LICENSE_REF)
    if ref is not None and ref.text:
        return ref.text.strip()
    link = node.find(".//ext-link")
    if link is not None and link.get(_XLINK_HREF):
        return link.get(_XLINK_HREF)
    return node.get("license-type") or " ".join(_flatten(node).split())[:200] or None


def parse_article(article) -> dict | None:
    meta = article.find("front/article-meta")
    body = article.find("body")
    if meta is None or body is None:
        return None
    ids = {i.get("pub-id-type"): (i.text or "").strip() for i in meta.findall("article-id")}
    pmcid = ids.get("pmcid") or ids.get("pmc")
    if not pmcid:
        return None
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    title_node = meta.find("title-group/article-title")
    title = " ".join(_flatten(title_node).split()) if title_node is not None else ""
    lines = [title]
    for abstract in meta.findall("abstract"):
        lines += _blocks(abstract)
    lines += _blocks(body)
    return {
        "pmcid": pmcid,
        "pmid": ids.get("pmid"),
        "doi": ids.get("doi"),
        "title": title,
        "text": unicodedata.normalize("NFC", "\n".join(lines)),
        "year": meta.findtext("pub-date/year"),
        "journal": article.findtext("front/journal-meta/journal-title-group/journal-title"),
        "license": _license_of(meta),
        "is_preprint": article.get("article-type") == "preprint",
        "retracted": article.get("article-type") == "retraction",
    }


def _to_row(record: dict) -> tuple[DocRow, np.ndarray]:
    fps = fingerprints(record["text"])
    row = DocRow(
        ext_id=f"pmc:{record['pmcid']}",
        source="pmc",
        title=record["title"],
        year=record["year"],
        url=f"https://pmc.ncbi.nlm.nih.gov/articles/{record['pmcid']}/",
        is_preprint=record["is_preprint"],
        meta={
            "doi": record["doi"], "pmid": record["pmid"], "journal": record["journal"],
            "license": record["license"], "retracted": record["retracted"],
        },
        ztext=compress_text(record["text"]),
    )
    return row, np.fromiter(fps, dtype=np.uint64, count=len(fps))


def _search(client: httpx.Client, query: str) -> dict:
    response = _get_with_retry(client, EUTILS_URL + "esearch.fcgi", {
        "db": "pmc", "term": query, "usehistory": "y", "retmax": 0, "retmode": "json",
    })
    time.sleep(0.4)  # APIキーなしは毎秒3リクエストまで
    return response.json()["esearchresult"]


def _month_filter(first: int, last: int) -> str:
    return f"{first // 12}/{first % 12 + 1:02d}:{last // 12}/{last % 12 + 1:02d}[pdat]"


def _month_windows(client: httpx.Client, query: str, first: int, last: int) -> list[tuple[int, int]]:
    """1回で取り切れる大きさになるまで、出版月の範囲を二分する。月は 年*12+月-1 で表す。"""
    count = int(_search(client, f"{query} AND {_month_filter(first, last)}")["count"])
    if count <= MAX_QUERY_RECORDS or first == last:
        return [(first, last)] if count else []
    middle = (first + last) // 2
    return _month_windows(client, query, first, middle) + _month_windows(client, query, middle + 1, last)


def _add_results(store: Store, client: httpx.Client, query: str, name: str, progress) -> int:
    result = _search(client, query)
    count = int(result["count"])
    limit = min(count, MAX_QUERY_RECORDS)
    if count > limit:
        progress(f"E-utilities の上限のため {count:,} 件中 先頭 {limit:,} 件のみ取得します: {query}")

    rows, hash_parts, id_parts = [], [], []
    for start in range(0, limit, QUERY_BATCH):
        response = _get_with_retry(client, EUTILS_URL + "efetch.fcgi", {
            "db": "pmc", "query_key": result["querykey"], "WebEnv": result["webenv"],
            "retstart": start, "retmax": min(QUERY_BATCH, limit - start), "retmode": "xml",
        })
        root = etree.fromstring(response.content, etree.XMLParser(huge_tree=True, resolve_entities=False))
        for article in root.iter("article"):
            record = parse_article(article)
            if record:
                row, fps = _to_row(record)
                if fps.size:
                    hash_parts.append(fps)
                    id_parts.append(np.full(fps.size, len(rows), dtype=np.uint32))
                    rows.append(row)
        progress(f"  {min(start + QUERY_BATCH, limit):,}/{limit:,} 件")
        time.sleep(0.4)
    if not rows:
        return store.add_batch(name, "pmc", [], np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32))
    return store.add_batch(name, "pmc", rows, np.concatenate(hash_parts), np.concatenate(id_parts))


def add_pmc_query(store: Store, term: str, progress=print) -> dict:
    """検索式に合うCCライセンスの論文の全文を、NCBIが自動取得を認めている E-utilities で取り込む。

    E-utilities は1検索につき先頭9,999件までしか返さないので、超える分は出版月で分けて取得する。
    分けた範囲ごとに登録するので、途中で止まっても再実行で続きから進む。
    """
    query = f"({term}) AND {LICENSE_FILTER}"
    name = "pmc_query_" + hashlib.sha256(term.encode("utf-8")).hexdigest()[:16]
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(timeout=300, follow_redirects=True, transport=transport) as client:
        count = int(_search(client, query)["count"])
        progress(f"検索結果 {count:,} 件")
        if count <= MAX_QUERY_RECORDS:
            return {"added": _add_results(store, client, query, name, progress), "found": count}

        windows = _month_windows(client, query, EARLIEST_YEAR * 12, LATEST_YEAR * 12 + 11)
        progress(f"出版月で {len(windows)} 範囲に分けて取得します")
        added = 0
        for first, last in windows:
            part = f"{name}_{first}_{last}"
            if not store.is_ingested(part):
                added += _add_results(store, client, f"{query} AND {_month_filter(first, last)}", part, progress)
    store.add_batch(name, "pmc", [], np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32))
    return {"added": added, "found": count, "windows": len(windows)}
