import json
import re
import time
from hashlib import blake2b
from pathlib import Path

import numpy as np

from .extract import extract_document
from .flags import Flag
from .integrity import _DOI_RE, _dois_in, _title_match_length
from .sections import spans_of
from .store import Store
from .textnorm import tokenize

TITLE_KEY_TOKENS = 6
MIN_MANUSCRIPT_REFS = 5
MIN_SHARED = 8
# 同じ分野の論文どうしの重なりより上に置いた値（evaluation/detection/results.md）
MIN_COUPLING = 0.6
MAX_EXAMPLES = 5
_PMID_RE = re.compile(r"\b(?:PMID|PubMed(?:\s+ID)?)\s*:?\s*(\d{4,9})\b", re.IGNORECASE)
_SQL_BATCH = 500
_WRITE_BATCH = 2_000_000
_READ_BATCH = 5_000_000


def _hash(text: str) -> int:
    digest = blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def _title_hash(tokens) -> int:
    return _hash("\x1f".join(tokens))


def canonical_key(doc_ext_id: str, doi: str | None) -> str:
    return f"doi:{doi.lower()}" if doi else doc_ext_id


def build_title_index(store: Store, progress=print) -> int:
    """題名から文献を引けるようにする。文献を増やしたら作り直す。"""
    with store._lock, store.conn:
        # 全件でも現実的な時間で終わるよう、索引は最後に作る
        store.conn.execute("DROP INDEX IF EXISTS title_keys_hash")
        store.conn.execute("DELETE FROM title_keys")
        rows, total = [], 0
        for doc_id, title in store.conn.cursor().execute("SELECT doc_id, title FROM docs WHERE title IS NOT NULL"):
            words = tokenize(title).tokens
            if len(words) >= TITLE_KEY_TOKENS:
                rows.append((_title_hash(words[:TITLE_KEY_TOKENS]), doc_id))
            if len(rows) >= 100_000:
                store.conn.executemany("INSERT INTO title_keys VALUES (?,?)", rows)
                total += len(rows)
                rows = []
                progress(f"  題名 {total:,} 件")
        store.conn.executemany("INSERT INTO title_keys VALUES (?,?)", rows)
        total += len(rows)
        store.conn.execute("CREATE INDEX title_keys_hash ON title_keys (hash)")
    return total


def _resolve_titles(text: str, store: Store) -> list[tuple[int, str]]:
    """参考文献の中の、照合先の文献の題名を探して (文字位置, キー) を返す。"""
    t = tokenize(text)
    tokens = t.tokens
    by_hash: dict[int, list[int]] = {}
    for i in range(len(tokens) - TITLE_KEY_TOKENS + 1):
        by_hash.setdefault(_title_hash(tokens[i:i + TITLE_KEY_TOKENS]), []).append(i)
    if not by_hash:
        return []
    candidates: dict[int, list[int]] = {}
    hashes = list(by_hash)
    with store._lock:
        for i in range(0, len(hashes), _SQL_BATCH):
            chunk = hashes[i:i + _SQL_BATCH]
            marks = ",".join("?" * len(chunk))
            for h, doc_id in store.conn.execute(f"SELECT hash, doc_id FROM title_keys WHERE hash IN ({marks})", chunk):
                candidates.setdefault(doc_id, []).extend(by_hash[h])
    found = []
    docs = _doc_keys(store, list(candidates))
    for doc_id, positions in candidates.items():
        if doc_id not in docs:
            continue
        title, key = docs[doc_id]
        words = tuple(tokenize(title).tokens)
        for i in positions:
            if _title_match_length(tokens, i, words):
                found.append((t.starts[i], key))
                break
    return found


def _doc_keys(store: Store, doc_ids: list[int]) -> dict[int, tuple[str, str]]:
    """本文を読まずに (題名, キー) を取り出す。"""
    result = {}
    with store._lock:
        for i in range(0, len(doc_ids), _SQL_BATCH):
            chunk = doc_ids[i:i + _SQL_BATCH]
            marks = ",".join("?" * len(chunk))
            for doc_id, title, ext_id, doi in store.conn.execute(
                f"SELECT doc_id, title, ext_id, json_extract(meta, '$.doi') FROM docs WHERE doc_id IN ({marks})", chunk,
            ):
                result[doc_id] = (title or "", canonical_key(ext_id, doi))
    return result


def _resolve_pmid(pmid: str, store: Store) -> str:
    with store._lock:
        row = store.conn.execute(
            "SELECT ext_id, json_extract(meta, '$.doi') FROM docs WHERE ext_id=?", (f"pmid:{pmid}",)
        ).fetchone()
    return canonical_key(row[0], row[1]) if row else f"pmid:{pmid}"


def reference_keys(references: str, store: Store) -> list[str]:
    """参考文献の文章から、並び順のキーを作る。DOIのない文献は照合先の題名で特定する。"""
    found = _resolve_titles(references, store)
    positions = {}
    for m in _DOI_RE.finditer(references):
        positions.setdefault(m.group().rstrip(".,;)]>").lower(), m.start())
    # 行末で折り返されたDOIは元の文章に現れないので、位置は末尾扱いにする
    for doi in _dois_in(references):
        found.append((positions.get(doi, len(references)), f"doi:{doi}"))
    for m in _PMID_RE.finditer(references):
        found.append((m.start(), _resolve_pmid(m.group(1), store)))
    ordered = dict.fromkeys(key for _, key in sorted(found))
    return list(ordered)


def _manuscript_references(text: str) -> str | None:
    spans = spans_of(text, "references")
    return "\n".join(text[s:e] for s, e in spans) if spans else None


def reset_citations(store: Store):
    with store._lock, store.conn:
        store.conn.execute("DROP TABLE IF EXISTS citations")
        store.conn.execute("DELETE FROM citation_links")
        store.conn.execute("DELETE FROM citation_counts")


def _save_citations(store: Store, doc_id: int, keys: list[str]):
    with store._lock, store.conn:
        store.conn.executemany(
            "INSERT OR REPLACE INTO citation_links VALUES (?,?,?)",
            ((_hash(key), doc_id, i) for i, key in enumerate(keys)),
        )
        store.conn.execute("INSERT OR REPLACE INTO citation_counts VALUES (?,?)", (doc_id, len(keys)))


def index_library_citations(store: Store, progress=print) -> dict:
    """手元のPDFを読み直して引用の一覧を作る。索引には参考文献を入れていないため。"""
    doc_ids = store.iter_doc_ids("library")
    indexed = missing = 0
    for done, i in enumerate(range(0, len(doc_ids), 20), 1):
        for doc_id, doc in store.get_docs(doc_ids[i:i + 20]).items():
            path = doc.meta.get("path")
            if not path or not Path(path).exists():
                missing += 1
                continue
            try:
                references = _manuscript_references(extract_document(path).text)
            except Exception:
                missing += 1
                continue
            keys = reference_keys(references, store) if references else []
            if keys:
                _save_citations(store, doc_id, keys)
                indexed += 1
        if done % 10 == 0:
            progress(f"  手元のPDF {min(i + 20, len(doc_ids))}/{len(doc_ids)}")
    return {"library_indexed": indexed, "library_unreadable": missing}


def _canonical_refs(refs: list[dict], resolve) -> list[str]:
    keys = []
    for ref in refs:
        if ref.get("doi"):
            keys.append(f"doi:{ref['doi'].lower()}")
        elif ref.get("pmid"):
            keys.append(resolve(ref["pmid"]))
    return list(dict.fromkeys(keys))


def index_pubmed_citations(store: Store, progress=print) -> dict:
    """PubMedの引用を登録する。全件では数億行になるので、メモリ上で並べ替えてからまとめて書き込む。"""
    reader = store.conn.cursor()
    with store._lock:
        doi_by_pmid = {
            ext_id.removeprefix("pmid:"): doi.lower()
            for ext_id, doi in reader.execute(
                "SELECT ext_id, json_extract(meta, '$.doi') FROM docs "
                "WHERE source='pubmed' AND json_extract(meta, '$.doi') IS NOT NULL"
            )
        }
    progress(f"  DOIのある文献 {len(doi_by_pmid):,} 件")

    def resolve(pmid: str) -> str:
        doi = doi_by_pmid.get(pmid)
        return f"doi:{doi}" if doi else f"pmid:{pmid}"

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    refs_buf, docs_buf, positions_buf = [], [], []

    def flush():
        parts.append((
            np.array(refs_buf, dtype=np.int64), np.array(docs_buf, dtype=np.int32),
            np.array(positions_buf, dtype=np.int32),
        ))
        refs_buf.clear(), docs_buf.clear(), positions_buf.clear()

    indexed = without = 0
    with store._lock:
        for doc_id, refs in reader.execute(
            "SELECT doc_id, json_extract(meta, '$.refs') FROM docs WHERE source='pubmed' ORDER BY doc_id"
        ):
            keys = _canonical_refs(json.loads(refs), resolve) if refs else []
            if not keys:
                without += 1
                continue
            refs_buf.extend(_hash(key) for key in keys)
            docs_buf.extend([doc_id] * len(keys))
            positions_buf.extend(range(len(keys)))
            indexed += 1
            if len(refs_buf) >= _READ_BATCH:
                flush()
                progress(f"  引用あり {indexed:,} 件 / 引用 {sum(p[0].size for p in parts):,} 行")
    flush()
    del doi_by_pmid

    refs = np.concatenate([p[0] for p in parts])
    doc_ids = np.concatenate([p[1] for p in parts])
    positions = np.concatenate([p[2] for p in parts])
    del parts
    order = np.argsort(refs, kind="stable")  # 文献の順に読んだので、同じ引用の中も文献の順になる
    progress(f"  並べ替え完了 {order.size:,} 行")
    for i in range(0, order.size, _WRITE_BATCH):
        chunk = order[i:i + _WRITE_BATCH]
        with store._lock, store.conn:
            store.conn.executemany(
                "INSERT OR IGNORE INTO citation_links VALUES (?,?,?)",
                zip(refs[chunk].tolist(), doc_ids[chunk].tolist(), positions[chunk].tolist()),
            )
        if (i // _WRITE_BATCH) % 25 == 0:
            progress(f"  書き込み {min(i + _WRITE_BATCH, order.size):,}/{order.size:,} 行")
    counted, counts = np.unique(doc_ids, return_counts=True)
    with store._lock, store.conn:
        store.conn.executemany(
            "INSERT OR REPLACE INTO citation_counts VALUES (?,?)", zip(counted.tolist(), counts.tolist())
        )
    return {"pubmed_indexed": indexed, "pubmed_without_refs": without, "citations": int(order.size)}


def _longest_common_subsequence(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def coupled_documents(keys: list[str], store: Store, excluded_ids: set[int] = frozenset()) -> list[dict]:
    """同じ文献を引用している文書を、重なりの大きい順に返す。"""
    hashes = [_hash(key) for key in keys]
    shared: dict[int, list[tuple[int, int]]] = {}
    with store._lock:
        for i in range(0, len(hashes), _SQL_BATCH):
            chunk = hashes[i:i + _SQL_BATCH]
            marks = ",".join("?" * len(chunk))
            for doc_id, ref, position in store.conn.execute(
                f"SELECT doc_id, ref, position FROM citation_links WHERE ref IN ({marks})", chunk,
            ):
                if doc_id not in excluded_ids:
                    shared.setdefault(doc_id, []).append((position, ref))
    results = []
    for doc_id, hits in shared.items():
        if len(hits) < MIN_SHARED:
            continue
        with store._lock:
            total = store.conn.execute("SELECT n FROM citation_counts WHERE doc_id=?", (doc_id,)).fetchone()[0]
        source_order = [ref for _, ref in sorted(hits)]
        common = set(source_order)
        manuscript_order = [h for h in hashes if h in common]
        results.append({
            "doc_id": doc_id,
            "shared": len(hits),
            "manuscript_refs": len(keys),
            "source_refs": total,
            # 短いほうの一覧に対する割合。総説の長い一覧に引きずられないようにする
            "coupling": len(hits) / min(len(keys), total),
            "order_share": _longest_common_subsequence(manuscript_order, source_order) / len(hits),
        })
    return sorted(results, key=lambda r: (-r["coupling"], -r["shared"]))


def has_citation_index(store: Store) -> bool:
    with store._lock:
        return store.conn.execute("SELECT 1 FROM citation_links LIMIT 1").fetchone() is not None


def check_citations(text: str, store: Store, excluded_ids: set[int] = frozenset()) -> tuple[list[Flag], list[str]]:
    if not has_citation_index(store):
        return [], ["引用の一覧が未作成のため、引用パターンの確認は行っていません（index-citations で作成します）。"]
    references = _manuscript_references(text)
    if not references:
        return [], []
    keys = reference_keys(references, store)
    if len(keys) < MIN_MANUSCRIPT_REFS:
        return [], [
            f"参考文献のうち照合先で特定できたのが {len(keys)} 件だけなので、引用パターンは確認していません。"
        ]
    coupled = [r for r in coupled_documents(keys, store, excluded_ids) if r["coupling"] >= MIN_COUPLING]
    if not coupled:
        return [], []
    docs = store.get_docs([r["doc_id"] for r in coupled[:MAX_EXAMPLES]])
    examples = [
        f"{docs[r['doc_id']].title or docs[r['doc_id']].ext_id}（共通 {r['shared']} 件、"
        f"相手の文献一覧 {r['source_refs']} 件、順序の一致 {r['order_share'] * 100:.0f}%）"
        for r in coupled[:MAX_EXAMPLES] if r["doc_id"] in docs
    ]
    return [Flag(
        "引用の重なり",
        "参考文献の多くが同じ文書があります。言い換えや翻訳でも変わりにくい手がかりですが、同じ分野の論文や同じ著者の"
        "続報でも重なります。",
        len(coupled),
        examples,
    )], []
