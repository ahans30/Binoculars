import gc
import json
import os
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import DATA_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id INTEGER PRIMARY KEY,
    ext_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    title TEXT,
    year TEXT,
    url TEXT,
    is_preprint INTEGER NOT NULL DEFAULT 0,
    meta TEXT,
    text BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS ingested (name TEXT PRIMARY KEY, kind TEXT, n_docs INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS segments (name TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS citation_links (
    ref INTEGER NOT NULL, doc_id INTEGER NOT NULL, position INTEGER NOT NULL, PRIMARY KEY (ref, doc_id)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS citation_counts (doc_id INTEGER PRIMARY KEY, n INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS title_keys (hash INTEGER NOT NULL, doc_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS title_keys_hash ON title_keys (hash);
INSERT OR IGNORE INTO counters VALUES ('next_doc_id', 1);
"""

# これより多くの文書に現れる指紋は定型句とみなし、候補検索に使わない
DEFAULT_MAX_DF = 5000


@dataclass(slots=True)
class DocRow:
    ext_id: str
    source: str
    title: str
    year: str | None
    url: str | None
    is_preprint: bool
    meta: dict
    ztext: bytes


@dataclass(slots=True)
class Doc:
    doc_id: int
    ext_id: str
    source: str
    title: str
    year: str | None
    url: str | None
    is_preprint: bool
    meta: dict
    text: str


def compress_text(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", "replace"), 6)


class Store:
    def __init__(self, root: str | Path = DATA_DIR):
        self.root = Path(root)
        self.seg_dir = self.root / "segments"
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        # Gradio はハンドラを別スレッドで呼ぶので接続を共有してロックで守る
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.root / "docs.sqlite", check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self._segments = None

    def close(self):
        self._segments = None
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 登録 ----

    def is_ingested(self, name: str) -> bool:
        with self._lock:
            return self.conn.execute("SELECT 1 FROM ingested WHERE name=?", (name,)).fetchone() is not None

    def has_ext_id(self, ext_id: str) -> bool:
        with self._lock:
            return self.conn.execute("SELECT 1 FROM docs WHERE ext_id=?", (ext_id,)).fetchone() is not None

    def add_batch(self, name: str, kind: str, rows: list[DocRow], hashes: np.ndarray, local_ids: np.ndarray):
        """文書と指紋を1単位で登録する。local_ids は rows の添字。登録した件数を返す。

        登録済みの文献は残す。doc_id が変わると、言い換え検索の索引や引用の一覧が指す先がなくなるため。
        """
        with self._lock:
            rows, hashes, local_ids = self._only_new(rows, hashes, local_ids)
            if not rows:
                with self.conn:
                    self.conn.execute("INSERT OR REPLACE INTO ingested VALUES (?,?,?,?)", (name, kind, 0, time.time()))
                return 0
            # 別のプロセスが同時に登録しても番号が重ならないよう、先に確保する
            with self.conn:
                self.conn.execute("UPDATE counters SET value=value+? WHERE name='next_doc_id'", (len(rows),))
                base = self.conn.execute("SELECT value FROM counters WHERE name='next_doc_id'").fetchone()[0] - len(rows)
            # 同じ検索式の再実行で、前回の指紋を上書きしない
            taken = self.conn.execute("SELECT 1 FROM segments WHERE name=?", (name,)).fetchone()
            segment = f"{name}_{base}" if taken else name
            order = np.argsort(hashes, kind="stable")
            self._write_segment(segment, hashes[order], (local_ids[order].astype(np.int64) + base).astype(np.uint32))
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO docs (doc_id, ext_id, source, title, year, url, is_preprint, meta, text)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        (base + i, r.ext_id, r.source, r.title, r.year, r.url, int(r.is_preprint),
                         json.dumps(r.meta, ensure_ascii=False), r.ztext)
                        for i, r in enumerate(rows)
                    ),
                )
                self.conn.execute("INSERT OR REPLACE INTO segments VALUES (?)", (segment,))
                self.conn.execute(
                    "INSERT OR REPLACE INTO ingested VALUES (?,?,?,?)", (name, kind, len(rows), time.time())
                )
        self._segments = None
        return len(rows)

    def _only_new(self, rows: list[DocRow], hashes: np.ndarray, local_ids: np.ndarray):
        seen = set()
        for i in range(0, len(rows), 500):
            chunk = [r.ext_id for r in rows[i:i + 500]]
            marks = ",".join("?" * len(chunk))
            seen.update(r[0] for r in self.conn.execute(f"SELECT ext_id FROM docs WHERE ext_id IN ({marks})", chunk))
        new_rows, new_index = [], np.full(len(rows), -1, dtype=np.int64)
        for i, row in enumerate(rows):
            if row.ext_id not in seen:
                seen.add(row.ext_id)
                new_index[i] = len(new_rows)
                new_rows.append(row)
        if len(new_rows) == len(rows):
            return rows, hashes, local_ids
        mapped = new_index[local_ids]
        keep = mapped >= 0
        return new_rows, hashes[keep], mapped[keep].astype(local_ids.dtype)

    def remove_source(self, source: str):
        """指定した種類の文書を消す。統合済みの索引に残る指紋は、検索時に文書が見つからず無視される。"""
        self._segments = None
        gc.collect()
        with self._lock, self.conn:
            names = [r[0] for r in self.conn.execute("SELECT name FROM ingested WHERE kind=?", (source,))]
            self.conn.execute("DELETE FROM docs WHERE source=?", (source,))
            self.conn.execute("DELETE FROM ingested WHERE kind=?", (source,))
            self.conn.executemany("DELETE FROM segments WHERE name=?", [(n,) for n in names])
        for n in names:
            for suffix in (".h.npy", ".d.npy"):
                (self.seg_dir / f"{n}{suffix}").unlink(missing_ok=True)

    def _write_segment(self, name: str, hashes: np.ndarray, doc_ids: np.ndarray):
        for suffix, arr in ((".h.npy", hashes), (".d.npy", doc_ids)):
            tmp = self.seg_dir / f"{name}{suffix}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, self.seg_dir / f"{name}{suffix}")

    # ---- 検索 ----

    def _load_segments(self):
        if self._segments is None:
            with self._lock:
                names = [r[0] for r in self.conn.execute("SELECT name FROM segments")]
            self._segments = [
                (
                    np.load(self.seg_dir / f"{n}.h.npy", mmap_mode="r"),
                    np.load(self.seg_dir / f"{n}.d.npy", mmap_mode="r"),
                )
                for n in names
            ]
        return self._segments

    def doc_count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def lookup(self, hashes, limit: int, max_df: int = DEFAULT_MAX_DF, region_of: dict[int, int] | None = None,
               per_region: int = 0) -> list[tuple[int, float]]:
        """共有する指紋の重み（珍しいほど高い）の合計が大きい順に (doc_id, 重み) を返す。

        region_of を渡すと、原稿の領域ごとの上位 per_region 件も加える。短い一致しかない文献が、
        原稿の別の箇所で定型句を多く共有する文献に押し出されないようにするため。
        """
        query = np.unique(np.fromiter(hashes, dtype=np.uint64))
        if query.size == 0:
            return []
        segments = self._load_segments()
        ranges = [(h.searchsorted(query, "left"), h.searchsorted(query, "right")) for h, _ in segments]
        df = np.zeros(query.size, dtype=np.int64)
        for left, right in ranges:
            df += right - left
        usable = (df > 0) & (df <= max_df)
        if not usable.any():
            return []
        n_docs = max(self.doc_count(), 1)
        idf = np.log1p(n_docs / np.maximum(df, 1))

        doc_parts, weight_parts, region_parts = [], [], []
        for (_, doc_ids), (left, right) in zip(segments, ranges):
            for q in np.flatnonzero(usable & (right > left)):
                part = np.asarray(doc_ids[left[q]:right[q]])
                doc_parts.append(part)
                weight_parts.append(np.full(part.size, idf[q]))
                if region_of:
                    region_parts.append(np.full(part.size, region_of.get(int(query[q]), -1)))
        if not doc_parts:
            return []
        ids, inverse = np.unique(np.concatenate(doc_parts), return_inverse=True)
        weights = np.concatenate(weight_parts)
        totals = np.bincount(inverse, weights=weights)
        chosen = dict.fromkeys(np.argsort(-totals)[:limit].tolist())
        if region_of and per_region:
            regions = np.concatenate(region_parts)
            for region in np.unique(regions[regions >= 0]):
                mask = regions == region
                local = np.bincount(inverse[mask], weights=weights[mask], minlength=ids.size)
                best = np.argpartition(-local, min(per_region, ids.size - 1))[:per_region]
                chosen.update(dict.fromkeys(int(i) for i in best if local[i] > 0))
        return [(int(ids[i]), float(totals[i])) for i in chosen]

    def document_frequencies(self, hashes) -> dict[int, int]:
        query = np.unique(np.fromiter(hashes, dtype=np.uint64))
        df = np.zeros(query.size, dtype=np.int64)
        for h, _ in self._load_segments():
            df += h.searchsorted(query, "right") - h.searchsorted(query, "left")
        return dict(zip(query.tolist(), df.tolist()))

    def get_docs(self, doc_ids: list[int]) -> dict[int, Doc]:
        result = {}
        with self._lock:
            for i in range(0, len(doc_ids), 500):
                chunk = doc_ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for row in self.conn.execute(
                    "SELECT doc_id, ext_id, source, title, year, url, is_preprint, meta, text"
                    f" FROM docs WHERE doc_id IN ({marks})",
                    chunk,
                ):
                    result[row[0]] = Doc(
                        row[0], row[1], row[2], row[3], row[4], row[5], bool(row[6]),
                        json.loads(row[7] or "{}"), zlib.decompress(row[8]).decode("utf-8", "replace"),
                    )
        return result

    def iter_doc_ids(self, source: str) -> list[int]:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT doc_id FROM docs WHERE source=? ORDER BY doc_id", (source,))]

    def id_for_ext(self, ext_id: str) -> int | None:
        with self._lock:
            row = self.conn.execute("SELECT doc_id FROM docs WHERE ext_id=?", (ext_id,)).fetchone()
        return row[0] if row else None

    def stats(self) -> dict:
        with self._lock:
            by_source = dict(self.conn.execute("SELECT source, COUNT(*) FROM docs GROUP BY source").fetchall())
            names = [r[0] for r in self.conn.execute("SELECT name FROM segments")]
        postings = 0
        for h, _ in self._load_segments():
            postings += h.shape[0]
        size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        return {
            "docs": by_source,
            "segments": len(names),
            "postings": postings,
            "disk_gb": round(size / 1024**3, 3),
        }

    # ---- 圧縮 ----

    def compact(self, name: str = "main") -> int:
        """全セグメントを1つに統合する。セグメント数が多いと検索時のディスク読み込みが増えるため。"""
        segments = self._load_segments()
        if len(segments) <= 1:
            return len(segments)
        total = sum(h.shape[0] for h, _ in segments)
        new_name = f"{name}_{int(time.time())}"
        h_tmp = self.seg_dir / f"{new_name}.h.npy.tmp"
        d_tmp = self.seg_dir / f"{new_name}.d.npy.tmp"
        h_out = np.lib.format.open_memmap(h_tmp, mode="w+", dtype=np.uint64, shape=(total,))
        d_out = np.lib.format.open_memmap(d_tmp, mode="w+", dtype=np.uint32, shape=(total,))

        # 上位8ビットで分けて順に並べれば、メモリを抑えたまま全体が整列する
        pos = 0
        for bucket in range(256):
            low = np.uint64(bucket << 56)
            high = np.uint64((bucket + 1) << 56) if bucket < 255 else None
            h_parts, d_parts = [], []
            for h, d in segments:
                a = int(h.searchsorted(low, "left"))
                b = int(h.searchsorted(high, "left")) if high is not None else h.shape[0]
                if b > a:
                    h_parts.append(np.asarray(h[a:b]))
                    d_parts.append(np.asarray(d[a:b]))
            if not h_parts:
                continue
            merged_h = np.concatenate(h_parts)
            merged_d = np.concatenate(d_parts)
            order = np.argsort(merged_h, kind="stable")
            h_out[pos:pos + order.size] = merged_h[order]
            d_out[pos:pos + order.size] = merged_d[order]
            pos += order.size
        h_out.flush()
        d_out.flush()
        del h_out, d_out

        os.replace(h_tmp, self.seg_dir / f"{new_name}.h.npy")
        os.replace(d_tmp, self.seg_dir / f"{new_name}.d.npy")

        with self._lock, self.conn:
            old = [r[0] for r in self.conn.execute("SELECT name FROM segments")]
            self.conn.execute("DELETE FROM segments")
            self.conn.execute("INSERT INTO segments VALUES (?)", (new_name,))
        # Windows では mmap 中のファイルを消せないので先に手放す
        del segments
        self._segments = None
        gc.collect()
        for n in old:
            for suffix in (".h.npy", ".d.npy"):
                try:
                    (self.seg_dir / f"{n}{suffix}").unlink()
                except OSError:
                    pass
        return 1
