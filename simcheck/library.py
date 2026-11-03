import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .extract import SUPPORTED_SUFFIXES, extract_document, sha256_file
from .sections import strip_references
from .store import DocRow, Store, compress_text
from .textnorm import fingerprints

# これより文字が少ないPDFはスキャン画像とみなす（OCRは未対応）
MIN_CHARS_PER_PAGE = 200
BATCH_SIZE = 50


def _process_pdf(path: str, sha: str):
    try:
        ext = extract_document(path)
    except Exception as e:
        return path, None, None, f"読み込み失敗: {e}"
    if len(ext.text) < MIN_CHARS_PER_PAGE * max(ext.n_pages, 1) / 4:
        return path, None, None, "テキスト層なし（スキャンPDFの可能性）"
    body = strip_references(ext.text)
    fps = fingerprints(body)
    if not fps:
        return path, None, None, "指紋を作れない（本文が短すぎる）"
    row = DocRow(
        ext_id=f"sha256:{sha}",
        source="library",
        title=ext.title,
        year=None,
        url=f"https://doi.org/{ext.doi}" if ext.doi else None,
        is_preprint=False,
        meta={"path": path, "doi": ext.doi, "n_pages": ext.n_pages},
        ztext=compress_text(body),
    )
    return path, row, np.fromiter(fps, dtype=np.uint64, count=len(fps)), None


def add_library(store: Store, root: str | Path, workers: int | None = None, progress=print) -> dict:
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES - {".md", ".txt"}]
    progress(f"{len(paths)} ファイルを確認します: {root}")

    jobs, duplicates = [], 0
    seen = set()
    for p in paths:
        sha = sha256_file(p)
        if sha in seen or store.has_ext_id(f"sha256:{sha}"):
            duplicates += 1
            continue
        seen.add(sha)
        jobs.append((str(p), sha))
    progress(f"新規 {len(jobs)} 件 / 重複または登録済み {duplicates} 件")

    added, skipped = 0, []
    rows, hash_parts, id_parts = [], [], []

    def flush():
        nonlocal rows, hash_parts, id_parts
        if not rows:
            return
        name = f"library_{abs(hash(rows[0].ext_id)):x}_{len(rows)}"
        store.add_batch(name, "library", rows, np.concatenate(hash_parts), np.concatenate(id_parts))
        rows, hash_parts, id_parts = [], [], []

    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_pdf, path, sha) for path, sha in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            path, row, fps, error = future.result()
            if error:
                skipped.append((path, error))
            else:
                id_parts.append(np.full(fps.size, len(rows), dtype=np.uint32))
                hash_parts.append(fps)
                rows.append(row)
                added += 1
                if len(rows) >= BATCH_SIZE:
                    flush()
            if done % 50 == 0:
                progress(f"  {done}/{len(jobs)} 処理済み")
    flush()
    return {"added": added, "duplicates": duplicates, "skipped": skipped}
