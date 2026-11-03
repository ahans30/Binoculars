import re
from pathlib import Path

import numpy as np
import pymupdf

from .flags import Flag

# 1ページ約3秒かかるので、確認するページを絞る
MAX_PAGES = 4
RENDER_DPI = 150
MIN_PAGE_TOKENS = 60
# 普通のPDFと、文字の対応表を細工したPDFの間に置いた値（evaluation/detection/results.md）
MIN_AGREEMENT = 0.6

_LATIN_WORD = re.compile(r"[a-z]{4,}")
_CJK_RUN = re.compile(r"[぀-ヿ㐀-鿿]{2,}")
_ocr = None


def _tokens(text: str) -> set[str]:
    text = text.lower()
    tokens = set(_LATIN_WORD.findall(text))
    for run in _CJK_RUN.findall(text):
        tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR

        # モデルはパッケージに同梱されていて、原稿の画像は外部に送らない
        _ocr = RapidOCR(params={"Global.log_level": "error"})
    return _ocr


def _sample_pages(doc) -> list[tuple[int, set[str]]]:
    pages = []
    for number in range(doc.page_count):
        tokens = _tokens(doc[number].get_text())
        if len(tokens) >= MIN_PAGE_TOKENS:
            pages.append((number, tokens))
    if len(pages) <= MAX_PAGES:
        return pages
    step = len(pages) / MAX_PAGES
    return [pages[int(i * step)] for i in range(MAX_PAGES)]


def page_agreements(path: str | Path) -> list[tuple[int, float]]:
    """(ページ番号, 表示された語のうち抽出した文字にもある割合) を返す。"""
    ocr = _get_ocr()
    results = []
    with pymupdf.open(path) as doc:
        for number, layer in _sample_pages(doc):
            pix = doc[number].get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csRGB, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            seen = _tokens(" ".join(ocr(image).txts or []))
            if len(seen) >= MIN_PAGE_TOKENS:
                results.append((number + 1, len(seen & layer) / len(seen)))
    return results


def check_rendered_text(path: str | Path) -> tuple[list[Flag], list[str]]:
    """PDFの表示と抽出した文字を比べる。フォントの対応表を細工すると、見た目を変えずに抽出結果だけを変えられる。"""
    try:
        agreements = page_agreements(path)
    except ImportError:
        return [], ["rapidocr が未導入のため、PDFの表示と抽出文字の比較は行っていません。"]
    except Exception as e:
        return [], [f"PDFの表示と抽出文字の比較に失敗しました: {e}"]
    low = [(page, share) for page, share in agreements if share < MIN_AGREEMENT]
    if not low:
        return [], []
    return [Flag(
        "表示と抽出の不一致",
        "ページに表示された文字と、PDFから抽出した文字が大きく異なります。フォントの細工で照合を妨げている場合があり、"
        "その場合は類似度が実際より低く出ます。文字化けしたPDFでも起こります。",
        len(low),
        [f"p.{page}: 表示された語の {share * 100:.0f}% のみ抽出文字と一致" for page, share in low],
    )], []
