import re
from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf
from docx.shared import Pt, RGBColor

from .textnorm import HOMOGLYPH_WORD, INVISIBLE_CHARS, _has_latin

_SOFT_HYPHEN = "\u00ad"
_ZERO_WIDTH = re.compile("[" + "".join(sorted(INVISIBLE_CHARS - {_SOFT_HYPHEN})) + "]")
_PRIVATE_USE = re.compile("[-]")
# "A B S T R A C T" のような字間の広い見出しは出版PDFで普通にあるので、長いものだけ拾う
_SPACED_LETTERS = re.compile(r"(?<![A-Za-z])(?:[A-Za-z] ){19,}[A-Za-z](?![A-Za-z])")
_MIXED_WORD = re.compile(r"[A-Za-zЀ-ӿ԰-֏]*(?:[A-Za-z][Ѐ-ӿ԰-֏]|[Ѐ-ӿ԰-֏][A-Za-z])[A-Za-zЀ-ӿ԰-֏]*")
# 点線リーダーや図中の1文字ラベルを数えないよう、単語らしい部分だけ数える
_WORD_LIKE = re.compile(r"[A-Za-z]{3,}|[぀-鿿]{2,}")

MIN_SUSPICIOUS_CHARS = 20
MIN_INVISIBLE_CHARS = 5
MIN_SOFT_HYPHENS = 20
MIN_HOMOGLYPH_WORDS = 3
MIN_MIXED_WORD_LEN = 4
TINY_FONT_PT = 3.0
WHITE = 0xFFFFFF
FULL_PAGE_RATIO = 0.7


@dataclass(slots=True)
class Flag:
    kind: str
    message: str
    count: int
    examples: list[str]


def check_text(text: str) -> list[Flag]:
    flags = []
    if (n := len(_ZERO_WIDTH.findall(text))) >= MIN_INVISIBLE_CHARS:
        flags.append(Flag("不可視文字", "目に見えない文字が入っています。語の照合を妨げる目的で使われることがあります。", n, []))
    # Webページからのコピーでも混ざるので、多いときだけ示す
    if (n := text.count(_SOFT_HYPHEN)) >= MIN_SOFT_HYPHENS:
        flags.append(Flag("ソフトハイフン", "表示されないハイフンが多く入っています。Webページからのコピーでも混ざります。", n, []))
    if (n := len(_PRIVATE_USE.findall(text))) >= MIN_INVISIBLE_CHARS:
        flags.append(Flag("私用領域の文字", "標準でない文字コードが含まれています。フォントの都合で生じることもあります。", n, []))
    mixed = [w for w in _MIXED_WORD.findall(text) if len(w) >= MIN_MIXED_WORD_LEN]
    # τau のように本物のギリシャ文字を含む語もあるので、複数あるときだけ数える
    homoglyphs = [m.group() for m in HOMOGLYPH_WORD.finditer(text) if not m.group().isascii() and _has_latin(m.group())]
    if len(homoglyphs) >= MIN_HOMOGLYPH_WORDS:
        mixed += homoglyphs
    if mixed:
        flags.append(Flag("文字の置き換え", "ラテン文字の単語に、形の似た他の文字が混ざっています。", len(mixed), mixed[:5]))
    spaced = _SPACED_LETTERS.findall(text)
    if spaced:
        flags.append(Flag("字間の空白", "1文字ごとに空白が入った箇所があります。", len(spaced), [s[:60] for s in spaced[:3]]))
    return flags


def _covered_by_dark_or_image(rect, fills) -> bool:
    return any(rect.intersects(fill) for fill in fills)


def check_pdf(path: str | Path) -> list[Flag]:
    white, tiny, offpage = [], [], []
    invisible_chars = 0
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, 1):
            page_rect = page.rect
            page_area = abs(page_rect) or 1
            image_rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
            # スキャンPDFのOCR層は透明文字で正常なので除外する
            is_scan = any(abs(r & page_rect) / page_area > FULL_PAGE_RATIO for r in image_rects)
            fills = list(image_rects)
            for drawing in page.get_drawings():
                fill = drawing.get("fill")
                if fill and sum(fill) < 2.4:
                    fills.append(pymupdf.Rect(drawing["rect"]))

            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        rect = pymupdf.Rect(span["bbox"])
                        if not rect.intersects(page_rect):
                            offpage.append((page_no, text))
                        elif span.get("size", 10) < TINY_FONT_PT:
                            tiny.append((page_no, text))
                        elif span.get("color") == WHITE and not _covered_by_dark_or_image(rect, fills):
                            white.append((page_no, text))

            if not is_scan:
                for span in page.get_texttrace():
                    if span.get("type") == 3 or span.get("opacity", 1) == 0:
                        invisible_chars += len(span.get("chars", ()))

    flags = []
    for kind, message, spans in (
        ("白文字", "背景に溶け込む白い文字があります。", white),
        ("極小文字", f"{TINY_FONT_PT:g}pt未満の文字があります。", tiny),
        ("ページ外の文字", "ページの表示範囲の外に文字があります。", offpage),
    ):
        worded = [(p, t) for p, t in spans if _WORD_LIKE.search(t)]
        chars = sum(len(t) for _, t in worded)
        if chars >= MIN_SUSPICIOUS_CHARS:
            examples = [f"p.{p}: {t[:60]}" for p, t in worded[:3]]
            flags.append(Flag(kind, message, chars, examples))
    if invisible_chars >= MIN_SUSPICIOUS_CHARS:
        flags.append(Flag("透明文字", "透明に描画された文字があります。", invisible_chars, []))
    return flags


def check_docx(path: str | Path) -> list[Flag]:
    hidden, white, tiny = [], [], []
    d = docx.Document(str(path))
    paragraphs = list(d.paragraphs)
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs += cell.paragraphs
    for paragraph in paragraphs:
        for run in paragraph.runs:
            text = run.text.strip()
            if not text:
                continue
            if run.font.hidden:
                hidden.append(text)
            elif run.font.size is not None and run.font.size < Pt(TINY_FONT_PT):
                tiny.append(text)
            elif run.font.color is not None and run.font.color.rgb == RGBColor(0xFF, 0xFF, 0xFF):
                white.append(text)
    flags = []
    for kind, message, spans in (
        ("隠し文字", "Wordの「隠し文字」書式の文字があります。", hidden),
        ("白文字", "白色に設定された文字があります。", white),
        ("極小文字", f"{TINY_FONT_PT:g}pt未満の文字があります。", tiny),
    ):
        chars = sum(len(t) for t in spans)
        if chars >= MIN_SUSPICIOUS_CHARS:
            flags.append(Flag(kind, message, chars, [t[:60] for t in spans[:3]]))
    return flags


def check_document(path: str | Path, text: str) -> list[Flag]:
    suffix = Path(path).suffix.lower()
    flags = check_text(text)
    if suffix == ".pdf":
        flags += check_pdf(path)
    elif suffix == ".docx":
        flags += check_docx(path)
    return flags
