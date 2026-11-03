import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import docx
import pymupdf
from docx.table import Table
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from docx.text.run import Run

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")
_BAD_TITLE_RE = re.compile(r"\.(?:pdf|indd|docx?|tex|dvi)\b|^untitled|^microsoft word", re.IGNORECASE)

pymupdf.TOOLS.mupdf_display_errors(False)


@dataclass(slots=True)
class Extracted:
    text: str  # ページ区切りは \f
    title: str
    doi: str | None
    n_pages: int


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_running_lines(pages: list[str]) -> list[str]:
    # 全ページに繰り返される柱やフッターは、無関係な文書どうしの一致の原因になる
    if len(pages) < 4:
        return pages
    key = lambda line: re.sub(r"\d+", "", line).strip().lower()
    counts = Counter(k for page in pages for k in {key(line) for line in page.splitlines()} if k)
    limit = max(3, len(pages) // 2)
    repeated = {k for k, c in counts.items() if c >= limit and len(k) < 120}
    return ["\n".join(line for line in page.splitlines() if key(line) not in repeated) for page in pages]


def _extract_pdf(path: Path) -> Extracted:
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text") for page in doc]
        title = (doc.metadata or {}).get("title") or ""
    pages = _strip_running_lines(pages)
    text = unicodedata.normalize("NFC", "\f".join(pages))
    title = title.strip()
    if len(title) < 10 or _BAD_TITLE_RE.search(title):
        title = path.stem
    m = _DOI_RE.search(pages[0]) if pages else None
    doi = m.group().rstrip(".,;)") if m else None
    return Extracted(text, title, doi, len(pages))


def _is_superscript(run: Run) -> bool:
    if run.font.superscript:
        return True
    return run.font.superscript is None and bool(run.style and run.style.font.superscript)


def paragraph_pieces(paragraph: Paragraph) -> tuple[list[tuple[Run | Hyperlink, int, int]], str]:
    """段落の本文と、各 run が本文のどの範囲にあたるかを返す。"""
    # 上付きの文献番号が前の語とつながると（"2019" + "4,6,7"）、その語が一致しなくなる
    pieces, text, previous_super = [], "", False
    for item in paragraph.iter_inner_content():
        if not item.text:
            continue
        superscript = isinstance(item, Run) and _is_superscript(item)
        if text and superscript != previous_super:
            text += " "
        pieces.append((item, len(text), len(text) + len(item.text)))
        text += item.text
        previous_super = superscript
    return pieces, text


def _paragraph_text(paragraph: Paragraph) -> str:
    return paragraph_pieces(paragraph)[1]


def _extract_docx(path: Path) -> Extracted:
    d = docx.Document(str(path))
    # 表を末尾にまとめると参考文献の範囲に入ってしまうので、文書の順序どおりに読む
    parts = []
    for block in d.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                parts.append(" ".join(cell.text for cell in row.cells))
        else:
            parts.append(_paragraph_text(block))
    title = (d.core_properties.title or "").strip() or path.stem
    return Extracted(unicodedata.normalize("NFC", "\n".join(parts)), title, None, 1)


def _extract_text(path: Path) -> Extracted:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp932"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", "replace")
    # CRLF のままだと行単位の見出し検出が効かない
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return Extracted(unicodedata.normalize("NFC", text), path.stem, None, 1)


def extract_document(path: str | Path) -> Extracted:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in {".txt", ".md"}:
        return _extract_text(path)
    raise ValueError(f"未対応の形式です: {suffix}（対応: {', '.join(sorted(SUPPORTED_SUFFIXES))}）")
