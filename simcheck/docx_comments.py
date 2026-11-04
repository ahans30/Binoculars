"""原稿のコピーに、著者が選んだ対応をWordのコメントとして付ける。本文は変更しない。"""
import copy
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from .extract import paragraph_pieces

AUTHOR = "Binoculars"
INITIALS = "BC"
_PLAIN_RUN_TAGS = {"rPr", "t"}


@dataclass(slots=True)
class Comment:
    sentence: str
    text: str


def _paragraphs(document):
    yield from document.paragraphs
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _is_plain(run: Run) -> bool:
    return all(child.tag.rsplit("}", 1)[-1] in _PLAIN_RUN_TAGS for child in run._r)


def _split(run: Run, offset: int) -> Run:
    """run を offset で2つに分け、後ろの run を返す。書式は両方に残す。"""
    text = run.text
    tail = copy.deepcopy(run._r)
    run._r.addnext(tail)
    run.text = text[:offset]
    tail_run = Run(tail, run._parent)
    tail_run.text = text[offset:]
    return tail_run


def _anchor_runs(paragraph: Paragraph, start: int, end: int) -> list[Run]:
    # 文の後ろから分けると、前の位置がずれない
    for boundary in (end, start):
        for item, s, e in paragraph_pieces(paragraph)[0]:
            if isinstance(item, Run) and s < boundary < e and _is_plain(item):
                _split(item, boundary - s)
                break
    pieces, _ = paragraph_pieces(paragraph)
    runs = [item for item, s, e in pieces if isinstance(item, Run) and s < end and start < e]
    return runs or [item for item, _, _ in pieces if isinstance(item, Run)]


def _locate(document, sentence: str, used: set[tuple[int, int]]):
    wanted = unicodedata.normalize("NFC", sentence)
    for number, paragraph in enumerate(_paragraphs(document)):
        _, text = paragraph_pieces(paragraph)
        if not text:
            continue
        exact = unicodedata.normalize("NFC", text) == text
        position = text.find(wanted) if exact else (0 if wanted in unicodedata.normalize("NFC", text) else -1)
        while position >= 0 and (number, position) in used:
            position = text.find(wanted, position + 1) if exact else -1
        if position >= 0:
            used.add((number, position))
            # 正規化で文字位置がずれる段落では、段落全体をコメントの範囲にする
            return paragraph, position, position + len(wanted) if exact else len(text)
    return None


def write_comments(source: str | Path, comments: list[Comment], out: str | Path) -> dict:
    """コメントを付けたコピーを out に保存する。位置を特定できなかった文は missed で返す。"""
    source, out = Path(source), Path(out)
    if source.suffix.lower() != ".docx":
        raise ValueError("コメントを付けられるのはDOCXだけです。")
    if source.resolve() == out.resolve():
        raise ValueError("元のファイルには書き込みません。")
    document = docx.Document(str(source))
    used: set[tuple[int, int]] = set()
    written, missed = 0, []
    for comment in comments:
        place = _locate(document, comment.sentence, used)
        if place is None:
            missed.append(comment)
            continue
        paragraph, start, end = place
        runs = _anchor_runs(paragraph, start, end)
        if not runs:
            missed.append(comment)
            continue
        document.add_comment(runs, text=comment.text, author=AUTHOR, initials=INITIALS)
        written += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(out))
    return {"written": written, "missed": missed, "path": out}
