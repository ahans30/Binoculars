"""確認結果を、表紙・概要・出典一覧・ハイライト入りの原稿ページからなるPDFにする。"""
import os
import re
import shutil
import subprocess
import tempfile
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

import pymupdf
from rapidfuzz.distance import Levenshtein

from .engine import Report
from .extract import sha256_file
from .report import _PALETTE, corpus_summary
from .textnorm import WORD_WEIGHT, Tokens, tokenize
from .triage import LABELS

_FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_REGULAR = ("YuGothR.ttc", "meiryo.ttc", "msgothic.ttc")
_BOLD = ("YuGothB.ttc", "meiryob.ttc", "msgothic.ttc")
_WORD_TIMEOUT = 180
# 文字位置が取れた語の割合がこれ未満なら、ハイライトの漏れを表紙に明記する
_MIN_MAPPED = 0.9

# 原稿ページを縮小して、左に番号、上下にヘッダーとフッターの余白を作る
_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_Y = 42, 5, 30
_BADGE_W, _BADGE_H, _BADGE_GAP = 14, 11, 1
_BADGE_SLOTS = 3
_CONTENT_X, _CONTENT_TOP, _CONTENT_BOTTOM = 40, 50, 60
_MAX_PDF_EXAMPLES = 5


def _rgb(value: str) -> tuple[float, float, float]:
    return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))


_INK = _rgb("#1f2933")
_MUTED = _rgb("#5f6c7b")
_LINE = _rgb("#d9dee5")
_CARD = _rgb("#f7f8fa")
_NOTE = _rgb("#dbeafe")
_HIGHLIGHT = _rgb("#f87171")
_EXCLUDED = _rgb("#9aa5b1")
_SEMANTIC = _rgb("#b45309")
_KIND_COLORS = {"PubMed抄録": "#bfdbfe", "PMC全文": "#a5f3fc", "手元のPDF": "#bbf7d0", "プレプリント": "#ddd6fe"}
_TRIAGE_COLORS = {"review": "#fecaca", "boilerplate": "#e2e8f0", None: "#fde68a"}
_TRIAGE_ORDER = {"review": 0, None: 1, "boilerplate": 2}
_TRIAGE_MAX_CHARS = 400


def _font(names: tuple[str, ...]) -> pymupdf.Font:
    for name in names:
        path = _FONT_DIR / name
        if path.exists():
            return pymupdf.Font(fontfile=str(path))
    return pymupdf.Font("cjk")


class _Writer:
    """日本語を含むページを順に描くための薄い補助。"""

    def __init__(self, doc: pymupdf.Document, size: pymupdf.Rect):
        self.doc = doc
        self.size = size
        self.regular = _font(_REGULAR)
        self.bold = _font(_BOLD)
        # 区分ごとの開始ページ。ヘッダーを最後にまとめて描くときに使う
        self.sections: list[tuple[int, str]] = []

    def new_page(self, section: str | None = None) -> pymupdf.Page:
        page = self.doc.new_page(width=self.size.width, height=self.size.height)
        if section:
            self.sections.append((page.number, section))
        return page

    def text(self, page, x, y, text, size=10.0, color=_INK, bold=False):
        writer = pymupdf.TextWriter(page.rect)
        writer.append((x, y), text, font=self.bold if bold else self.regular, fontsize=size)
        writer.write_text(page, color=color)

    def width(self, text, size, bold=False) -> float:
        return (self.bold if bold else self.regular).text_length(text, fontsize=size)

    def wrap(self, text: str, size: float, width: float, bold=False) -> list[str]:
        # 英単語は途中で切らず、それ以外は1文字ずつ折り返す
        lines, current = [], ""
        for chunk in re.findall(r"[!-~]+\s*|\s+|.", text.replace("\n", " ")):
            if current and self.width(current + chunk.rstrip(), size, bold) > width:
                lines.append(current.rstrip())
                current = chunk.lstrip()
            else:
                current += chunk
        if current.strip():
            lines.append(current.rstrip())
        return lines

    def paragraph(self, page, x, y, text, width, size=9.0, color=_INK, leading=1.5, bold=False) -> float:
        for line in self.wrap(text, size, width, bold):
            self.text(page, x, y, line, size, color, bold)
            y += size * leading
        return y

    def ellipsize(self, text: str, size: float, width: float, bold=False) -> str:
        if self.width(text, size, bold) <= width:
            return text
        while text and self.width(text + "…", size, bold) > width:
            text = text[:-1]
        return text + "…"

    def pill(self, page, x, y, label, fill, height=14.0, width=None, size=8.0) -> float:
        width = width or self.width(label, size) + 14
        rect = pymupdf.Rect(x, y, x + width, y + height)
        page.draw_rect(rect, color=None, fill=fill, radius=0.5, overlay=True)
        self.text(page, x + (width - self.width(label, size)) / 2, y + height / 2 + size * 0.36, label, size)
        return width

    def rule(self, page, y):
        page.draw_line((_CONTENT_X, y), (page.rect.width - _CONTENT_X, y), color=_LINE, width=0.7)


def _word_to_pdf(path: Path, out: Path) -> bool:
    # 開いている文書に触れないよう、作業用のコピーを別のWordで変換する。文書が残っていたらWordを閉じない
    script = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0; "
        "try { $d = $w.Documents.Open($env:SIMCHECK_IN, $false, $true, $false); "
        "$d.ExportAsFixedFormat($env:SIMCHECK_OUT, 17); $d.Close(0) } "
        "finally { if ($w.Documents.Count -eq 0) { $w.Quit() } }"
    )
    env = {**os.environ, "SIMCHECK_IN": str(path), "SIMCHECK_OUT": str(out)}
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env=env, capture_output=True, timeout=_WORD_TIMEOUT, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.exists()


def _soffice_to_pdf(path: Path) -> Path | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        subprocess.run(
            [soffice, f"-env:UserInstallation={(path.parent / 'profile').as_uri()}", "--headless", "--norestore",
             "--convert-to", "pdf", "--outdir", str(path.parent), str(path)],
            capture_output=True, timeout=_WORD_TIMEOUT, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = path.with_suffix(".pdf")
    return out if out.exists() else None


def _typeset(text: str) -> pymupdf.Document:
    doc = pymupdf.open()
    writer = _Writer(doc, pymupdf.paper_rect("a4"))
    left, top, size, leading = 72, 80, 10.5, 16
    width = writer.size.width - left * 2
    for block in text.split("\f"):
        page, y = writer.new_page(), top
        for paragraph in block.split("\n"):
            for line in writer.wrap(paragraph, size, width) or [""]:
                if y > writer.size.height - top:
                    page, y = writer.new_page(), top
                if line:
                    writer.text(page, left, y, line, size)
                y += leading
    return doc


def _open_base(path: Path, text: str) -> tuple[pymupdf.Document, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pymupdf.open(path), []
    if suffix == ".docx":
        work = Path(tempfile.mkdtemp())
        try:
            shutil.copy(path, work / "input.docx")
            if _word_to_pdf(work / "input.docx", work / "output.pdf"):
                # 作業フォルダを消せるようにメモリへ読み込む
                return pymupdf.open("pdf", (work / "output.pdf").read_bytes()), []
            converted = _soffice_to_pdf(work / "input.docx")
            if converted:
                return pymupdf.open("pdf", converted.read_bytes()), [
                    "LibreOfficeで変換したため、原稿ページのページと行はWordでの表示と異なる場合があります。"
                ]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return _typeset(text), ["Wordで開けなかったため、原稿ページは本文だけを組み直したものです。ページと行は原稿と一致しません。"]
    return _typeset(text), []


def _collect_chars(doc: pymupdf.Document) -> tuple[str, list]:
    flags = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    chars, info = [], []
    for page in doc:
        for block in page.get_text("rawdict", flags=flags)["blocks"]:
            for line_no, line in enumerate(block.get("lines", [])):
                for span in line["spans"]:
                    # 本文の抽出と同じく、上付きの文献番号は前後の語から離す
                    superscript = span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT
                    if superscript:
                        chars.append(" ")
                        info.append(None)
                    for ch in span["chars"]:
                        chars.append(ch["c"])
                        info.append((page.number, (block["number"], line_no), ch["bbox"]))
                    if superscript:
                        chars.append(" ")
                        info.append(None)
                chars.append(" ")
                info.append(None)
    return "".join(chars), info


class _Locator:
    """抽出した本文の文字位置を、原稿ページ上の座標に対応づける。"""

    def __init__(self, text: str, base: pymupdf.Document):
        joined, self.info = _collect_chars(base)
        self.doc_tokens: Tokens = tokenize(text)
        self.pdf_tokens: Tokens = tokenize(joined)
        # 行番号など原稿ページにしかない語があるので、全体を編集距離で対応づける
        self.mapping = [-1] * len(self.doc_tokens)
        for op in Levenshtein.opcodes(self.doc_tokens.tokens, self.pdf_tokens.tokens):
            if op.tag == "equal" or (op.tag == "replace" and op.src_end - op.src_start == op.dest_end - op.dest_start):
                for k in range(op.src_end - op.src_start):
                    self.mapping[op.src_start + k] = op.dest_start + k
        self.mapped_share = sum(j >= 0 for j in self.mapping) / max(len(self.mapping), 1)

    def rects(self, start: int, end: int) -> list[tuple[int, pymupdf.Rect]]:
        """範囲が占める領域を、ページ内の行ごとに出現順で返す。"""
        a = bisect_left(self.doc_tokens.starts, start)
        b = bisect_left(self.doc_tokens.starts, end)
        lines: dict[tuple, pymupdf.Rect] = {}
        for i in range(a, b):
            j = self.mapping[i]
            if j < 0:
                continue
            for k in range(self.pdf_tokens.starts[j], self.pdf_tokens.ends[j]):
                entry = self.info[k]
                if entry is None:
                    continue
                key = (entry[0], entry[1])
                if key in lines:
                    lines[key] |= pymupdf.Rect(entry[2])
                else:
                    lines[key] = pymupdf.Rect(entry[2])
        return [(key[0], rect) for key, rect in lines.items()]


def _pct(value: float) -> str:
    return "<0.1%" if 0 < value < 0.1 else f"{value:.1f}%"


def _kind(doc) -> str:
    if doc.is_preprint:
        return "プレプリント"
    return {"pubmed": "PubMed抄録", "pmc": "PMC全文"}.get(doc.source, "手元のPDF")


def _words(text: str) -> float:
    return sum(tokenize(text).weights) / WORD_WEIGHT


def _match_groups(report: Report) -> dict:
    # 一致箇所を、出典とどれだけ同じかで分ける
    groups = {"near": [0, 0.0], "partial": [0, 0.0]}
    for s, e, rank in report.owned_spans:
        overlapping = [m.similarity for m in report.sources[rank].matches if m.q_start < e and s < m.q_end]
        key = "near" if max(overlapping, default=0) >= 0.95 else "partial"
        groups[key][0] += 1
        groups[key][1] += _words(report.text[s:e])
    return groups


def _filters(report: Report) -> list[str]:
    options = report.options
    filters = []
    if options.exclude_references:
        filters.append("参考文献")
    if options.exclude_quotes:
        filters.append("引用符内")
    if options.exclude_methods:
        filters.append("方法セクション")
    if options.exclude_common:
        filters.append("定型表現（ライセンス表記など）")
    if options.exclude_preprints:
        filters.append("プレプリント")
    filters.append(f"{options.min_words}語未満の一致")
    if options.exclude_ext_ids:
        filters.append(f"指定した出典 {len(options.exclude_ext_ids)} 件")
    return filters


def _cover(w: _Writer, report: Report, path: Path, check_id: str, base_pages: int, notices: list[str]):
    page = w.new_page("表紙")
    right = page.rect.width - _CONTENT_X
    y = page.rect.height * 0.45
    for line in w.wrap(report.file_name, 15, right - _CONTENT_X, bold=True):
        w.text(page, _CONTENT_X, y, line, 15, bold=True)
        y += 21
    w.text(page, _CONTENT_X, y, "Binoculars 類似度チェック（投稿前の自己確認）", 9, _MUTED)
    y += 14
    w.rule(page, y)
    y += 24
    w.text(page, _CONTENT_X, y, "文書の詳細", 11, bold=True)
    box_top = y + 12
    y += 24
    details = (
        ("確認ID", check_id),
        ("確認日時", datetime.now().astimezone().strftime("%Y年%m月%d日 %H:%M (UTC%z)")),
        ("ファイル名", report.file_name),
        ("ファイルサイズ", f"{path.stat().st_size / 1024:.1f} KB"),
    )
    for label, value in details:
        w.text(page, _CONTENT_X, y, label, 8, _MUTED)
        w.text(page, _CONTENT_X, y + 13, w.ellipsize(value, 8.5, 340, bold=True), 8.5, bold=True)
        y += 34

    docs = report.index_stats.get("docs", {})
    box = pymupdf.Rect(right - 130, box_top, right, box_top + 76)
    page.draw_rect(box, color=None, fill=_CARD, overlay=True)
    total_words = report.words_checked + report.words_excluded
    stats = (f"{base_pages} ページ", f"{total_words:,.0f} 語", f"{len(report.text.replace(chr(12), '')):,} 文字")
    for i, line in enumerate(stats):
        w.text(page, box.x0 + 12, box.y0 + 20 + i * 20, line, 8.5, bold=True)

    y += 6
    w.rule(page, y)
    y += 18
    y = w.paragraph(
        page, _CONTENT_X, y,
        f"照合先: {corpus_summary(docs)}。"
        "商用サービスの判定結果ではなく、データベースも計算方法も異なるので、値は比較できません。"
        "照合先にない文献との一致は検出できないため、低い値でも盗用がないことの証明にはなりません。",
        right - _CONTENT_X, 8, _MUTED,
    )
    for notice in notices:
        y = w.paragraph(page, _CONTENT_X, y + 4, notice, right - _CONTENT_X, 8, _SEMANTIC)


def _overview(w: _Writer, report: Report):
    page = w.new_page("概要")
    right = page.rect.width - _CONTENT_X
    middle = page.rect.width / 2 + 10
    bottom = page.rect.height - _CONTENT_BOTTOM
    denominator = report.words_checked or 1

    y = _CONTENT_TOP + 22
    headline = _pct(report.score * 100)
    w.text(page, _CONTENT_X, y, headline, 20, bold=True)
    w.text(page, _CONTENT_X + w.width(headline, 20, True) + 12, y, "全体的な類似性", 16)
    y += 16
    w.text(page, _CONTENT_X, y, "照合先と一致した語の割合。複数の出典に一致する箇所は1回だけ数えます。", 8, _MUTED)
    y += 26
    w.text(page, _CONTENT_X, y, "レポートからフィルタ済み", 10.5, bold=True)
    y += 16
    for item in _filters(report):
        w.text(page, _CONTENT_X + 4, y, f"▸ {item}", 8)
        y += 13
    w.text(page, _CONTENT_X + 4, y, f"除外した語数: {report.words_excluded:,.0f} 語 / 確認対象: {report.words_checked:,.0f} 語", 8, _MUTED)
    y += 12
    w.rule(page, y)
    y += 24

    # 左に一致の内訳、右に照合先ごとの内訳を並べる
    w.text(page, _CONTENT_X, y, "一致グループ", 10.5, bold=True)
    w.text(page, middle, y, "照合先ごとの内訳", 10.5, bold=True)
    top = y + 16
    groups = _match_groups(report)
    rows = (
        (groups["near"][0], _pct(groups["near"][1] / denominator * 100), "ほぼ同じ一致", "出典との一致率が95%以上", "#fecaca"),
        (groups["partial"][0], _pct(groups["partial"][1] / denominator * 100), "一部が異なる一致", "出典との一致率が95%未満", "#fed7aa"),
        (len(report.semantic_hits), "—", "言い換えの候補", "類似度には含めません", "#fde68a"),
    )
    y = top
    for count, share, title, note, color in rows:
        w.pill(page, _CONTENT_X, y - 4, str(count), _rgb(color), height=13, width=20, size=7.5)
        w.text(page, _CONTENT_X + 26, y + 5, f"{title}  {share}", 8.5, bold=True)
        w.text(page, _CONTENT_X + 26, y + 17, note, 7.5, _MUTED)
        y += 30

    shares = {kind: 0.0 for kind in _KIND_COLORS}
    for source in report.sources:
        shares[_kind(source.doc)] += source.unique_weight / WORD_WEIGHT / denominator * 100
    side = top
    for kind, share in shares.items():
        w.pill(page, middle, side - 4, _pct(share), _rgb(_KIND_COLORS[kind]), height=13, width=38, size=7.5)
        w.text(page, middle + 44, side + 5, kind, 8.5)
        side += 22
    contributing = sum(1 for source in report.sources if source.unique_weight)
    w.text(page, middle, side + 5, f"一致した出典 {contributing} 件", 8, _MUTED)

    y = max(y, side + 14)
    w.rule(page, y)
    y += 24
    w.text(page, _CONTENT_X, y, "インテグリティフラグ", 10.5, bold=True)
    y += 16
    if report.flags:
        w.text(page, _CONTENT_X, y, f"レビューの対象となる {len(report.flags)} 個のフラグ", 8.5, bold=True)
        y += 15
        for flag in report.flags:
            lines = [(f"▸ {flag.kind}（{flag.count:,} 件）: {flag.message}", 0, 8, _INK)]
            lines += [(example, 10, 7.5, _MUTED) for example in flag.examples[:_MAX_PDF_EXAMPLES]]
            if len(flag.examples) > _MAX_PDF_EXAMPLES:
                rest = len(flag.examples) - _MAX_PDF_EXAMPLES
                lines.append((f"ほか {rest} 件はHTMLレポートを参照してください。", 10, 7.5, _MUTED))
            for line, indent, size, color in lines:
                if y > bottom:
                    page, y = w.new_page(), _CONTENT_TOP + 8
                y = w.paragraph(page, _CONTENT_X + indent, y, line, right - _CONTENT_X - indent, size, color)
            y += 3
    else:
        w.text(page, _CONTENT_X, y, "レビューの対象となる 0 個のフラグ", 8.5, bold=True)
        y += 15
        w.text(page, _CONTENT_X, y, "不審なテキスト操作は見つかりませんでした。", 8)
        y += 12

    y += 10
    if y + 44 > bottom:
        page, y = w.new_page(), _CONTENT_TOP + 8
    note = pymupdf.Rect(_CONTENT_X, y, right, y + 44)
    page.draw_rect(note, color=None, fill=_NOTE, radius=0.12, overlay=True)
    w.paragraph(
        page, note.x0 + 12, note.y0 + 16,
        "フラッグと言い換えの候補は確認のための手がかりで、不正の証拠ではありません。"
        "図の中の文字やPDFの作成方法が原因で出ることもあるので、内容を確認してください。",
        note.width - 24, 8,
    )


def _source_list(w: _Writer, report: Report) -> dict[int, int]:
    pages: dict[int, int] = {}
    denominator = report.words_checked or 1
    page = w.new_page("出典一覧")
    right = page.rect.width - _CONTENT_X
    bottom = page.rect.height - _CONTENT_BOTTOM
    y = _CONTENT_TOP + 8
    w.text(page, _CONTENT_X, y, "上位の出典", 11, bold=True)
    y += 15
    w.text(page, _CONTENT_X, y, "確認対象の語数に対する寄与の大きい順です。重複する出典は表示せず、合計が類似度になります。", 8, _MUTED)
    y += 18
    if not report.sources:
        w.text(page, _CONTENT_X, y, "設定した最小語数以上の一致はありませんでした。", 9)
        return pages
    for rank, source in enumerate(report.sources):
        # 一致がすべて他の出典と重複するものは数字に寄与しないので載せない
        if not source.unique_weight:
            continue
        if y + 40 > bottom:
            page = w.new_page()
            y = _CONTENT_TOP
        pages[rank] = page.number
        w.pill(page, _CONTENT_X, y, str(rank + 1), _rgb(_PALETTE[rank % len(_PALETTE)]), width=26)
        kind = _kind(source.doc)
        w.pill(page, _CONTENT_X + 32, y, kind, _rgb(_KIND_COLORS[kind]), size=7.5)
        count = f"{len(source.matches)} 箇所"
        w.text(page, right - w.width(count, 7.5), y + 10, count, 7.5, _MUTED)
        y += 28
        title = (source.doc.title or source.doc.ext_id).strip()
        share = _pct(source.unique_weight / WORD_WEIGHT / denominator * 100)
        w.text(page, _CONTENT_X, y, w.ellipsize(title, 8.5, right - _CONTENT_X - 50, bold=True), 8.5, bold=True)
        w.text(page, right - w.width(share, 8.5, True), y, share, 8.5, bold=True)
        if source.doc.url:
            page.insert_link({
                "kind": pymupdf.LINK_URI, "uri": source.doc.url,
                "from": pymupdf.Rect(_CONTENT_X, y - 10, right - 50, y + 3),
            })
        y += 8
        w.rule(page, y)
        y += 10
    return pages


def _manuscript(w: _Writer, report: Report, base: pymupdf.Document, locator: _Locator, source_pages: dict[int, int]) -> dict[tuple[int, int], int]:
    first = w.doc.page_count
    w.sections.append((first, "原稿"))
    transforms = []
    for src in base:
        width, height = src.rect.width, src.rect.height
        page = w.doc.new_page(width=width, height=height)
        scale = min((width - _MARGIN_LEFT - _MARGIN_RIGHT) / width, (height - _MARGIN_Y * 2) / height)
        origin = pymupdf.Point(_MARGIN_LEFT, (height - height * scale) / 2)
        page.show_pdf_page(pymupdf.Rect(origin, origin + (width * scale, height * scale)), base, src.number)
        transforms.append((scale, origin))

    def place(pno: int, rect: pymupdf.Rect) -> tuple[pymupdf.Page, pymupdf.Rect]:
        scale, origin = transforms[pno]
        return w.doc[first + pno], pymupdf.Rect(
            rect.x0 * scale + origin.x, rect.y0 * scale + origin.y,
            rect.x1 * scale + origin.x, rect.y1 * scale + origin.y,
        )

    def wash(spans, color, opacity):
        for s, e, *_ in spans:
            for pno, rect in locator.rects(s, e):
                page, placed = place(pno, rect)
                page.draw_rect(placed, color=None, fill=color, fill_opacity=opacity, overlay=True)

    wash(report.excluded_spans, _EXCLUDED, 0.18)
    wash(report.owned_spans, _HIGHLIGHT, 0.45)

    # 言い換えの候補は点線の下線で示し、参照先のページを返す
    hit_pages: dict[tuple[int, int], int] = {}
    for hit in report.semantic_hits:
        for i, (pno, rect) in enumerate(locator.rects(hit.q_start, hit.q_end)):
            page, placed = place(pno, rect)
            page.draw_line(placed.bl, placed.br, color=_SEMANTIC, width=0.9, dashes="[2 1.5] 0", overlay=True)
            if i == 0:
                hit_pages[(hit.q_start, hit.q_end)] = page.number + 1

    # 一致の始まる行の左余白に出典番号を出す
    slots: dict[tuple[int, int], list[int]] = {}
    for s, e, rank in report.owned_spans:
        rects = locator.rects(s, e)
        if not rects:
            continue
        pno, rect = rects[0]
        page, placed = place(pno, rect)
        key = (page.number, round(placed.y0))
        used = slots.setdefault(key, [])
        if rank in used:
            continue
        used.append(rank)
        if len(used) > _BADGE_SLOTS:
            continue
        x = 2 + (len(used) - 1) * (_BADGE_W + _BADGE_GAP)
        label = str(rank + 1) if len(used) < _BADGE_SLOTS else "+"
        y = (placed.y0 + placed.y1 - _BADGE_H) / 2
        w.pill(page, x, y, label, _rgb(_PALETTE[rank % len(_PALETTE)]), height=_BADGE_H, width=_BADGE_W, size=7)
        if rank in source_pages:
            page.insert_link({
                "kind": pymupdf.LINK_GOTO, "page": source_pages[rank], "to": pymupdf.Point(0, 0),
                "from": pymupdf.Rect(x, y, x + _BADGE_W, y + _BADGE_H),
            })
    return hit_pages


def _semantic_appendix(w: _Writer, report: Report, hit_pages: dict[tuple[int, int], int]):
    if not report.semantic_hits:
        return
    page = w.new_page("言い換えの候補")
    right = page.rect.width - _CONTENT_X
    bottom = page.rect.height - _CONTENT_BOTTOM
    y = _CONTENT_TOP + 8
    w.text(page, _CONTENT_X, y, "言い換えの候補", 11, bold=True)
    y += 15
    y = w.paragraph(
        page, _CONTENT_X, y,
        "意味が近い文の候補で、類似度には含めていません。一般的な表現でも挙がるので、機械的に盗用と判断せず内容を確認してください。"
        "原稿ページでは点線の下線で示しています。出典の本文はHTMLレポートで確認できます。",
        right - _CONTENT_X, 8, _MUTED,
    )
    y += 10
    for hit in sorted(report.semantic_hits, key=lambda h: -h.score):
        lines = w.wrap(report.text[hit.q_start:hit.q_end], 8.5, right - _CONTENT_X)
        if y + 28 + len(lines) * 13 > bottom:
            page = w.new_page()
            y = _CONTENT_TOP
        where = hit_pages.get((hit.q_start, hit.q_end))
        location = f"{where} ページ · " if where else ""
        w.text(page, _CONTENT_X, y, f"{location}類似 {hit.score:.2f}", 8, bold=True)
        y += 13
        for line in lines:
            w.text(page, _CONTENT_X, y, line, 8.5)
            y += 13
        title = w.ellipsize(hit.doc.title or hit.doc.ext_id, 7.5, right - _CONTENT_X - 60)
        w.text(page, _CONTENT_X, y, f"{title} p.{hit.doc.text.count(chr(12), 0, hit.s_start) + 1}", 7.5, _MUTED)
        y += 6
        w.rule(page, y)
        y += 12


def _triage_appendix(w: _Writer, report: Report, locator: _Locator, first_manuscript_page: int):
    if not report.triage:
        return
    page = w.new_page("一致箇所の仕分け")
    right = page.rect.width - _CONTENT_X
    bottom = page.rect.height - _CONTENT_BOTTOM
    y = _CONTENT_TOP + 8
    w.text(page, _CONTENT_X, y, "Geminiによる一致箇所の仕分け", 11, bold=True)
    y += 15
    y = w.paragraph(
        page, _CONTENT_X, y,
        "確認の順番を決めるための補助で、精度は評価していません。類似度の値には影響しません。"
        "「定型句」とされた箇所にも問題がないとは限らないので、最終的な判断はご自身で行ってください。",
        right - _CONTENT_X, 8, _MUTED,
    )
    y += 18
    for item in sorted(report.triage, key=lambda t: (_TRIAGE_ORDER[t.label], t.start)):
        text = report.text[item.start:item.end]
        if len(text) > _TRIAGE_MAX_CHARS:
            text = text[:_TRIAGE_MAX_CHARS] + "…"
        lines = w.wrap(text, 8.5, right - _CONTENT_X)
        reason = w.wrap(item.reason, 7.5, right - _CONTENT_X) if item.reason else []
        if y + 34 + len(lines) * 13 + len(reason) * 11 > bottom:
            page = w.new_page()
            y = _CONTENT_TOP + 8
        label = LABELS.get(item.label, "未分類")
        w.pill(page, _CONTENT_X, y - 9, label, _rgb(_TRIAGE_COLORS[item.label]), height=13, size=7.5)
        rects = locator.rects(item.start, item.end)
        location = f"{first_manuscript_page + rects[0][0] + 1} ページ · " if rects else ""
        w.text(page, _CONTENT_X + 56, y, f"{location}出典 {item.rank + 1}", 7.5, _MUTED)
        y += 15
        for line in lines:
            w.text(page, _CONTENT_X, y, line, 8.5)
            y += 13
        for line in reason:
            w.text(page, _CONTENT_X, y, line, 7.5, _MUTED)
            y += 11
        y -= 5
        w.rule(page, y)
        y += 18


def _stamp_headers(w: _Writer, check_id: str):
    total = w.doc.page_count
    label = f"確認ID  {check_id}"
    section = ""
    starts = dict(w.sections)
    for page in w.doc:
        section = starts.get(page.number, section)
        for y in (20, page.rect.height - 14):
            w.text(page, _CONTENT_X, y, "Binoculars", 8, bold=True)
            w.text(page, _CONTENT_X + 52, y, f"{page.number + 1}/{total} ページ - {section}", 7)
            w.text(page, page.rect.width - _CONTENT_X - w.width(label, 7), y, label, 7)


def render_pdf(report: Report, path: str | Path) -> bytes:
    path = Path(path)
    check_id = sha256_file(path)[:16]
    base, notices = _open_base(path, report.text)
    locator = _Locator(report.text, base)
    if locator.mapped_share < _MIN_MAPPED:
        notices.append(
            f"原稿ページ上の位置を特定できた語は {locator.mapped_share * 100:.0f}% で、"
            "一部の一致はハイライトされていません。類似度の値には影響しません。"
        )

    doc = pymupdf.open()
    writer = _Writer(doc, base[0].rect if base.page_count else pymupdf.paper_rect("a4"))
    _cover(writer, report, path, check_id, base.page_count, [*report.notices, *notices])
    _overview(writer, report)
    source_pages = _source_list(writer, report)
    first_manuscript_page = doc.page_count
    hit_pages = _manuscript(writer, report, base, locator, source_pages)
    _semantic_appendix(writer, report, hit_pages)
    _triage_appendix(writer, report, locator, first_manuscript_page)
    _stamp_headers(writer, check_id)
    doc.set_metadata({"title": f"類似度レポート: {report.file_name}", "creator": "Binoculars"})
    doc.subset_fonts()
    return doc.tobytes(garbage=4, deflate=True)


def write_pdf(report: Report, source_path: str | Path, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_pdf(report, source_path))
    return out
