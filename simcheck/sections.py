import re



def _ja(*words: str) -> str:
    # 日本語の見出しは「文　献」のように字間を空けることが多い
    return "|".join(r"[ \t　]*".join(word) for word in words)


_HEADINGS = {
    "abstract": r"abstract|summary|" + _ja("要旨", "抄録", "要約"),
    "introduction": r"introduction|background|" + _ja("緒言", "はじめに", "背景"),
    "methods": (
        r"(?:(?:materials?|patients?|subjects?|participants?)\s*(?:and|&)\s*)?methods?"
        r"|methodology|experimental\s+procedures|study\s+design|"
        + _ja("方法", "対象と方法", "対象および方法", "対象及び方法", "材料と方法", "材料および方法")
    ),
    "results": r"results(?:\s+and\s+discussion)?|" + _ja("結果"),
    "discussion": r"discussion|" + _ja("考察", "考按"),
    "conclusion": r"conclusions?|" + _ja("結論", "結語", "まとめ"),
    "acknowledgments": r"acknowledge?ments?|" + _ja("謝辞"),
    "references": r"references|bibliography|literature\s+cited|" + _ja("参考文献", "引用文献", "文献"),
    "other": (
        r"figure\s+legends?|figure\s+captions?|legends?|tables?"
        r"|supplementary\s+(?:materials?|information|data)"
        r"|funding|conflicts?\s+of\s+interest|disclosures?|" + _ja("図の説明", "図表")
    ),
}

_HEADING_RE = re.compile(
    r"^[ \t　]*(?:\d+(?:\.\d+)*\.?[ \t]+|[ivx]+\.[ \t]+)?(?:"
    + "|".join(f"(?P<{name}>{pattern})" for name, pattern in _HEADINGS.items())
    + r")[ \t　]*[:：.]?[ \t　]*$",
    re.IGNORECASE | re.MULTILINE,
)

# 参考文献の後ろに続く図表（投稿原稿でよくある構成）を巻き込まないための目印
_CAPTION_RE = re.compile(
    r"^[ \t　]*(?:(?:supplementary\s+|supplemental\s+)?(?:table|figure|fig\.?)|eTable|eFigure|表|図)"
    r"[ \t　]*S?\d+",
    re.IGNORECASE | re.MULTILINE,
)

_QUOTE_RE = re.compile(
    r"“[^”]{20,2000}”|\"[^\"\n]{20,2000}\"|「[^」]{10,2000}」|『[^』]{10,2000}』"
)


def detect_sections(text: str) -> list[tuple[str, int, int]]:
    """(見出しの種類, 開始, 終了) を返す。短い文書を除き、本文前半の「文献」は見出しと見なさない。"""
    found = []
    for m in _HEADING_RE.finditer(text):
        name = m.lastgroup
        if name == "references" and len(text) > 2000 and m.start() < len(text) * 0.3:
            continue
        found.append((name, m.start()))
    sections = []
    for i, (name, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        if name == "references":
            caption = _CAPTION_RE.search(text, start, end)
            if caption:
                end = caption.start()
        sections.append((name, start, end))
    return sections


def spans_of(text: str, kind: str) -> list[tuple[int, int]]:
    return [(s, e) for name, s, e in detect_sections(text) if name == kind]


def quote_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _QUOTE_RE.finditer(text)]


def strip_references(text: str) -> str:
    # 照合先の参考文献まで索引すると、書誌情報どうしの一致がノイズになる
    kept, pos = [], 0
    for s, e in spans_of(text, "references"):
        kept.append(text[pos:s])
        pos = e
    kept.append(text[pos:])
    return "".join(kept)
