import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from hashlib import blake2b

# 重みは「英単語1語 = 5」。日本語は1文字 = 2 なので、英語5語ぶんのシングルは日本語13文字に相当する
WORD_WEIGHT = 5
CJK_WEIGHT = 2
SHINGLE_WEIGHT = 5 * WORD_WEIGHT
# 連続する一致が 5 + 6 - 1 = 10 語以上あれば、必ず1つは共通の指紋が選ばれる
WINNOW_WINDOW = 6

# 剽窃チェック回避に使われる、ラテン文字と同形のキリル文字
# アルメニア文字も医学論文の本文には現れないので、位置を問わず戻す
CONFUSABLES = str.maketrans("аеорсхуіјѕһԁօսոհցա", "aeopcxyijshdounhgw")
# ギリシャ文字は TGFα のように本物の用法が多いので、語の途中でラテン小文字に挟まれたときだけ同形文字とみなす
CONTEXT_CONFUSABLE_PAIRS = {
    "α": "a", "ε": "e", "ο": "o", "ι": "i", "υ": "u", "ν": "v", "ρ": "p", "κ": "k", "τ": "t", "χ": "x",
    "ɑ": "a", "ɡ": "g", "ı": "i", "ȷ": "j",
}
CONTEXT_CONFUSABLES = str.maketrans(CONTEXT_CONFUSABLE_PAIRS)
_CONFUSABLE_LETTERS = "".join(CONTEXT_CONFUSABLE_PAIRS)
# 語の途中でラテン小文字に挟まれた文字か、語頭の文字のあとがラテン小文字だけの語（ιn、οf）を対象にする
HOMOGLYPH_WORD = re.compile(
    rf"(?<![A-Za-z{_CONFUSABLE_LETTERS}])(?:"
    rf"[A-Za-z{_CONFUSABLE_LETTERS}]*[a-z{_CONFUSABLE_LETTERS}][{_CONFUSABLE_LETTERS}][a-z{_CONFUSABLE_LETTERS}]"
    rf"[A-Za-z{_CONFUSABLE_LETTERS}]*"
    rf"|[{_CONFUSABLE_LETTERS}][a-z]+(?![A-Za-z0-9\u0370-\u03ff])"
    r")"
)
# 表示されないのに語を分断する文字
INVISIBLE_CHARS = frozenset(
    "­͏؜ᅟᅠ឴឵᠋᠌᠍᠎᠏"
    "​‌‍‎‏‪‫‬‭‮"
    "⁠⁡⁢⁣⁤⁦⁧⁨⁩⁪⁫⁬⁭⁮⁯"
    "ㅤ﻿ﾠ￹￺￻"
) | frozenset(map(chr, range(0xFE00, 0xFE10))) | frozenset(map(chr, range(0xE0000, 0xE0080))) \
  | frozenset(map(chr, range(0xE0100, 0xE01F0)))
DROP_CHARS = INVISIBLE_CHARS
LINE_END_HYPHENS = frozenset("-‐‑")


def _has_latin(word: str) -> bool:
    return any("a" <= c.lower() <= "z" for c in word)


def fold_homoglyph_words(raw: str) -> str:
    """同形文字の語をラテン文字に戻す。1文字を1文字に置き換えるので位置は変わらない。"""
    return HOMOGLYPH_WORD.sub(
        lambda m: m.group().translate(CONTEXT_CONFUSABLES) if _has_latin(m.group()) else m.group(), raw
    )

TOKEN_RE = re.compile(
    r"[0-9]+\.[0-9]+"
    r"|[0-9a-zß-ɏͰ-ϿЀ-ӿ]+"
    r"|[々぀-ヿ㐀-鿿豈-﫿]"
)

_char_cache: dict[str, str] = {}


def normalize(raw: str) -> tuple[str, list[int] | None]:
    """正規化した文字列と、各文字の元の位置を返す。ASCIIのみなら位置は変わらないので None。"""
    if raw.isascii() and "-\n" not in raw:
        return raw.lower(), None

    raw = fold_homoglyph_words(raw)
    out, offsets = [], []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch in LINE_END_HYPHENS:
            # 行末ハイフンで分割された単語をつなぐ（"ophthal-\nmology" → "ophthalmology"）
            j = i + 1
            while j < n and raw[j] in " \t\r":
                j += 1
            if j + 1 < n and raw[j] == "\n" and raw[j + 1].islower():
                i = j + 1
                continue
        if ch not in DROP_CHARS:
            mapped = _char_cache.get(ch)
            if mapped is None:
                mapped = unicodedata.normalize("NFKC", ch).lower().translate(CONFUSABLES)
                _char_cache[ch] = mapped
            for m in mapped:
                out.append(m)
                offsets.append(i)
        i += 1
    return "".join(out), offsets


@dataclass(slots=True)
class Tokens:
    tokens: list[str]
    starts: list[int]
    ends: list[int]
    weights: list[int]

    def __len__(self):
        return len(self.tokens)

    def subset(self, indices: list[int]) -> "Tokens":
        return Tokens(
            [self.tokens[i] for i in indices],
            [self.starts[i] for i in indices],
            [self.ends[i] for i in indices],
            [self.weights[i] for i in indices],
        )


def tokenize(raw: str) -> Tokens:
    norm, offsets = normalize(raw)
    tokens, starts, ends, weights = [], [], [], []
    for m in TOKEN_RE.finditer(norm):
        s, e = m.span()
        tok = m.group()
        tokens.append(tok)
        if offsets is None:
            starts.append(s)
            ends.append(e)
        else:
            starts.append(offsets[s])
            ends.append(offsets[e - 1] + 1)
        weights.append(CJK_WEIGHT if len(tok) == 1 and tok >= "々" else WORD_WEIGHT)
    return Tokens(tokens, starts, ends, weights)


def shingle_hashes(t: Tokens) -> list[tuple[int, int]]:
    """各開始位置の (ハッシュ, 開始トークン位置) を返す。長さは重みで決まるので言語に依存しない。"""
    result = []
    tokens, weights = t.tokens, t.weights
    n = len(tokens)
    j = acc = 0
    for i in range(n):
        while j < n and acc < SHINGLE_WEIGHT:
            acc += weights[j]
            j += 1
        if acc < SHINGLE_WEIGHT:
            break
        digest = blake2b("\x1f".join(tokens[i:j]).encode("utf-8", "replace"), digest_size=8).digest()
        result.append((int.from_bytes(digest, "little"), i))
        acc -= weights[i]
    return result


def winnow(hashes: list[int], window: int = WINNOW_WINDOW) -> set[int]:
    if len(hashes) < window:
        return set(hashes)
    selected = set()
    dq: deque[int] = deque()
    for i, h in enumerate(hashes):
        while dq and hashes[dq[-1]] >= h:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - window:
            dq.popleft()
        if i >= window - 1:
            selected.add(hashes[dq[0]])
    return selected


def fingerprints(raw: str) -> set[int]:
    return winnow([h for h, _ in shingle_hashes(tokenize(raw))])
