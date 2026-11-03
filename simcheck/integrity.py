import csv
import os
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx

from . import DATA_DIR
from .flags import Flag
from .sections import spans_of, strip_references
from .textnorm import tokenize

INTEGRITY_DIR = DATA_DIR.parent / "integrity"
TORTURED_PATH = INTEGRITY_DIR / "tortured" / "tortured.csv"
RETRACTION_PATH = INTEGRITY_DIR / "retraction_watch" / "retraction_watch.csv"
TORTURED_URL = "https://dbrech.irit.fr/pls/apex/f?p=9999:5::IR%5Btortured%5D_CSV::::"
RETRACTION_URL = "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
TORTURED_QUERY_COLUMN = "Fingerprint - Tortured Phrase"
TORTURED_EXPECTED_COLUMN = "Expected Text"

# 登録済みの文献では1語句の一致は大半が正当な用法で、2語句以上の一致は0件だった（evaluation/integrity/results.md）
MIN_TORTURED_PHRASES = 2
# 短い題名は別の論文と偶然一致するので、題名では照合しない
MIN_TITLE_TOKENS = 6
# 省略や誤りを許すのは、偶然の一致が起きにくい長さの題名だけにする
FUZZY_MIN_TOKENS = 8
MIN_PREFIX_SHARE = 0.6
TITLE_KEY_TOKENS = 5
# 題名の直後にDOIがあれば、同じ題名の別論文かどうかを確かめられる
DOI_AFTER_TITLE_CHARS = 200
MAX_EXAMPLES = 20

_QUERY_RE = re.compile(r'["“”]([^"“”]*)["“”](?:~(\d+))?|([()])|([^\s()"“”]+)')
# 文や見出しをまたいだ語の並びは語句として数えない（"sample preparation. Data were" など）
_PHRASE_BREAK = re.compile(r"[.,;:!?]|\n\s*\n|\n\s*[A-Z]")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
_DOI_WRAPPED = re.compile(r"(10\.\d{4,9}/(?:[^\s\"<>]*[./\-_(])?)\s*\n\s*(?=\S)")
_PMID_RE = re.compile(r"\b(?:PMID|PubMed(?:\s+ID)?)\s*:?\s*(\d{4,9})\b", re.IGNORECASE)
_NO_ID = {"", "0", "unavailable"}


class _Body:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.positions: dict[str, list[int]] = defaultdict(list)
        for i, token in enumerate(self.tokens.tokens):
            self.positions[token].append(i)

    def _joined(self, i: int) -> bool:
        t = self.tokens
        return not _PHRASE_BREAK.search(self.text, t.ends[i], t.starts[i + 1] + 1)

    def find(self, term: "_Term") -> list[int]:
        if term.prefix:
            return [i for token, found in self.positions.items() if token.startswith(term.words[0]) for i in found]
        if any(word not in self.positions for word in term.words):
            return []
        if term.slop is not None:
            return self._near(term)
        tokens = self.tokens.tokens
        n = len(term.words)
        return [
            i for i in self.positions[term.words[0]]
            if tuple(tokens[i:i + n]) == term.words and all(self._joined(j) for j in range(i, i + n - 1))
        ]

    def _near(self, term: "_Term") -> list[int]:
        span = len(term.words) + term.slop
        found = []
        for i in self.positions[term.words[0]]:
            for word in term.words[1:]:
                places = self.positions[word]
                k = bisect_right(places, i + span - 1) - 1
                if k < 0 or places[k] < i - span + 1:
                    break
            else:
                found.append(i)
        return found


@dataclass(frozen=True, slots=True)
class _Term:
    words: tuple[str, ...]
    slop: int | None = None
    prefix: bool = False


def _parse_query(query: str):
    """PPSの検索式を (評価する関数, 先頭の語句) にする。先頭の語句が式の成立に必須でない式は扱わない。"""
    items = []
    for m in _QUERY_RE.finditer(query):
        phrase, slop, paren, word = m.groups()
        if paren:
            items.append(paren)
        elif phrase is not None:
            items.append(_make_term(phrase, int(slop) if slop else None))
        elif word.upper() in ("AND", "OR", "NOT"):
            items.append(word.upper())
        else:
            items.append(_make_term(word.rstrip("*"), None, word.endswith("*")))
    pos = 0

    def peek():
        return items[pos] if pos < len(items) else None

    def parse_or():
        nonlocal pos
        parts = [parse_and()]
        while peek() == "OR":
            pos += 1
            parts.append(parse_and())
        return parts[0] if len(parts) == 1 else lambda body: any(p(body) for p in parts)

    def parse_and():
        nonlocal pos
        required, forbidden = [parse_atom()], []
        while True:
            item = peek()
            if item == "AND":
                pos += 1
                required.append(parse_atom())
            elif item == "NOT":
                pos += 1
                forbidden.append(parse_atom())
            elif isinstance(item, _Term) or item == "(":
                required.append(parse_atom())
            else:
                break
        return lambda body: all(p(body) for p in required) and not any(p(body) for p in forbidden)

    def parse_atom():
        nonlocal pos
        item = peek()
        pos += 1
        if item == "(":
            inner = parse_or()
            if peek() != ")":
                raise ValueError(query)
            pos += 1
            return inner
        if isinstance(item, _Term):
            return lambda body: bool(body.find(item))
        raise ValueError(query)

    if not items or not isinstance(items[0], _Term) or items[0].prefix:
        raise ValueError(query)
    match = parse_or()
    if pos != len(items) or _has_top_level_or(items):
        raise ValueError(query)
    return match, items[0]


def _has_top_level_or(items) -> bool:
    depth = 0
    for item in items:
        depth += (item == "(") - (item == ")")
        if item == "OR" and depth == 0:
            return True
    return False


def _make_term(phrase: str, slop: int | None = None, prefix: bool = False) -> _Term:
    words = tuple(tokenize(phrase).tokens)
    if not words:
        raise ValueError(phrase)
    return _Term(words, slop, prefix)


@dataclass(slots=True)
class _Fingerprint:
    query: str
    expected: str
    match: object
    term: _Term


@dataclass(slots=True)
class TorturedHit:
    phrase: str
    expected: str
    count: int


@dataclass(slots=True)
class TorturedList:
    by_first_word: dict[str, list[_Fingerprint]]
    n_loaded: int
    n_skipped: int


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def parse_tortured(path: Path) -> TorturedList:
    by_first_word = defaultdict(list)
    n_loaded = n_skipped = 0
    for row in _read_csv(path):
        query = row[TORTURED_QUERY_COLUMN]
        try:
            match, term = _parse_query(query)
        except ValueError:
            n_skipped += 1
            continue
        by_first_word[term.words[0]].append(_Fingerprint(query, row[TORTURED_EXPECTED_COLUMN], match, term))
        n_loaded += 1
    return TorturedList(dict(by_first_word), n_loaded, n_skipped)


def find_tortured(text: str, fingerprints: TorturedList) -> list[TorturedHit]:
    body = _Body(strip_references(text))
    hits = {}
    for word in body.positions:
        for fp in fingerprints.by_first_word.get(word, ()):
            if fp.match(body):
                phrase = " ".join(fp.term.words)
                hits.setdefault(phrase, TorturedHit(phrase, fp.expected, len(body.find(fp.term))))
    return sorted(hits.values(), key=lambda h: (-h.count, h.phrase))


@dataclass(slots=True)
class RetractionEntry:
    title: str
    doi: str
    pmid: str
    natures: set[str] = field(default_factory=set)
    date: str = ""

    @property
    def status(self) -> str | None:
        # 撤回の取り消しが記録された論文は撤回扱いにしない。訂正だけの論文も対象外
        if "Reinstatement" in self.natures:
            return None
        if "Retraction" in self.natures:
            return "撤回"
        if "Expression of concern" in self.natures:
            return "懸念表明"
        return None


@dataclass(slots=True)
class RetractionData:
    by_doi: dict[str, RetractionEntry]
    by_pmid: dict[str, RetractionEntry]
    by_title_key: dict[tuple[str, ...], list[tuple[tuple[str, ...], RetractionEntry, int]]]


def _clean_id(value: str) -> str:
    value = value.strip().lower()
    return "" if value in _NO_ID else value


def _format_date(value: str) -> str:
    try:
        return datetime.strptime(value.split()[0], "%m/%d/%Y").strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return value


def parse_retractions(path: Path) -> RetractionData:
    papers: dict[str, RetractionEntry] = {}
    for row in _read_csv(path):
        doi = _clean_id(row["OriginalPaperDOI"])
        pmid = _clean_id(row["OriginalPaperPubMedID"])
        title = row["Title"].strip()
        key = doi or (f"pmid:{pmid}" if pmid else f"title:{title.lower()}")
        entry = papers.setdefault(key, RetractionEntry(title, doi, pmid))
        entry.natures.add(row["RetractionNature"])
        if row["RetractionNature"] == "Retraction" or not entry.date:
            entry.date = _format_date(row["RetractionDate"])

    by_doi, by_pmid, by_title_key = {}, {}, defaultdict(list)
    for entry in papers.values():
        if entry.status is None:
            continue
        if entry.doi:
            by_doi[entry.doi] = entry
        if entry.pmid:
            by_pmid[entry.pmid] = entry
        words = tuple(tokenize(entry.title).tokens)
        if len(words) >= MIN_TITLE_TOKENS:
            by_title_key[words[:TITLE_KEY_TOKENS]].append((words, entry, 0))
            # 題名の前半に誤りがあっても見つかるよう、末尾の語でも引けるようにする
            if len(words) >= FUZZY_MIN_TOKENS:
                by_title_key[words[-TITLE_KEY_TOKENS:]].append((words, entry, len(words) - TITLE_KEY_TOKENS))
    return RetractionData(by_doi, by_pmid, dict(by_title_key))


def _dois_in(text: str) -> set[str]:
    # 行末で折り返されたDOIはつないだ形も試す。誤ってつないでも照合先にないだけで害はない
    dois = set()
    for source in (text, _DOI_WRAPPED.sub(r"\1", text)):
        dois.update(m.group().rstrip(".,;)]>").lower() for m in _DOI_RE.finditer(source))
    return dois


def _references_of(text: str) -> str | None:
    spans = spans_of(text, "references")
    return "\n".join(text[s:e] for s, e in spans) if spans else None


def _title_match_length(tokens: list[str], start: int, words: tuple[str, ...]) -> int:
    """参考文献の start からの語が題名と一致すれば、一致した語数を返す。省略と少しの誤りは許す。"""
    n = len(words)
    window = tokens[start:start + n]
    if tuple(window) == words:
        return n
    if n < FUZZY_MIN_TOKENS or len(window) < FUZZY_MIN_TOKENS:
        return 0
    if len(window) == n and sum(a != b for a, b in zip(window, words)) <= max(1, n // 10):
        return n
    prefix = 0
    for a, b in zip(window, words):
        if a != b:
            break
        prefix += 1
    return prefix if prefix >= FUZZY_MIN_TOKENS and prefix / n >= MIN_PREFIX_SHARE else 0


def find_retracted(text: str, data: RetractionData) -> tuple[list[RetractionEntry], list[RetractionEntry], bool]:
    """(DOIかPMIDで一致した論文, 題名だけで一致した論文, 参考文献の範囲を判定できたか) を返す。"""
    references = _references_of(text)
    found = references is not None
    # 範囲が分からないときはDOIとPMIDだけ全文から探す。題名は本文の語と偶然一致しうるので照合しない
    scope = references if found else text

    by_id = {}
    for doi in _dois_in(scope):
        if doi in data.by_doi:
            by_id[id(data.by_doi[doi])] = data.by_doi[doi]
    for m in _PMID_RE.finditer(scope):
        if m.group(1) in data.by_pmid:
            entry = data.by_pmid[m.group(1)]
            by_id[id(entry)] = entry

    by_title = {}
    if found:
        t = tokenize(scope)
        tokens = t.tokens
        for i in range(len(tokens) - TITLE_KEY_TOKENS + 1):
            for words, entry, offset in data.by_title_key.get(tuple(tokens[i:i + TITLE_KEY_TOKENS]), ()):
                start = i - offset
                matched = _title_match_length(tokens, start, words) if start >= 0 else 0
                if not matched or id(entry) in by_id:
                    continue
                end = start + matched
                after = scope[t.ends[end - 1]:t.ends[end - 1] + DOI_AFTER_TITLE_CHARS]
                # 直後の次の文献のDOIを拾わないよう、参考文献1件ぶんの範囲に限る
                nearby = _DOI_RE.search(after.split("\n\n")[0])
                if nearby and nearby.group().rstrip(".,;)]>").lower() != entry.doi:
                    continue
                by_title[id(entry)] = entry
    order = lambda e: e.title.lower()
    return sorted(by_id.values(), key=order), sorted(by_title.values(), key=order), found


@lru_cache(maxsize=1)
def _cached_tortured(path: Path, mtime: float) -> TorturedList:
    return parse_tortured(path)


@lru_cache(maxsize=1)
def _cached_retractions(path: Path, mtime: float) -> RetractionData:
    return parse_retractions(path)


def _fetched_on(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _entry_example(entry: RetractionEntry) -> str:
    identifier = f"DOI {entry.doi}" if entry.doi else f"PMID {entry.pmid}" if entry.pmid else "IDなし"
    return f"{entry.title[:120]}（{entry.status} {entry.date}、{identifier}）"


def _tortured_results(text: str, fingerprints: TorturedList) -> tuple[list[Flag], list[str]]:
    hits = find_tortured(text, fingerprints)
    examples = [f"{hit.phrase} → {hit.expected}（{hit.count}回）" for hit in hits[:MAX_EXAMPLES]]
    if not hits:
        return [], []
    if len(hits) < MIN_TORTURED_PHRASES:
        return [], [
            f"不自然な語句のリストに1語句だけ一致しました: {examples[0]}。"
            "正当な用法でも一致するので、文脈を確認してください。"
        ]
    return [Flag(
        "不自然な語句",
        f"言い換えツールの痕跡として知られる語句に {len(hits)} 種類一致しました（Problematic Paper Screener）。"
        "正当な用語や非英語圏の著者の表現でも一致することがあります。",
        sum(hit.count for hit in hits),
        examples,
    )], []


def _retraction_flags(entries: list[RetractionEntry], by_title: bool) -> list[Flag]:
    flags = []
    for status in ("撤回", "懸念表明"):
        matched = [e for e in entries if e.status == status]
        if not matched:
            continue
        kind = f"{status}論文の引用" + ("（題名の一致）" if by_title else "")
        message = f"Retraction Watchで{status}と記録された論文が参考文献にあります。"
        message += (
            "題名だけの一致なので、同じ題名の別の論文の可能性があります。" if by_title
            else "撤回や懸念を論じるための引用なら問題ありません。"
        )
        flags.append(Flag(kind, message, len(matched), [_entry_example(e) for e in matched[:MAX_EXAMPLES]]))
    return flags


def check_integrity(text: str) -> tuple[list[Flag], list[str]]:
    flags, notices = [], []
    if TORTURED_PATH.exists():
        fingerprints = _cached_tortured(TORTURED_PATH, TORTURED_PATH.stat().st_mtime)
        tortured_flags, tortured_notices = _tortured_results(text, fingerprints)
        flags += tortured_flags
        notices += tortured_notices
    else:
        notices.append("語句リストが未取得のため、不自然な語句の確認は行っていません（update-integrity で取得します）。")
    if RETRACTION_PATH.exists():
        data = _cached_retractions(RETRACTION_PATH, RETRACTION_PATH.stat().st_mtime)
        by_id, by_title, found = find_retracted(text, data)
        flags += _retraction_flags(by_id, False) + _retraction_flags(by_title, True)
        notices.append(
            f"撤回の確認にはRetraction Watch（{_fetched_on(RETRACTION_PATH)} 取得）を使いました。"
            "省略された題名や、DOIとPMIDのない文献は照合できないことがあります。"
        )
        if not found:
            notices.append("参考文献の範囲を判定できなかったため、撤回の確認はDOIとPMIDのみで行いました。")
    else:
        notices.append("撤回データが未取得のため、撤回論文の確認は行っていません（update-integrity で取得します）。")
    return flags, notices


def _download(url: str, path: Path, required_column: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
    # エラーページで既存のデータを上書きしないよう、列名を確かめてから置き換える
    with open(tmp, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    if required_column not in header:
        tmp.unlink()
        raise IOError(f"想定と異なる形式です: {url}")
    os.replace(tmp, path)


def update_integrity() -> dict:
    _download(TORTURED_URL, TORTURED_PATH, TORTURED_QUERY_COLUMN)
    _download(RETRACTION_URL, RETRACTION_PATH, "OriginalPaperDOI")
    fingerprints = parse_tortured(TORTURED_PATH)
    retractions = parse_retractions(RETRACTION_PATH)
    return {
        "tortured": fingerprints.n_loaded,
        "tortured_skipped": fingerprints.n_skipped,
        "retraction_dois": len(retractions.by_doi),
    }
