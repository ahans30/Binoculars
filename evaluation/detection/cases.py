"""検出率の評価に使う標本と合成原稿を作り、チェックの結果を正解と照らす。"""
import hashlib
import json
import random
import re
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simcheck import DATA_DIR
from simcheck.sections import spans_of
from simcheck.textnorm import tokenize

DATA = DATA_DIR.parent / "detection_eval"
AI_FILLER = DATA_DIR.parent / "ai_threshold" / "ai_local.jsonl"

BASE_WORDS = 120
MAX_SENTENCE_WORDS = 80
JA_BASE_CHARS = 400
N_PASSAGES = {"pubmed": 200, "library": 200, "library_ja": 100}
N_HUMAN_DOCS = 20
ABSTRACTS_PER_HUMAN_DOC = 8
FILLERS_PER_DOC = 6
PASSAGES_PER_DOC = 5
DETECTED_COVERAGE = 0.5
MAX_PER_JA_DOC = 3

_LINE_END_HYPHEN = re.compile(r"(?<=[a-z])[-‐‑]\s*\n\s*(?=[a-z])")
_SENTENCE_START = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_JA_CHAR = re.compile(r"[぀-鿿]")
_HIRAGANA = re.compile(r"[ぁ-ん]")
_BIBLIOGRAPHY = re.compile(r"\bet al\b.*\bet al\b|\b(?:19|20)\d{2}\s*[;:]\s*\d+|\bdoi\b|\b\d+\s*:\s*\d+\s*[-–]\s*\d+", re.I | re.S)
VOCABULARY = (
    "analysis clinical outcome patient method result study treatment finding measure group effect sample "
    "evidence model review factor response change value"
).split()


def split_of(ext_id: str) -> str:
    return "dev" if hashlib.sha256(ext_id.encode()).digest()[0] % 2 == 0 else "test"


def stable_rng(*parts) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def clean_passage(raw: str) -> str:
    return " ".join(_LINE_END_HYPHEN.sub("", raw).split())


def _mostly_prose(text: str) -> bool:
    if not text or sum(c.isascii() for c in text) / len(text) < 0.97:
        return False
    # 文献一覧の断片を本文として数えない
    if _BIBLIOGRAPHY.search(text):
        return False
    return sum(c.isalpha() or c.isspace() for c in text) / len(text) >= 0.85


def english_passage(doc_text: str, rng: random.Random) -> str | None:
    """文の先頭から始まる、BASE_WORDS語以上の連続した文章を取り出す。"""
    starts = [m.end() for m in _SENTENCE_START.finditer(doc_text)]
    rng.shuffle(starts)
    for start in starts[:30]:
        # 1語が平均10文字としても足りる範囲だけを切り出す
        sentences = _SENTENCE_START.split(clean_passage(doc_text[start:start + BASE_WORDS * 25]))
        kept, words = [], 0
        for sentence in sentences:
            n = len(sentence.split())
            if n > MAX_SENTENCE_WORDS:
                break
            kept.append(sentence)
            words += n
            if words >= BASE_WORDS:
                break
        text = " ".join(kept)
        if words >= BASE_WORDS and _mostly_prose(text) and not spans_of(text, "references"):
            return text
    return None


def japanese_passage(doc_text: str, rng: random.Random) -> str | None:
    starts = [m.end() for m in re.finditer("。", doc_text)]
    rng.shuffle(starts)
    for start in starts[:30]:
        chunk = "".join(doc_text[start:start + JA_BASE_CHARS * 2].split())
        end = chunk.find("。", JA_BASE_CHARS)
        text = chunk[:end + 1] if end >= 0 else ""
        # 表紙や図のOCRの断片を避けるため、かなが十分にある文章だけを使う
        if text and len(_JA_CHAR.findall(text)) / len(text) >= 0.8 and len(_HIRAGANA.findall(text)) / len(text) >= 0.3:
            return text
    return None


def _load_fillers(split: str) -> list[str]:
    fillers = []
    with open(AI_FILLER, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["mode"] in ("new", "long") and split_of(row["source_id"]) == split:
                # 参考文献の見出しがあると、その後ろが除外範囲になって地の文として測れない
                if not spans_of(row["text"], "references"):
                    fillers.append(row["text"].strip())
    return fillers


def prepare_cases(store, split: str) -> dict:
    """標本を作って保存する。照合先が増えても比較できるよう、一度作ったら作り直さない。"""
    path = DATA / f"cases_{split}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    rows = store.conn.execute("SELECT doc_id, ext_id, source FROM docs").fetchall()
    ids = {
        source: sorted(doc_id for doc_id, ext_id, s in rows if s == source and split_of(ext_id) == split)
        for source in ("pubmed", "library")
    }
    rng = stable_rng("cases", split)
    passages: dict[str, list[dict]] = {}

    pubmed_order = rng.sample(ids["pubmed"], len(ids["pubmed"]))
    passages["pubmed"], used = [], set()
    for i in range(0, len(pubmed_order), 500):
        docs = store.get_docs(pubmed_order[i:i + 500])
        for doc in docs.values():
            text = english_passage(doc.text, rng)
            if text:
                passages["pubmed"].append({"ext_id": doc.ext_id, "text": text})
                used.add(doc.ext_id)
        if len(passages["pubmed"]) >= N_PASSAGES["pubmed"]:
            passages["pubmed"] = passages["pubmed"][:N_PASSAGES["pubmed"]]
            break

    passages["library"], passages["library_ja"] = [], []
    library = list(store.get_docs(ids["library"]).values())
    rng.shuffle(library)
    for doc in library:
        ja_share = len(_JA_CHAR.findall(doc.text)) / max(len(doc.text), 1)
        if ja_share > 0.3:
            for _ in range(MAX_PER_JA_DOC):
                text = japanese_passage(doc.text, rng)
                if text and len(passages["library_ja"]) < N_PASSAGES["library_ja"]:
                    passages["library_ja"].append({"ext_id": doc.ext_id, "text": text})
        elif len(passages["library"]) < N_PASSAGES["library"]:
            text = english_passage(doc.text, rng)
            if text:
                passages["library"].append({"ext_id": doc.ext_id, "text": text})

    human, human_needed = [], N_HUMAN_DOCS * ABSTRACTS_PER_HUMAN_DOC
    for doc_id in pubmed_order:
        if len(human) >= human_needed:
            break
        doc = store.get_docs([doc_id]).get(doc_id)
        if doc and doc.ext_id not in used and len(doc.text.split()) >= 80 and _mostly_prose(doc.text):
            human.append({"ext_id": doc.ext_id, "text": doc.text})

    cases = {
        "split": split, "passages": passages, "human": human, "fillers": _load_fillers(split),
        "index_stats": store.stats(),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    return cases


def _words_every(text: str, n: int) -> list[int]:
    return list(range(n - 1, len(text.split()), n))


def drop_every(n: int) -> Callable:
    def apply(text, rng):
        drop = set(_words_every(text, n))
        return " ".join(w for i, w in enumerate(text.split()) if i not in drop)
    return apply


def replace_every(n: int) -> Callable:
    def apply(text, rng):
        words = text.split()
        for i in _words_every(text, n):
            words[i] = rng.choice([v for v in VOCABULARY if v != words[i].lower()])
        return " ".join(words)
    return apply


def swap_every(n: int) -> Callable:
    def apply(text, rng):
        words = text.split()
        for i in _words_every(text, n):
            if i + 1 < len(words):
                words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)
    return apply


def shuffle_sentences(text: str, rng: random.Random) -> str:
    sentences = _SENTENCE_START.split(text)
    rng.shuffle(sentences)
    return " ".join(sentences)


def map_chars(table: dict[str, str]) -> Callable:
    translation = str.maketrans(table)
    return lambda text, rng: text.translate(translation)


def insert_inside_words(mark: str, every: int = 3) -> Callable:
    """語の途中に、every文字ごとに文字を挟む。"""
    def fill(m):
        word = m.group()
        return mark.join(word[i:i + every] for i in range(0, len(word), every))
    return lambda text, rng: re.sub(r"[A-Za-z]{4,}", fill, text)


def replace_spaces(space: str) -> Callable:
    return lambda text, rng: text.replace(" ", space)


_FULLWIDTH = {chr(c): chr(c - 0x41 + 0xFF21) for c in range(0x41, 0x5B)}
_FULLWIDTH.update({chr(c): chr(c - 0x61 + 0xFF41) for c in range(0x61, 0x7B)})
_MATH_BOLD = {chr(c): chr(c - 0x41 + 0x1D400) for c in range(0x41, 0x5B)}
_MATH_BOLD.update({chr(c): chr(c - 0x61 + 0x1D41A) for c in range(0x61, 0x7B)})


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    group: str
    strata: tuple[str, ...]
    words: int = BASE_WORDS
    transform: Callable | None = None
    llm_mode: str | None = None
    semantic: bool = False
    control: str | None = None


_BOTH = ("pubmed", "library")
CONDITIONS = [Condition(f"verbatim_{n}", "そのままの転用", _BOTH, n) for n in (8, 10, 12, 15, 20, 30, 60)]
CONDITIONS += [Condition(f"drop_every_{n}", "語の編集", _BOTH, 60, drop_every(n)) for n in (5, 8, 12)]
CONDITIONS += [Condition(f"replace_every_{n}", "語の編集", _BOTH, 60, replace_every(n)) for n in (5, 8, 12, 20)]
CONDITIONS += [Condition(f"swap_every_{n}", "語の編集", _BOTH, 60, swap_every(n)) for n in (5, 10)]
CONDITIONS += [Condition("shuffle_sentences", "語の編集", _BOTH, BASE_WORDS, shuffle_sentences)]
_CHAR_CONDITIONS = {
    "homoglyph_cyrillic": map_chars(dict(zip("aeopcxyi", "аеорсхуі"))),
    "homoglyph_greek": map_chars(dict(zip("aeoiuvp", "αεοιυνρ"))),
    "homoglyph_armenian": map_chars(dict(zip("ounhg", "օսոհց"))),
    "fullwidth": map_chars(_FULLWIDTH),
    "math_bold": map_chars(_MATH_BOLD),
    "zero_width": insert_inside_words("​"),
    "soft_hyphen": insert_inside_words("­"),
    "variation_selector": insert_inside_words("️"),
    "tag_character": insert_inside_words("\U000e0061"),
    "bidi_control": insert_inside_words("‭‬"),
    "mongolian_separator": insert_inside_words("᠎"),
    "combining_joiner": insert_inside_words("͏"),
    "special_spaces": replace_spaces(" "),
}
CONDITIONS += [Condition(n, "文字の加工", ("pubmed",), 60, t) for n, t in _CHAR_CONDITIONS.items()]
CONDITIONS += [Condition(f"llm_{m}", "LLMの書き換え", _BOTH, llm_mode=m, semantic=True) for m in ("light", "heavy")]
CONDITIONS += [Condition("llm_translate_en", "LLMの書き換え", ("library_ja",), llm_mode="translate_en", semantic=True)]
CONDITIONS += [
    Condition("control_filler", "対照", ("-",), semantic=True, control="filler"),
    Condition("control_human", "対照", ("-",), semantic=True, control="human"),
]
CONDITION_BY_NAME = {c.name: c for c in CONDITIONS}


def apply_condition(condition: Condition, passage_text: str, passage_key) -> str:
    words = passage_text.split()
    text = " ".join(words[:condition.words]) if len(words) > condition.words else passage_text
    if condition.transform:
        text = condition.transform(text, stable_rng(condition.name, passage_key))
    return text


@dataclass(slots=True)
class Case:
    text: str
    # (開始, 終了, 本来の出典のID)
    spans: list[tuple[int, int, str]]
    exclude_ext_ids: list[str]


def compose_case(fillers: list[str], inserted: list[tuple[str, str]]) -> Case:
    """地の文の間に文章を入れて、位置を記録する。"""
    parts, spans, pos = [], [], 0
    remaining = list(inserted)
    for filler in fillers:
        for block, ext_id in ([(filler, None)] + ([remaining.pop(0)] if remaining else [])):
            if parts:
                parts.append("\n\n")
                pos += 2
            parts.append(block)
            if ext_id is not None:
                spans.append((pos, pos + len(block), ext_id))
            pos += len(block)
    for block, ext_id in remaining:
        parts.append("\n\n")
        pos += 2
        parts.append(block)
        spans.append((pos, pos + len(block), ext_id))
        pos += len(block)
    return Case("".join(parts), spans, [])


def choose_fillers(fillers: list[str], doc_index: int) -> list[str]:
    return stable_rng("fillers", doc_index).sample(fillers, min(FILLERS_PER_DOC, len(fillers)))


def build_cases(condition: Condition, stratum: str, data: dict, paraphrases: dict | None = None) -> list[Case]:
    if condition.control == "filler":
        n_docs = N_PASSAGES["pubmed"] // PASSAGES_PER_DOC
        return [Case(*_plain(choose_fillers(data["fillers"], i)), []) for i in range(n_docs)]
    if condition.control == "human":
        human = data["human"]
        cases = []
        for i in range(0, len(human) - ABSTRACTS_PER_HUMAN_DOC + 1, ABSTRACTS_PER_HUMAN_DOC):
            chunk = human[i:i + ABSTRACTS_PER_HUMAN_DOC]
            cases.append(Case(*_plain([row["text"] for row in chunk]), [row["ext_id"] for row in chunk]))
        return cases

    inserted = []
    for index, passage in enumerate(data["passages"][stratum]):
        key = f"{stratum}:{index}"
        if condition.llm_mode:
            text = (paraphrases or {}).get((key, condition.llm_mode))
            if not text:
                continue
        else:
            text = apply_condition(condition, passage["text"], key)
        inserted.append((text, passage["ext_id"]))
    return [
        compose_case(choose_fillers(data["fillers"], i // PASSAGES_PER_DOC), inserted[i:i + PASSAGES_PER_DOC])
        for i in range(0, len(inserted), PASSAGES_PER_DOC)
    ]


def _plain(blocks: list[str]) -> tuple[str, list]:
    return "\n\n".join(blocks), []


def score_report(report, case: Case) -> dict:
    """1件の原稿の結果を、語の単位で正解と照らす。"""
    tokens = tokenize(report.text)
    starts = tokens.starts
    owned = sorted(report.owned_spans)
    owned_starts = [s for s, _, _ in owned]

    def owner_of(position: int) -> str | None:
        i = bisect_right(owned_starts, position) - 1
        if i >= 0 and position < owned[i][1]:
            return report.sources[owned[i][2]].doc.ext_id
        return None

    owners = [owner_of(s) for s in starts]
    in_span = [False] * len(starts)
    trials = []
    for s, e, ext_id in case.spans:
        a, b = bisect_left(starts, s), bisect_left(starts, e)
        in_span[a:b] = [True] * (b - a)
        n = b - a
        hits = [h for h in report.semantic_hits if h.q_start < e and h.q_end > s]
        trials.append({
            "ext_id": ext_id,
            "tokens": n,
            "coverage": sum(o is not None for o in owners[a:b]) / n if n else 0.0,
            "correct_coverage": sum(o == ext_id for o in owners[a:b]) / n if n else 0.0,
            "semantic_correct": any(h.doc.ext_id == ext_id for h in hits),
            "semantic_any": bool(hits),
        })
    filler_tokens = [i for i, inside in enumerate(in_span) if not inside]
    outside_hits = sum(
        1 for h in report.semantic_hits
        if not any(h.q_start < e and h.q_end > s for s, e, _ in case.spans)
    )
    return {
        "trials": trials,
        "matched_in_span": sum(o is not None for o, inside in zip(owners, in_span) if inside),
        "filler_tokens": len(filler_tokens),
        "filler_matched": sum(owners[i] is not None for i in filler_tokens),
        "filler_semantic_hits": outside_hits,
        "flags": sorted({flag.kind for flag in report.flags}),
        "score": report.score,
    }


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize(scored: list[dict], seconds: float | None = None) -> dict:
    trials = [t for doc in scored for t in doc["trials"]]
    n = len(trials)
    detected = sum(t["coverage"] >= DETECTED_COVERAGE for t in trials)
    correct = sum(t["correct_coverage"] >= DETECTED_COVERAGE for t in trials)
    semantic = sum(t["semantic_correct"] for t in trials)
    either = sum(t["coverage"] >= DETECTED_COVERAGE or t["semantic_correct"] for t in trials)
    matched_span = sum(doc["matched_in_span"] for doc in scored)
    filler_tokens = sum(doc["filler_tokens"] for doc in scored)
    filler_matched = sum(doc["filler_matched"] for doc in scored)
    flag_kinds: dict[str, int] = {}
    for doc in scored:
        for kind in doc["flags"]:
            flag_kinds[kind] = flag_kinds.get(kind, 0) + 1
    return {
        "docs": len(scored),
        "n": n,
        "detected": detected,
        "detected_rate": detected / n if n else None,
        "detected_ci": wilson_interval(detected, n),
        "correct_source_rate": correct / n if n else None,
        "semantic_rate": semantic / n if n else None,
        "semantic_ci": wilson_interval(semantic, n),
        "either_rate": either / n if n else None,
        "mean_coverage": sum(t["coverage"] for t in trials) / n if n else None,
        "flagged_doc_rate": sum(bool(doc["flags"]) for doc in scored) / len(scored) if scored else None,
        "flag_kinds": flag_kinds,
        "filler_match_rate": filler_matched / filler_tokens if filler_tokens else None,
        "precision": matched_span / (matched_span + filler_matched) if matched_span + filler_matched else None,
        "filler_semantic_per_1000": (
            1000 * sum(doc["filler_semantic_hits"] for doc in scored) / filler_tokens if filler_tokens else None
        ),
        "seconds_per_doc": seconds,
    }
