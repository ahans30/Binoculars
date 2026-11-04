"""撤回論文のチェックを、参考文献の書式ごとに評価する。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import VOCABULARY, split_of, stable_rng, wilson_interval
from simcheck.integrity import MIN_TITLE_TOKENS, RETRACTION_PATH, find_retracted, parse_retractions
from simcheck.textnorm import tokenize

N_RETRACTED = 200
N_CONTROL = 400
REFS_PER_DOC = 20
_AUTHORS = "Lee A, Kim B, Sato C"
_VENUE = "J Med Res. 2019;12(3):45-52"


def _reference(title: str, identifier: str = "") -> str:
    return f"{_AUTHORS}. {title}. {_VENUE}.{(' ' + identifier) if identifier else ''}"


def _wrap_middle(title: str) -> str:
    words = title.split()
    middle = len(words) // 2
    return " ".join(words[:middle]) + "\n" + " ".join(words[middle:])


def _typo(title: str, rng: random.Random) -> str:
    words = title.split()
    middle = len(words) // 2
    words[middle] = rng.choice([v for v in VOCABULARY if v != words[middle].lower()])
    return " ".join(words)


def _abbreviate(title: str) -> str:
    words = title.split()
    return " ".join(words[:max(MIN_TITLE_TOKENS, int(len(words) * 0.7))])


def _wrap_doi(doi: str) -> str:
    head, tail = doi.split("/", 1)
    return f"{head}/\n{tail}"


# entry.title, entry.doi, entry.pmid から1件の参考文献を作る。識別子がない書式はNoneを返す
FORMATS = {
    "DOIあり": lambda e, rng: _reference(e.title, f"doi:{e.doi}") if e.doi else None,
    "DOIのURL": lambda e, rng: _reference(e.title, f"https://doi.org/{e.doi}") if e.doi else None,
    "DOIが行末で折り返し": lambda e, rng: _reference(e.title, f"doi:{_wrap_doi(e.doi)}") if e.doi else None,
    "PMIDのみ": lambda e, rng: _reference(_typo(e.title, rng), f"PMID: {e.pmid}") if e.pmid else None,
    "題名のみ": lambda e, rng: _reference(e.title),
    "題名が行末で折り返し": lambda e, rng: _reference(_wrap_middle(e.title)),
    "題名が小文字": lambda e, rng: _reference(e.title.lower()),
    "題名を省略": lambda e, rng: _reference(_abbreviate(e.title)),
    "題名に1語の誤り": lambda e, rng: _reference(_typo(e.title, rng)),
}
CONTROL_FORMATS = ("DOIあり", "題名のみ")


def as_document(references: list[str]) -> str:
    numbered = "\n".join(f"{i}. {ref}" for i, ref in enumerate(references, 1))
    # 文書の前半にある「文献」は見出しと見なされないので、本文を実際の原稿に近い長さにする
    sentence = "Body text of the manuscript. "
    body = sentence * (len(numbered) * 3 // len(sentence) + 1)
    return f"{body}\n\nReferences\n{numbered}"


def _hits(references: list[str], data) -> set[int]:
    by_id, by_title, _ = find_retracted(as_document(references), data)
    return {id(e) for e in by_id} | {id(e) for e in by_title}


def _sample_retracted(data, split: str) -> list:
    entries = [
        e for doi, e in sorted(data.by_doi.items())
        if e.status == "撤回" and split_of(doi) == split and len(tokenize(e.title).tokens) >= MIN_TITLE_TOKENS
    ]
    return stable_rng("retracted", split).sample(entries, min(N_RETRACTED, len(entries)))


def _sample_control(store, data, split: str) -> list:
    """撤回の記録がないPubMed文献を、参考文献の材料にする。"""
    rows = store.conn.execute("SELECT doc_id, ext_id FROM docs WHERE source='pubmed' ORDER BY doc_id").fetchall()
    ids = [doc_id for doc_id, ext_id in rows if split_of(ext_id) == split]
    order = stable_rng("control", split).sample(ids, len(ids))
    control = []
    for i in range(0, len(order), 500):
        for doc in store.get_docs(order[i:i + 500]).values():
            doi = (doc.meta.get("doi") or "").lower()
            if doi and doi not in data.by_doi and not doc.meta.get("retracted"):
                control.append(_ControlEntry(doc.title, doi))
        if len(control) >= N_CONTROL:
            return control[:N_CONTROL]
    return control


class _ControlEntry:
    def __init__(self, title: str, doi: str):
        self.title, self.doi, self.pmid = title, doi, ""


def evaluate_retraction(store, split: str, data=None) -> dict:
    data = data or parse_retractions(RETRACTION_PATH)
    retracted = _sample_retracted(data, split)
    results = {"formats": {}, "control": {}}
    for name, make in FORMATS.items():
        usable = []
        for entry in retracted:
            reference = make(entry, stable_rng("format", name, entry.doi))
            if reference:
                usable.append((entry, reference))
        detected = 0
        for i in range(0, len(usable), REFS_PER_DOC):
            chunk = usable[i:i + REFS_PER_DOC]
            found = _hits([ref for _, ref in chunk], data)
            detected += sum(id(entry) in found for entry, _ in chunk)
        results["formats"][name] = {
            "n": len(usable), "detected": detected,
            "rate": detected / len(usable) if usable else None,
            "ci": wilson_interval(detected, len(usable)),
        }

    control = _sample_control(store, data, split)
    for name in CONTROL_FORMATS:
        references = [FORMATS[name](entry, stable_rng("control", name, entry.doi)) for entry in control]
        flagged = sum(
            len(_hits(references[i:i + REFS_PER_DOC], data)) for i in range(0, len(references), REFS_PER_DOC)
        )
        results["control"][name] = {
            "n": len(references), "flagged": flagged,
            "per_1000": 1000 * flagged / len(references) if references else None,
            "ci": wilson_interval(flagged, len(references)),
        }
    return results
