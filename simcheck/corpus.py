from pathlib import Path

from .pmc import add_pmc_query
from .pubmed import add_pubmed_query
from .store import Store

_ADDERS = {"pubmed": add_pubmed_query, "pmc": add_pmc_query}


def read_queries(path: str | Path) -> list[tuple[str, str]]:
    """1行に `pubmed: 検索式` または `pmc: 検索式` を書いたファイルを読む。"""
    queries = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, separator, term = line.partition(":")
        kind, term = kind.strip().lower(), term.strip()
        if not separator or kind not in _ADDERS or not term:
            raise ValueError(f"{path}:{number}: `pubmed: 検索式` または `pmc: 検索式` の形式で書いてください")
        queries.append((kind, term))
    return queries


def build_corpus(store: Store, path: str | Path, progress=print) -> dict:
    # 途中で失敗した索引を使わないよう、1件でも失敗したら例外のまま止める
    totals = {"pubmed": 0, "pmc": 0}
    for kind, term in read_queries(path):
        progress(f"[{kind}] {term}")
        totals[kind] += _ADDERS[kind](store, term, progress=progress)["added"]
    return totals
