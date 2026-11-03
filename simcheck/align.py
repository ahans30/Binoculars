from dataclasses import dataclass
from itertools import accumulate

from rapidfuzz.distance import Levenshtein

from .textnorm import Tokens, shingle_hashes, WORD_WEIGHT

# 小さな編集をはさんだ一致を1つにまとめるときの、許容する隙間（英語8語ぶん）
MAX_GAP_WEIGHT = 8 * WORD_WEIGHT


@dataclass(slots=True)
class Alignment:
    q_start: int  # クエリ側トークン位置（end は含まない）
    q_end: int
    s_start: int
    s_end: int
    similarity: float  # トークン列の編集距離に基づく一致率
    weight: int  # クエリ側の重み


def build_query_index(q: Tokens) -> dict[int, list[int]]:
    index: dict[int, list[int]] = {}
    for h, i in shingle_hashes(q):
        index.setdefault(h, []).append(i)
    return index


def _exact_blocks(q: Tokens, s: Tokens, q_index) -> list[tuple[int, int, int, int]]:
    qt, st = q.tokens, s.tokens
    seeds = []
    for h, si in shingle_hashes(s):
        for qi in q_index.get(h, ()):
            seeds.append((qi, si))
    seeds.sort()

    blocks, used = [], set()
    for qi, si in seeds:
        if (qi, si) in used:
            continue
        qs, ss = qi, si
        while qs > 0 and ss > 0 and qt[qs - 1] == st[ss - 1]:
            qs -= 1
            ss -= 1
        qe, se = qi, si
        while qe < len(qt) and se < len(st) and qt[qe] == st[se]:
            qe += 1
            se += 1
        used.update((qs + k, ss + k) for k in range(qe - qs))
        blocks.append((qs, qe, ss, se))
    return blocks


def _non_overlapping(blocks, nq: int, ns: int):
    # 繰り返し表現で同じ箇所が何度も一致するため、長い一致を優先して重なりを除く
    q_used = bytearray(nq)
    s_used = bytearray(ns)
    picked = []
    for qs, qe, ss, se in sorted(blocks, key=lambda b: b[0] - b[1]):
        if any(q_used[qs:qe]) or any(s_used[ss:se]):
            continue
        q_used[qs:qe] = b"\x01" * (qe - qs)
        s_used[ss:se] = b"\x01" * (se - ss)
        picked.append((qs, qe, ss, se))
    picked.sort()
    return picked


def align(q: Tokens, s: Tokens, q_index: dict[int, list[int]]) -> list[Alignment]:
    blocks = _exact_blocks(q, s, q_index)
    if not blocks:
        return []
    picked = _non_overlapping(blocks, len(q), len(s))

    q_sum = [0, *accumulate(q.weights)]
    s_sum = [0, *accumulate(s.weights)]

    merged = []
    cur = list(picked[0])
    for qs, qe, ss, se in picked[1:]:
        q_gap = q_sum[qs] - q_sum[cur[1]]
        s_gap = s_sum[ss] - s_sum[cur[3]] if ss >= cur[3] else None
        if s_gap is not None and q_gap <= MAX_GAP_WEIGHT and s_gap <= MAX_GAP_WEIGHT:
            cur[1], cur[3] = qe, se
        else:
            merged.append(cur)
            cur = [qs, qe, ss, se]
    merged.append(cur)

    result = []
    for qs, qe, ss, se in merged:
        sim = Levenshtein.normalized_similarity(q.tokens[qs:qe], s.tokens[ss:se])
        result.append(Alignment(qs, qe, ss, se, sim, q_sum[qe] - q_sum[qs]))
    return result
