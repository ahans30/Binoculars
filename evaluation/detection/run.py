"""合成原稿で検出率を測り、results_<split>.json と results_<split>.md を書き出す。

    venv/Scripts/python.exe evaluation/detection/run.py --split dev
    venv/Scripts/python.exe evaluation/detection/run.py --split dev --set-baseline
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CONDITIONS, DATA, Case, build_cases, prepare_cases, score_report, summarize
from paraphrase import load_paraphrases
from retraction import evaluate_retraction
from simcheck import engine
from simcheck.engine import CheckOptions, check_document
from simcheck.rerank import DEFAULT_MIN_GAIN
from simcheck.semantic import DEFAULT_THRESHOLD, SemanticIndex
from simcheck.store import Store

LABELS = {
    "homoglyph_cyrillic": "同形文字（キリル）", "homoglyph_greek": "同形文字（ギリシャ）",
    "homoglyph_armenian": "同形文字（アルメニア）", "fullwidth": "全角", "math_bold": "数学用英字",
    "zero_width": "ゼロ幅文字", "soft_hyphen": "ソフトハイフン", "variation_selector": "異体字セレクタ",
    "tag_character": "タグ文字", "bidi_control": "双方向制御文字", "mongolian_separator": "モンゴル母音区切り",
    "combining_joiner": "結合用字", "special_spaces": "特殊な空白", "shuffle_sentences": "文の順序の入れ替え",
    "llm_light": "軽い推敲", "llm_heavy": "強い言い換え", "llm_translate_en": "日本語からの英訳",
    "control_filler": "地の文のみ", "control_human": "人間の抄録",
}


def code_hash() -> str:
    """どのコードで測った結果かを後から確かめられるようにする。"""
    digest = hashlib.sha256()
    for path in sorted((HERE.parents[1] / "simcheck").glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def label_of(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    head, _, n = name.rpartition("_")
    if head == "verbatim":
        return f"{n}語"
    return {"drop_every": f"{n}語ごとに削除", "replace_every": f"{n}語ごとに置換", "swap_every": f"{n}語ごとに入れ替え"}[head]


def run_case(case: Case, store: Store, options: CheckOptions, tmp_dir: str) -> dict:
    # 購読誌の本文を含むので、処理が終わったら消す
    fd, path = tempfile.mkstemp(suffix=".txt", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(case.text)
        options.exclude_ext_ids = set(case.exclude_ext_ids)
        return score_report(check_document(path, store, options), case)
    finally:
        os.unlink(path)


_store = None
_tmp_dir = None


def _worker_init(tmp_dir: str):
    global _store, _tmp_dir
    _store, _tmp_dir = Store(), tmp_dir


def _worker_run(job: tuple[Case, CheckOptions]) -> dict:
    case, options = job
    return run_case(case, _store, options, _tmp_dir)


class _SharedIndex:
    """言い換え検索の索引は約900MBあるので、毎回読み込まないようにする。"""
    _index = None
    exists = staticmethod(SemanticIndex.exists)

    def __new__(cls):
        if cls._index is None:
            cls._index = SemanticIndex()
        return cls._index


def _pct(value, digits=1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _with_ci(rate, ci) -> str:
    return "—" if rate is None else f"{rate * 100:.1f}%（{ci[0] * 100:.1f}〜{ci[1] * 100:.1f}）"


def _delta(cell: dict, base: dict | None, key: str) -> str:
    if not base or base.get(key) is None or cell.get(key) is None:
        return "—"
    return f"{(cell[key] - base[key]) * 100:+.1f}"


def render_markdown(results: dict, baseline: dict | None) -> str:
    lines = [
        f"# 検出率の評価結果（{results['split']}）", "",
        f"実行: {results['finished_at']} / 設定: 最小{results['min_words']}語、"
        f"言い換えの閾値{results['semantic_threshold']}、再評価{'あり' if results.get('use_rerank') else 'なし'} / 照合先: {json.dumps(results['index_stats']['docs'], ensure_ascii=False)}", "",
        "合成原稿での値で、実際の原稿での検出率ではありません。計画は [protocol.md](protocol.md) を参照してください。",
        "検出率は転用した文章の語の50%以上が一致した割合で、括弧内は95%信頼区間です。",
    ]
    if baseline:
        lines += ["", f"前回差は基準（{baseline['finished_at']}）との差で、単位はパーセントポイントです。"]
    base_cells = (baseline or {}).get("cells", {})
    groups: dict[str, list] = {}
    for cell_key, cell in results["cells"].items():
        groups.setdefault(cell["group"], []).append((cell_key, cell))
    for group, cells in groups.items():
        lines += ["", f"## {group}", ""]
        if group == "対照":
            lines += [
                "| 条件 | 原稿 | 平均の類似度 | 地の文の一致率 | 言い換え候補（1,000語あたり） | フラッグの率 | 秒/件 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for _, c in cells:
                lines.append(
                    f"| {label_of(c['condition'])} | {c['docs']} | {_pct(c['mean_score'])} | {_pct(c['filler_match_rate'])} | "
                    f"{c['filler_semantic_per_1000']:.2f} | {_pct(c['flagged_doc_rate'], 0)} | {c['seconds_per_doc']:.1f} |"
                )
            continue
        lines += [
            "| 条件 | 層 | n | 検出率 | 正しい出典 | 言い換え検索 | どちらか | フラッグの率 | 前回差 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cell_key, c in cells:
            semantic = _with_ci(c["semantic_rate"], c["semantic_ci"]) if c["semantic"] else "—"
            either = _pct(c["either_rate"]) if c["semantic"] else "—"
            lines.append(
                f"| {label_of(c['condition'])} | {c['stratum']} | {c['n']} | {_with_ci(c['detected_rate'], c['detected_ci'])} | "
                f"{_pct(c['correct_source_rate'])} | {semantic} | {either} | {_pct(c['flagged_doc_rate'], 0)} | "
                f"{_delta(c, base_cells.get(cell_key), 'detected_rate')} |"
            )

    retraction = results.get("retraction")
    if retraction:
        lines += ["", "## 撤回論文のチェック", "", "| 書式 | n | 検出率 |", "|---|---:|---:|"]
        for name, r in retraction["formats"].items():
            lines.append(f"| {name} | {r['n']} | {_with_ci(r['rate'], r['ci'])} |")
        lines += ["", "| 撤回の記録がない文献の書式 | n | 誤検出（件） | 1,000文献あたり |", "|---|---:|---:|---:|"]
        for name, r in retraction["control"].items():
            lines.append(f"| {name} | {r['n']} | {r['flagged']} | {r['per_1000']:.1f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--only", default="*", help="条件名のパターン（例: verbatim_*）")
    parser.add_argument("--max-docs", type=int, help="動作確認用（条件ごとの原稿数の上限）")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--min-words", type=int, default=CheckOptions().min_words)
    parser.add_argument("--semantic-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-rerank", action="store_true", help="言い換えの候補を再評価しない")
    parser.add_argument("--rerank-min-gain", type=float, default=DEFAULT_MIN_GAIN)
    parser.add_argument("--no-retraction", action="store_true")
    parser.add_argument("--tag", help="結果のファイル名に付ける名前（設定の比較用）")
    parser.add_argument("--set-baseline", action="store_true", help="この結果を以後の比較の基準にする")
    args = parser.parse_args()

    # 同じフォルダを別の作業が編集していることがあるので、計測中に変わったら結果に残す
    code_at_start = code_hash()
    suffix = f"{args.split}_{args.tag}" if args.tag else args.split
    json_path = HERE / f"results_{suffix}.json"
    baseline_path = HERE / f"baseline_{args.split}.json"
    previous = json.loads(json_path.read_text(encoding="utf-8")) if args.only != "*" and json_path.exists() else None

    with Store() as store:
        data = prepare_cases(store, args.split)
        paraphrases = load_paraphrases(args.split)
        cells = dict(previous["cells"]) if previous else {}
        DATA.mkdir(parents=True, exist_ok=True)
        engine.SemanticIndex = _SharedIndex
        with tempfile.TemporaryDirectory(dir=DATA) as tmp_dir, ProcessPoolExecutor(
            args.workers, initializer=_worker_init, initargs=(tmp_dir,),
        ) as pool:
            for condition in CONDITIONS:
                if not fnmatch(condition.name, args.only):
                    continue
                for stratum in condition.strata:
                    cases = build_cases(condition, stratum, data, paraphrases)[:args.max_docs]
                    if not cases:
                        print(f"{condition.name}/{stratum}: 対象なし（書き換え文が未生成）", flush=True)
                        continue
                    options = CheckOptions(
                        use_semantic=condition.semantic, min_words=args.min_words,
                        semantic_threshold=args.semantic_threshold,
                        use_rerank=not args.no_rerank, rerank_min_gain=args.rerank_min_gain,
                    )
                    started = time.time()
                    if condition.semantic:
                        # 言い換え検索はGPUを使うので並列にしない
                        scored = [run_case(case, store, options, tmp_dir) for case in cases]
                    else:
                        scored = list(pool.map(_worker_run, [(case, options) for case in cases]))
                    seconds = (time.time() - started) / len(cases)
                    summary = summarize(scored, seconds)
                    summary.update({
                        "condition": condition.name, "group": condition.group, "stratum": stratum,
                        "semantic": condition.semantic,
                        "mean_score": statistics.fmean(doc["score"] for doc in scored),
                    })
                    cells[f"{condition.name}/{stratum}"] = summary
                    print(
                        f"{condition.name}/{stratum}: n={summary['n']} 検出率={_pct(summary['detected_rate'])} "
                        f"言い換え={_pct(summary['semantic_rate'])} 地の文={_pct(summary['filler_match_rate'], 2)} "
                        f"フラッグ={_pct(summary['flagged_doc_rate'], 0)} {seconds:.1f}秒/件",
                        flush=True,
                    )

        results = {
            "split": args.split, "finished_at": time.strftime("%Y-%m-%d %H:%M"),
            "min_words": args.min_words, "semantic_threshold": args.semantic_threshold,
            "use_rerank": not args.no_rerank, "rerank_min_gain": args.rerank_min_gain,
            "code_hash": code_at_start, "code_changed_during_run": code_hash() != code_at_start,
            "index_stats": store.stats(), "cells": cells,
            "retraction": None if args.no_retraction else evaluate_retraction(store, args.split),
        }
        if previous and args.no_retraction:
            results["retraction"] = previous.get("retraction")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.set_baseline:
        shutil.copyfile(json_path, baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    json_path.with_suffix(".md").write_text(render_markdown(results, baseline), encoding="utf-8")
    print(f"結果: {json_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
