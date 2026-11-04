"""scores.jsonl を protocol.md の主解析に沿って集計し、results.json に書き出す。"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simcheck import DATA_DIR

DATA = DATA_DIR.parent / "ai_threshold"
OUT_DIR = Path(__file__).resolve().parent
SEED = 20261105
N_BOOT = 2000
CURRENT_THRESHOLD = 0.9015
JAPAN_STRATA = ["ophth_japan", "med_japan"]
ABSTRACT_STRATA = ["ophth_japan", "ophth_other", "med_japan", "med_other"]


def auc(human: np.ndarray, ai: np.ndarray) -> float:
    # スコアが低いほどAIなので、AIが人間より低い確率を求める
    ranks = np.concatenate([ai, human]).argsort().argsort().astype(float)
    # 同順位は実数スコアでは稀なので単純順位で扱う
    return 1 - (ranks[: len(ai)].sum() - len(ai) * (len(ai) - 1) / 2) / (len(ai) * len(human))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(fn, human: np.ndarray, ai: np.ndarray, rng) -> tuple[float, float]:
    values = [
        fn(rng.choice(human, len(human)), rng.choice(ai, len(ai)))
        for _ in range(N_BOOT)
    ]
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def threshold_at_fpr(calibration: np.ndarray, fpr: float) -> float:
    # スコアがこの値未満のときAIと判定する
    return float(np.quantile(calibration, fpr))


def rate(values: np.ndarray, threshold: float) -> dict:
    k, n = int((values < threshold).sum()), len(values)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson": wilson(k, n)}


def main():
    rng = np.random.default_rng(SEED)
    rows = [json.loads(line) for line in open(DATA / "scores.jsonl", encoding="utf-8")]
    human_abs = [r for r in rows if r["kind"] == "human"]
    ai_new = [r for r in rows if r["kind"] == "new"]
    generators = sorted({r["generator"] for r in ai_new})

    # 層ごとに半分へ分ける
    cal, val = [], []
    for stratum in ABSTRACT_STRATA:
        members = [r for r in human_abs if r["stratum"] == stratum]
        rng.shuffle(members)
        half = len(members) // 2
        cal += members[:half]
        val += members[half:]
    cal_s = np.array([r["score"] for r in cal])
    val_s = np.array([r["score"] for r in val])
    new_s = np.array([r["score"] for r in ai_new])

    result = {"n": {"human_calibration": len(cal), "human_validation": len(val), "ai_new": len(ai_new)}}
    point = auc(val_s, new_s)
    result["auc_new"] = {"point": point, "ci": boot_ci(auc, val_s, new_s, rng)}
    result["auc_by_generator"] = {}
    for g in generators:
        s = np.array([r["score"] for r in ai_new if r["generator"] == g])
        result["auc_by_generator"][g] = {"n": len(s), "point": auc(val_s, s), "ci": boot_ci(auc, val_s, s, rng)}

    result["thresholds"] = {}
    for fpr in (0.01, 0.05):
        t = threshold_at_fpr(cal_s, fpr)
        def tpr_at(human_boot, ai_boot, fpr=fpr):
            return float((ai_boot < t).mean())
        entry = {
            "threshold": t,
            "validation_fpr": rate(val_s, t),
            "tpr_new": {**rate(new_s, t), "ci": boot_ci(tpr_at, val_s, new_s, rng)},
            "tpr_by_generator": {
                g: rate(np.array([r["score"] for r in ai_new if r["generator"] == g]), t) for g in generators
            },
            "fpr_by_stratum": {
                s: rate(np.array([r["score"] for r in val if r["stratum"] == s]), t) for s in ABSTRACT_STRATA
            },
        }
        result["thresholds"][f"fpr_{int(fpr * 100)}"] = entry

    # 現在の既定値の性能
    result["current_threshold"] = {
        "threshold": CURRENT_THRESHOLD,
        "fpr_all_human_abstracts": rate(np.array([r["score"] for r in human_abs]), CURRENT_THRESHOLD),
        "tpr_new": rate(new_s, CURRENT_THRESHOLD),
    }

    # 主解析とは別に報告する条件
    t1 = result["thresholds"]["fpr_1"]["threshold"]
    t5 = result["thresholds"]["fpr_5"]["threshold"]
    result["secondary"] = {}
    polish = [r for r in rows if r["kind"] == "polish"]
    if polish:
        s = np.array([r["score"] for r in polish])
        result["secondary"]["polish"] = {
            "n": len(s), "auc": auc(val_s, s), "ci": boot_ci(auc, val_s, s, rng),
            "tpr_fpr1": rate(s, t1), "tpr_fpr5": rate(s, t5),
            "by_generator": {
                g: {"n": sum(r["generator"] == g for r in polish),
                    "auc": auc(val_s, np.array([r["score"] for r in polish if r["generator"] == g]))}
                for g in sorted({r["generator"] for r in polish})
            },
        }
    long_h = np.array([r["score"] for r in rows if r["kind"] == "human_long"])
    long_a = np.array([r["score"] for r in rows if r["kind"] == "long"])
    if len(long_h) and len(long_a):
        half = len(long_h) // 2
        shuffled = rng.permutation(long_h)
        cal_l, val_l = shuffled[:half], shuffled[half:]
        t_l = threshold_at_fpr(cal_l, 0.05)
        result["secondary"]["long"] = {
            "n_human": len(long_h), "n_ai": len(long_a), "auc": auc(long_h, long_a),
            "ci": boot_ci(auc, long_h, long_a, rng),
            "threshold_fpr5": t_l, "validation_fpr": rate(val_l, t_l), "tpr": rate(long_a, t_l),
            "median_words_human": float(np.median([r["n_words"] for r in rows if r["kind"] == "human_long"])),
            "median_words_ai": float(np.median([r["n_words"] for r in rows if r["kind"] == "long"])),
        }
    result["by_length"] = {}
    for limit in (50, 100, 200):
        key = f"score_{limit}w"
        h = np.array([r[key] for r in val if r[key] is not None])
        a = np.array([r[key] for r in ai_new if r[key] is not None])
        result["by_length"][f"{limit}w"] = {"n_human": len(h), "n_ai": len(a), "auc": auc(h, a), "ci": boot_ci(auc, h, a, rng)}

    def dist(values):
        q = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        return {f"p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)}

    result["distributions"] = {
        "human_abstract": dist(np.array([r["score"] for r in human_abs])),
        **{f"human_{s}": dist(np.array([r["score"] for r in human_abs if r["stratum"] == s])) for s in ABSTRACT_STRATA},
        **{f"new_{g}": dist(np.array([r["score"] for r in ai_new if r["generator"] == g])) for g in generators},
    }

    # 判定基準（protocol.md）
    fpr1 = result["thresholds"]["fpr_1"]
    checks = {
        "1_auc_lower_ge_0.90": result["auc_new"]["ci"][0] >= 0.90,
        "2_tpr_lower_ge_0.50": fpr1["tpr_new"]["ci"][0] >= 0.50,
        "3_japan_fpr_le_3pct": all(fpr1["fpr_by_stratum"][s]["rate"] <= 0.03 for s in JAPAN_STRATA),
        "4_each_generator_tpr_ge_0.50": all(v["rate"] >= 0.50 for v in fpr1["tpr_by_generator"].values()),
    }
    result["criteria"] = checks
    result["decision"] = "update_threshold" if all(checks.values()) else "remove_label"

    (OUT_DIR / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
