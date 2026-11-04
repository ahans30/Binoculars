"""人間の文章とAI文を Binoculars で採点して data/ai_threshold/scores.jsonl に保存する。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import app
from simcheck import DATA_DIR

DATA = DATA_DIR.parent / "ai_threshold"
OBSERVER = "Qwen/Qwen2.5-1.5B"
PERFORMER = "Qwen/Qwen2.5-1.5B-Instruct"
WORD_LIMITS = [50, 100, 200]
CLAUDE_MODEL = "claude-opus-5-5"
_SCORE = re.compile(r"スコア: ([0-9.]+)")
_TOKENS = re.compile(r"評価トークン数: (\d+)")


def load_records() -> list[dict]:
    records = []
    for line in open(DATA / "human.jsonl", encoding="utf-8"):
        r = json.loads(line)
        kind = "human_long" if r["stratum"] == "en_long" else "human"
        records.append({"label": 0, "kind": kind, "generator": "human", "mode": "human",
                        "stratum": r["stratum"], "source_id": r["id"], "text": r["text"]})
    for line in open(DATA / "ai_local.jsonl", encoding="utf-8"):
        r = json.loads(line)
        records.append({"label": 1, "kind": r["mode"], "generator": r["model"], "mode": r["mode"],
                        "stratum": r["stratum"], "source_id": r["source_id"], "text": r["text"]})
    raw = (DATA / "claude_raw.txt").read_text(encoding="utf-8")
    for block in re.split(r"^### ", raw, flags=re.M)[1:]:
        header, text = block.split("\n", 1)
        mode, source_id, stratum = header.strip().split("|")
        records.append({"label": 1, "kind": mode, "generator": CLAUDE_MODEL, "mode": mode,
                        "stratum": stratum, "source_id": source_id, "text": text.strip()})
    return records


def main():
    records = load_records()
    print(f"{len(records)} 件を採点します", flush=True)
    detector = app.BinocularsDetector(OBSERVER, PERFORMER)

    def score(text: str) -> tuple[float, int]:
        detail, _ = detector.evaluate(text)
        return float(_SCORE.search(detail).group(1)), int(_TOKENS.search(detail).group(1))

    with open(DATA / "scores.jsonl", "w", encoding="utf-8") as out:
        for i, r in enumerate(records, 1):
            full, n_tokens = score(r["text"])
            words = r["text"].split()
            row = {k: r[k] for k in ("label", "kind", "generator", "mode", "stratum", "source_id")}
            row.update(score=full, n_tokens=n_tokens, n_words=len(words))
            for limit in WORD_LIMITS:
                # 全文より短い場合は同じ文章になるので計算しない
                row[f"score_{limit}w"] = score(" ".join(words[:limit]))[0] if len(words) > limit else None
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(records)}", flush=True)
    print("完了", flush=True)


if __name__ == "__main__":
    main()
