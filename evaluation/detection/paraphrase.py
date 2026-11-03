"""LM Studio のローカルモデルで書き換え文を作り、data/detection_eval に追記する。中断後は続きから再開する。

    venv/Scripts/python.exe evaluation/detection/paraphrase.py --split dev
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai_threshold"))
from cases import DATA, prepare_cases
from generate_ai import API, LMS, MODELS, TEMPERATURE, clean, lms
from simcheck.store import Store

N_PER_CELL = 100
MAX_TOKENS = 900
MAX_FAILURES = 5
RETRY_WAIT_SECONDS = 60
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.5, 1.6

_FORMAT = " Output only the rewritten text, with no heading, no markdown, and no comments before or after it."
PROMPTS = {
    "light": "Lightly edit this passage from a medical article to improve its English. Keep all facts and numbers."
             + _FORMAT + "\n\n{text}",
    "heavy": "Rewrite this passage from a medical article completely in your own words. Keep the meaning and all "
             "numbers, but change the vocabulary and the sentence structure as much as you can." + _FORMAT + "\n\n{text}",
    "translate_en": "Translate this Japanese passage into natural academic English. Keep all facts and numbers."
                     + _FORMAT + "\n\n{text}",
}
JOBS = [("pubmed", "light"), ("library", "light"), ("pubmed", "heavy"), ("library", "heavy"), ("library_ja", "translate_en")]


def output_path(split: str) -> Path:
    return DATA / f"paraphrases_{split}.jsonl"


def load_paraphrases(split: str) -> dict:
    path = output_path(split)
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            result[(row["passage_key"], row["mode"])] = row["text"]
    return result


def build_jobs(data: dict) -> list[dict]:
    jobs = []
    for stratum, mode in JOBS:
        for index, passage in enumerate(data["passages"][stratum][:N_PER_CELL]):
            jobs.append({
                "passage_key": f"{stratum}:{index}", "mode": mode, "model": MODELS[index % len(MODELS)],
                "ext_id": passage["ext_id"], "source": passage["text"],
                "prompt": PROMPTS[mode].format(text=passage["text"]),
            })
    return jobs


def _ensure_loaded(model: str) -> bool:
    """未読み込みなら読み込む。他の作業が読み込んだモデルを後で外さないよう、読み込んだかどうかを返す。"""
    listing = subprocess.run([LMS, "ps"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if model in listing.stdout:
        return False
    lms("load", model, "--gpu", "max", "--context-length", "8192", "--identifier", model, "--yes")
    return True


def _usable(job: dict, text: str, finish: str | None) -> bool:
    if finish != "stop" or not text:
        return False
    # 翻訳は言語が変わるので長さを比べられない
    if job["mode"] == "translate_en":
        return sum(c.isascii() for c in text) / len(text) > 0.95
    ratio = len(text.split()) / max(len(job["source"].split()), 1)
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit", type=int, help="動作確認用（モデルごとの件数）")
    args = parser.parse_args()

    with Store() as store:
        data = prepare_cases(store, args.split)
    done = set(load_paraphrases(args.split))
    jobs = [j for j in build_jobs(data) if (j["passage_key"], j["mode"]) not in done]
    lms("server", "start")
    with httpx.Client(timeout=1200) as client, open(output_path(args.split), "a", encoding="utf-8") as out:
        for model in MODELS:
            mine = [j for j in jobs if j["model"] == model]
            if args.limit:
                # 各条件を確かめられるように順に取る
                mine = [next((j for j in mine if j["mode"] == m), None) for m in PROMPTS]
                mine = [j for j in mine if j][:args.limit]
            if not mine:
                continue
            print(f"{model}: {len(mine)} 件", flush=True)
            loaded_here = _ensure_loaded(model)
            try:
                started = time.time()
                failures = 0
                for i, job in enumerate(mine, 1):
                    try:
                        response = client.post(API, json={
                            "model": model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "stream": False,
                            # 思考過程でトークンを使い切ると本文が空になる
                            "reasoning_effort": "none",
                            "messages": [{"role": "user", "content": job["prompt"]}],
                        })
                        response.raise_for_status()
                    except httpx.HTTPError as e:
                        # LM Studioは他の作業でも使うので、一時的に使えなくても待って続ける
                        print(f"  失敗: {job['passage_key']} {job['mode']}: {e}", flush=True)
                        failures += 1
                        if failures >= MAX_FAILURES:
                            raise RuntimeError("失敗が続いたので中止します") from e
                        time.sleep(RETRY_WAIT_SECONDS)
                        _ensure_loaded(model)
                        continue
                    failures = 0
                    choice = response.json()["choices"][0]
                    text = clean(choice["message"].get("content") or "")
                    if not _usable(job, text, choice.get("finish_reason")):
                        print(f"  破棄: {job['passage_key']} {job['mode']} finish={choice.get('finish_reason')}", flush=True)
                        continue
                    out.write(json.dumps({
                        "passage_key": job["passage_key"], "mode": job["mode"], "model": model,
                        "ext_id": job["ext_id"], "text": text, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    if i % 20 == 0 or args.limit:
                        rate = (time.time() - started) / i
                        print(f"  {i}/{len(mine)} ({rate:.1f}秒/件) {job['mode']}", flush=True)
            finally:
                if loaded_here:
                    lms("unload", model)


if __name__ == "__main__":
    main()
