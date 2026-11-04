"""LM Studio のローカルモデルでAI文を生成し、data/ai_threshold/ai_local.jsonl に追記する。中断後は続きから再開する。"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simcheck import DATA_DIR

DATA = DATA_DIR.parent / "ai_threshold"
OUT = DATA / "ai_local.jsonl"
LMS = r"E:\LM Studio\resources\app\.webpack\lms.exe"
API = "http://127.0.0.1:1234/v1/chat/completions"
SEED = 20261103
TEMPERATURE = 0.7
ABSTRACT_STRATA = ["ophth_japan", "ophth_other", "med_japan", "med_other"]
MODELS = ["gemma-4-e4b-uncensored-hauhaucs-aggressive", "qwen3.8-27b-heretic-abliterated-uncensored"]
N_NEW_PER_STRATUM = 25
N_POLISH = 40
N_LONG = 20

_FORMAT = " Output only the text, with no title, no heading, no markdown, and no comments before or after it."
NEW_PROMPTS = [
    "Write a structured abstract of about 250 words for a medical research article titled: \"{title}\"." + _FORMAT,
    "You are a medical researcher preparing a manuscript for a peer-reviewed journal. Write the abstract "
    "(Purpose, Methods, Results, Conclusions; about 250 words) for your paper titled \"{title}\". Include "
    "plausible specific numbers." + _FORMAT,
    "Draft the abstract for this journal article in clear academic English, about 200 to 300 words: "
    "\"{title}\"." + _FORMAT,
]
POLISH_PROMPTS = [
    "Improve the English of this abstract so it reads as polished academic writing. Keep all facts and numbers."
    + _FORMAT + "\n\n{text}",
    "You are a professional medical editor. Rewrite this abstract for clarity and flow while keeping the meaning "
    "and all results." + _FORMAT + "\n\n{text}",
    "Edit this abstract for a high-impact medical journal. Keep every number and finding." + _FORMAT + "\n\n{text}",
]
LONG_PROMPTS = [
    "Write the Introduction and Discussion sections, about 1000 words in total, for a medical research article "
    "titled \"{title}\". Use plain paragraphs without citations." + _FORMAT,
    "You are writing a manuscript for a peer-reviewed medical journal titled \"{title}\". Write the Introduction "
    "and the Discussion, about 1000 words in total, in continuous paragraphs without reference numbers." + _FORMAT,
]


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)
    text = re.sub(r"^\s*(?:here(?:'s| is)|sure|certainly)[^\n]*:\s*\n", "", text, flags=re.I)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    text = text.replace("**", "").replace("__", "")
    return text.strip()


def build_jobs() -> list[dict]:
    rng = random.Random(SEED)
    human = [json.loads(line) for line in open(DATA / "human.jsonl", encoding="utf-8")]
    by_stratum = {s: [h for h in human if h["stratum"] == s] for s in ABSTRACT_STRATA}
    jobs = []
    for model in MODELS:
        seq = 0
        for stratum, rows in by_stratum.items():
            for row in rng.sample(rows, N_NEW_PER_STRATUM):
                template = seq % len(NEW_PROMPTS)
                jobs.append({"model": model, "mode": "new", "source_id": row["id"], "stratum": stratum,
                             "template": template, "prompt": NEW_PROMPTS[template].format(title=row["title"]),
                             "max_tokens": 900})
                seq += 1
        pool = [row for rows in by_stratum.values() for row in rows]
        for i, row in enumerate(rng.sample(pool, N_POLISH)):
            template = i % len(POLISH_PROMPTS)
            jobs.append({"model": model, "mode": "polish", "source_id": row["id"], "stratum": row["stratum"],
                         "template": template, "prompt": POLISH_PROMPTS[template].format(text=row["text"]),
                         "max_tokens": 1200})
        for i, row in enumerate(rng.sample(pool, N_LONG)):
            template = i % len(LONG_PROMPTS)
            jobs.append({"model": model, "mode": "long", "source_id": row["id"], "stratum": row["stratum"],
                         "template": template, "prompt": LONG_PROMPTS[template].format(title=row["title"]),
                         "max_tokens": 2500})
    return jobs


def lms(*args: str):
    subprocess.run([LMS, *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)


def generate(client: httpx.Client, job: dict) -> tuple[str, str | None]:
    response = client.post(API, json={
        "model": job["model"], "temperature": TEMPERATURE, "max_tokens": job["max_tokens"], "stream": False,
        # 思考過程でトークンを使い切ると本文が空になる
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": job["prompt"]}],
    })
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return clean(choice["message"].get("content") or ""), choice.get("finish_reason")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-model", type=int, help="動作確認用")
    args = parser.parse_args()

    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["model"], r["mode"], r["source_id"]))

    jobs = build_jobs()
    lms("server", "start")
    with httpx.Client(timeout=1200) as client, open(OUT, "a", encoding="utf-8") as out:
        for model in MODELS:
            mine = [j for j in jobs if j["model"] == model and (model, j["mode"], j["source_id"]) not in done]
            if args.limit_per_model:
                # 新規、推敲、長文を1件ずつ確かめられるように順に取る
                mine = [next((j for j in mine if j["mode"] == m), None) for m in ("new", "polish", "long")]
                mine = [j for j in mine if j][:args.limit_per_model]
            if not mine:
                continue
            print(f"{model}: {len(mine)} 件", flush=True)
            lms("load", model, "--gpu", "max", "--context-length", "8192", "--identifier", model, "--yes")
            try:
                started = time.time()
                for i, job in enumerate(mine, 1):
                    try:
                        text, finish = generate(client, job)
                    except httpx.HTTPError as e:
                        print(f"  失敗: {job['source_id']}: {e}", flush=True)
                        continue
                    if finish != "stop" or not text:
                        print(f"  不完全のため破棄: {job['source_id']} finish={finish} 文字数={len(text)}", flush=True)
                        continue
                    record = {**job, "temperature": TEMPERATURE, "finish_reason": finish, "text": text,
                              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    del record["max_tokens"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    if i % 10 == 0 or args.limit_per_model:
                        rate = (time.time() - started) / i
                        print(f"  {i}/{len(mine)} ({rate:.0f}秒/件) {job['mode']} 文字数={len(text)} finish={finish}", flush=True)
            finally:
                lms("unload", model)
    lms("server", "stop")


if __name__ == "__main__":
    main()
