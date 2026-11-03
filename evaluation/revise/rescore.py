"""保存した案（data/revise_eval/details.jsonl）を、LLMを呼ばずに今の検査で採点し直す。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simcheck import DATA_DIR
from simcheck.revise import _CJK, check_candidate, is_common_wording
from simcheck.store import Store

DETAILS = DATA_DIR.parent / "revise_eval" / "details.jsonl"


def main():
    rows = [json.loads(line) for line in open(DETAILS, encoding="utf-8")]
    kinds, total, passed, usable, with_passing, common = Counter(), 0, 0, 0, 0, []
    with Store() as store:
        for row in rows:
            if is_common_wording(row["sentence"], store):
                common.append(row["sentence"])
            any_passed = False
            for candidate in row.get("candidates", []):
                check = check_candidate(row["sentence"], candidate["text"], similarity=candidate.get("similarity"))
                total += 1
                usable += not _CJK.search(candidate["text"])
                passed += check.passed
                any_passed |= check.passed
                for problem in check.problems():
                    kinds[problem.split(":")[0].split("（")[0]] += 1
            with_passing += any_passed
    print(json.dumps({
        "sentences": len(rows), "common_wording": len(common), "candidates": total,
        "same_language": usable, "passed": passed, "sentences_with_passing_candidate": with_passing,
        "problems": dict(kinds),
    }, ensure_ascii=False, indent=1))
    for sentence in common:
        print("common:", sentence[:160])


if __name__ == "__main__":
    main()
