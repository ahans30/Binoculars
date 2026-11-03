import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "detection"))
from cases import (
    CONDITION_BY_NAME, CONDITIONS, Case, apply_condition, build_cases, compose_case, drop_every, insert_inside_words,
    replace_every, score_report, shuffle_sentences, split_of, summarize, swap_every, wilson_interval,
)
from retraction import FORMATS, as_document, evaluate_retraction
from run import label_of, render_markdown, run_case
from simcheck import integrity
from simcheck.engine import CheckOptions
from simcheck.integrity import find_retracted, parse_retractions
from simcheck.store import DocRow, Store, compress_text
from simcheck.textnorm import fingerprints, tokenize

SOURCE = (
    "Age-related macular degeneration is a leading cause of irreversible vision loss among older adults in developed "
    "countries, and anti-vascular endothelial growth factor therapy has transformed the management of the neovascular "
    "form of this disease over the past two decades of clinical practice in many hospitals around the world."
)
FILLER = (
    "Volcanic soils in the northern islands retain water differently depending on their mineral composition, which "
    "influences how farmers schedule irrigation throughout the short growing season every year."
)


@pytest.fixture(autouse=True)
def no_integrity_data(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "TORTURED_PATH", tmp_path / "no_tortured.csv")
    monkeypatch.setattr(integrity, "RETRACTION_PATH", tmp_path / "no_retraction.csv")


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index") as s:
        fps = fingerprints(SOURCE)
        s.add_batch(
            "pmid_1", "pubmed",
            [DocRow("pmid:1", "pubmed", "Source", "2024", None, False, {"doi": "10.1/a"}, compress_text(SOURCE))],
            np.fromiter(fps, dtype=np.uint64), np.zeros(len(fps), dtype=np.uint32),
        )
        yield s


def test_every_condition_has_a_label():
    assert all(label_of(c.name) for c in CONDITIONS)


def test_split_is_stable_and_uses_both_sides():
    assert split_of("pmid:1") == split_of("pmid:1")
    assert {split_of(f"pmid:{i}") for i in range(50)} == {"dev", "test"}


def test_compose_case_records_exact_positions():
    case = compose_case(["filler one", "filler two", "filler three"], [("copied a", "pmid:1"), ("copied b", "pmid:2")])
    assert [case.text[s:e] for s, e, _ in case.spans] == ["copied a", "copied b"]
    assert [ext for _, _, ext in case.spans] == ["pmid:1", "pmid:2"]
    assert "filler three" in case.text


def test_compose_case_keeps_passages_beyond_the_fillers():
    case = compose_case(["filler"], [("a", "1"), ("b", "2"), ("c", "3")])
    assert [case.text[s:e] for s, e, _ in case.spans] == ["a", "b", "c"]


def test_apply_condition_cuts_to_length_and_is_reproducible():
    text = " ".join(f"word{i}" for i in range(200))
    condition = CONDITION_BY_NAME["verbatim_10"]
    assert apply_condition(condition, text, "k").split() == [f"word{i}" for i in range(10)]
    replaced = CONDITION_BY_NAME["replace_every_5"]
    assert apply_condition(replaced, text, "k") == apply_condition(replaced, text, "k")


def test_word_edits_change_the_expected_words():
    text = " ".join(f"w{i}" for i in range(10))
    rng = random.Random(0)
    assert drop_every(5)(text, rng).split() == [f"w{i}" for i in range(10) if i not in (4, 9)]
    replaced = replace_every(5)(text, rng).split()
    assert len(replaced) == 10 and replaced[4] != "w4" and replaced[3] == "w3"
    assert swap_every(5)(text, rng).split()[4:6] == ["w5", "w4"]


def test_shuffle_keeps_every_sentence():
    text = "One is here. Two is here. Three is here. Four is here."
    shuffled = shuffle_sentences(text, random.Random(1))
    assert sorted(shuffled.replace(". ", ".|").split("|")) == sorted(text.replace(". ", ".|").split("|"))


def test_invisible_marks_stay_inside_words():
    assert insert_inside_words("|")("macular eye", None) == "mac|ula|r eye"


def test_build_cases_groups_passages_and_skips_missing_paraphrases():
    data = {
        "fillers": [FILLER] * 6,
        "human": [{"ext_id": f"h{i}", "text": FILLER} for i in range(16)],
        "passages": {"pubmed": [{"ext_id": f"p{i}", "text": SOURCE} for i in range(12)]},
    }
    cases = build_cases(CONDITION_BY_NAME["verbatim_15"], "pubmed", data)
    assert [len(c.spans) for c in cases] == [5, 5, 2]
    llm = build_cases(CONDITION_BY_NAME["llm_heavy"], "pubmed", data, {("pubmed:0", "heavy"): "rewritten"})
    assert len(llm) == 1 and len(llm[0].spans) == 1
    human = build_cases(CONDITION_BY_NAME["control_human"], "-", data)
    assert len(human) == 2 and human[0].spans == [] and len(human[0].exclude_ext_ids) == 8


def fake_report(text, owned, sources, semantic=()):
    return SimpleNamespace(
        text=text, owned_spans=owned, score=0.1, flags=[], semantic_hits=list(semantic),
        sources=[SimpleNamespace(doc=SimpleNamespace(ext_id=ext)) for ext in sources],
    )


def test_score_counts_words_inside_and_outside_the_passage():
    text = "aa bb cc dd ee ff gg hh"
    s, e = text.index("cc"), text.index("ff") + 2  # cc dd ee ff
    case = Case(text, [(s, e, "pmid:1")], [])
    # 正解のうち cc dd が本来の出典、ee が別の出典、地の文の aa も一致
    report = fake_report(text, [(text.index("cc"), text.index("dd") + 2, 0), (text.index("ee"), text.index("ee") + 2, 1), (0, 2, 1)], ["pmid:1", "pmid:9"])
    scored = score_report(report, case)
    trial = scored["trials"][0]
    assert trial["tokens"] == 4
    assert trial["coverage"] == 0.75 and trial["correct_coverage"] == 0.5
    assert scored["filler_tokens"] == 4 and scored["filler_matched"] == 1
    summary = summarize([scored])
    assert summary["detected"] == 1 and summary["correct_source_rate"] == 1.0
    assert summary["precision"] == 0.75 and summary["filler_match_rate"] == 0.25


def test_score_semantic_hit_needs_the_right_source():
    text = "aa bb cc dd"
    case = Case(text, [(0, 5, "pmid:1")], [])
    wrong = SimpleNamespace(q_start=0, q_end=5, doc=SimpleNamespace(ext_id="pmid:9"))
    outside = SimpleNamespace(q_start=6, q_end=11, doc=SimpleNamespace(ext_id="pmid:9"))
    scored = score_report(fake_report(text, [], [], [wrong, outside]), case)
    assert not scored["trials"][0]["semantic_correct"] and scored["trials"][0]["semantic_any"]
    assert scored["filler_semantic_hits"] == 1
    right = SimpleNamespace(q_start=0, q_end=5, doc=SimpleNamespace(ext_id="pmid:1"))
    assert score_report(fake_report(text, [], [], [right]), case)["trials"][0]["semantic_correct"]


def test_wilson_interval_bounds():
    low, high = wilson_interval(100, 100)
    assert 0.96 < low < 1.0 and high == pytest.approx(1.0)
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def checked(store, tmp_path, condition_name, source=SOURCE):
    text = apply_condition(CONDITION_BY_NAME[condition_name], source, "k")
    case = compose_case([FILLER, FILLER], [(text, "pmid:1")])
    scored = run_case(case, store, CheckOptions(use_semantic=False), str(tmp_path))
    assert list(tmp_path.glob("tmp*.txt")) == []  # 本文を残さない
    return scored


def test_end_to_end_verbatim_is_detected_with_the_right_source(store, tmp_path):
    scored = checked(store, tmp_path, "verbatim_30")
    assert scored["trials"][0]["coverage"] == 1.0
    assert scored["trials"][0]["correct_coverage"] == 1.0
    assert scored["filler_matched"] == 0


def test_end_to_end_filler_alone_has_no_match(store, tmp_path):
    case = Case("\n\n".join([FILLER, FILLER]), [], [])
    scored = run_case(case, store, CheckOptions(use_semantic=False), str(tmp_path))
    assert scored["filler_matched"] == 0 and scored["trials"] == []


def test_end_to_end_excluded_source_is_not_counted(store, tmp_path):
    case = Case(SOURCE, [], ["pmid:1"])
    scored = run_case(case, store, CheckOptions(use_semantic=False), str(tmp_path))
    assert scored["filler_matched"] == 0


def test_end_to_end_handled_character_tricks_are_detected(store, tmp_path):
    for name in ("homoglyph_cyrillic", "zero_width", "fullwidth", "math_bold", "special_spaces"):
        assert checked(store, tmp_path, name)["trials"][0]["coverage"] == 1.0, name


def test_retraction_formats_round_trip(tmp_path):
    path = tmp_path / "retraction.csv"
    title = "Oral statin use and the risk of primary open angle glaucoma in older adults"
    path.write_text(
        "Title,RetractionDate,OriginalPaperDOI,OriginalPaperPubMedID,RetractionNature\n"
        f"{title},3/10/2020 0:00,10.1001/retracted.1,12345678,Retraction\n",
        encoding="utf-8",
    )
    data = parse_retractions(path)
    entry = data.by_doi["10.1001/retracted.1"]
    rng = random.Random(0)
    for name, make in FORMATS.items():
        by_id, by_title, found = find_retracted(as_document([make(entry, rng)]), data)
        assert found
        hit = bool(by_id or by_title)
        # 省略と誤りは今の照合では見つからないはずなので、合否は評価の側で数える
        if name in ("DOIあり", "DOIのURL", "DOIが行末で折り返し", "PMIDのみ", "題名のみ", "題名が行末で折り返し", "題名が小文字"):
            assert hit, name


def test_retraction_evaluation_counts_control_flags(store, tmp_path):
    path = tmp_path / "retraction.csv"
    path.write_text(
        "Title,RetractionDate,OriginalPaperDOI,OriginalPaperPubMedID,RetractionNature\n"
        + "".join(
            f"Retracted study number {i} about statin use in older adults,3/10/2020 0:00,10.1001/r.{i},,Retraction\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    for split in ("dev", "test"):
        result = evaluate_retraction(store, split, parse_retractions(path))
        assert result["formats"]["DOIあり"]["rate"] in (None, 1.0)
        if result["control"]["DOIあり"]["n"]:
            assert result["control"]["DOIあり"]["flagged"] == 0


def test_render_markdown_shows_each_group():
    cell = summarize([{
        "trials": [{"coverage": 1.0, "correct_coverage": 1.0, "semantic_correct": False}], "matched_in_span": 4,
        "filler_tokens": 10, "filler_matched": 1, "filler_semantic_hits": 0, "flags": [], "score": 0.1,
    }], 1.0)
    cell.update(condition="verbatim_8", group="そのままの転用", stratum="pubmed", semantic=False, mean_score=0.1)
    control = dict(cell, condition="control_filler", group="対照", stratum="-", semantic=True)
    results = {
        "split": "dev", "finished_at": "now", "min_words": 8, "semantic_threshold": 0.85,
        "index_stats": {"docs": {"pubmed": 1}}, "cells": {"verbatim_8/pubmed": cell, "control_filler/-": control},
        "retraction": {
            "formats": {"DOIあり": {"n": 1, "rate": 1.0, "ci": (0.5, 1.0)}},
            "control": {"DOIあり": {"n": 1, "flagged": 0, "per_1000": 0.0, "ci": (0.0, 0.5)}},
        },
    }
    text = render_markdown(results, {"finished_at": "before", "cells": {"verbatim_8/pubmed": {"detected_rate": 0.5}}})
    assert "そのままの転用" in text and "地の文のみ" in text and "撤回論文のチェック" in text
    assert "+50.0" in text
    assert tokenize(text).tokens
