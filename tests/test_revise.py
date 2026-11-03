import json

import docx
import httpx
import numpy as np
import pytest
from docx.text.run import Run

from simcheck import revise
from simcheck.docx_comments import Comment, write_comments
from simcheck.engine import CheckOptions, check_document
from simcheck.extract import extract_document, paragraph_pieces
from simcheck.revise import (
    LLMUnavailable, available_model, build_suggestions, check_candidate, find_targets, residual_words, sentence_ranges,
)
from simcheck.store import DocRow, Store, compress_text
from simcheck.textnorm import fingerprints

SOURCE = (
    "Age-related macular degeneration is a leading cause of irreversible vision loss among older adults "
    "in developed countries, and anti-vascular endothelial growth factor therapy has transformed the "
    "management of the neovascular form of this disease over the past two decades of clinical practice."
)
ORIGINAL = "Bridges along the coastal highway were repainted last spring by the county maintenance crew."
FACT = "The prevalence was 48.7% in 2017, and it did not increase in the NDB sample of 1,000 patients 3,4."


def add_doc(store, ext_id, text, source="pubmed"):
    fps = fingerprints(text)
    store.add_batch(
        ext_id.replace(":", "_"), source,
        [DocRow(ext_id, source, f"Title of {ext_id}", "2024", None, False, {}, compress_text(text))],
        np.fromiter(fps, dtype=np.uint64), np.zeros(len(fps), dtype=np.uint32),
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE)
        yield s


def report_of(store, tmp_path, text):
    path = tmp_path / "manuscript.txt"
    path.write_text(text, encoding="utf-8")
    return check_document(path, store, CheckOptions(use_semantic=False))


def llm(results, seen=None):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "text-embedding-x"}, {"id": "local-model"}]})
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(results)},
        }]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sentence_ranges_split_lines_and_sentences():
    text = "Results\nThe rate was 3.5% in Japan. It rose later.\fNext page text"
    assert [text[s:e] for s, e in sentence_ranges(text)] == [
        "Results", "The rate was 3.5% in Japan.", "It rose later.", "Next page text",
    ]


def test_find_targets_returns_whole_sentence_with_source(store, tmp_path):
    report = report_of(store, tmp_path, f"{ORIGINAL} {SOURCE} {ORIGINAL}")
    targets = find_targets(report)
    assert len(targets) == 1
    assert targets[0].sentence == SOURCE
    assert targets[0].before == ORIGINAL and targets[0].after == ORIGINAL
    assert targets[0].source.ext_id == "pmid:1"
    assert "macular" in targets[0].excerpt


def test_faithful_candidate_passes():
    candidate = "In the NDB sample of 1,000 patients 3,4, the prevalence did not increase, standing at 48.7% in 2017."
    check = check_candidate(FACT, candidate)
    assert check.passed and check.problems() == []


@pytest.mark.parametrize("candidate, expected", [
    ("The prevalence was 47.8% in 2017, and it did not increase in the NDB sample of 1,000 patients 3,4.", "数値"),
    ("The prevalence was 48.7% in 2017, and it did not increase in the NDB sample of patients 3,4.", "数値"),
    ("The prevalence was 48.7% in 2017, and it did not increase in the NDB sample of 1,000 patients.", "数値"),
    ("The prevalence was 48.7% in 2017 in 2018, and it did not increase in the NDB sample of 1,000 patients 3,4.", "増えて"),
    ("The prevalence was 48.7% in 2017, and it did not increase in the sample of 1,000 patients 3,4.", "略語"),
    ("The prevalence was 48.7% in 2017, and it did increase in the NDB sample of 1,000 patients 3,4.", "否定"),
    ("The prevalence was 48.7% in 2017, and it did not decrease in the NDB sample of 1,000 patients 3,4.", "方向"),
    ("The prevalence was significantly 48.7% in 2017, and it did not increase in the NDB sample of 1,000 patients 3,4.", "有意"),
    ("有病率は2017年に48.7%で、NDBの1,000人 3,4 では増加しなかった。", "言語"),
])
def test_altered_candidates_are_caught(candidate, expected):
    check = check_candidate(FACT, candidate)
    assert not check.passed
    assert any(expected in problem for problem in check.problems())


def test_low_similarity_fails_and_unknown_similarity_does_not():
    assert not check_candidate(FACT, FACT, similarity=0.5).passed
    assert check_candidate(FACT, FACT, similarity=None).passed


def test_residual_and_source_run_are_measured(store):
    assert residual_words(SOURCE, store) > 30
    assert residual_words(ORIGINAL, store) == 0
    doc = store.get_docs([next(iter(store.iter_doc_ids("pubmed")))])
    source = next(iter(doc.values()))
    assert check_candidate(SOURCE, SOURCE, store, source).longest_source_run > 30


def test_build_suggestions_orders_failed_candidates_last(store, tmp_path, monkeypatch):
    monkeypatch.setattr(revise, "similarities", lambda original, texts: [None] * len(texts))
    report = report_of(store, tmp_path, SOURCE)
    seen = []
    bad = SOURCE.replace("two decades", "2020 decades")
    good = (
        "Over the past two decades of clinical practice, anti-vascular endothelial growth factor therapy has changed "
        "how the neovascular form is managed, and age-related macular degeneration remains a leading cause of "
        "irreversible vision loss among older adults in developed countries."
    )
    results = {"action": "rewrite", "reason": "表現が近い", "candidates": [bad, good, good, SOURCE]}
    suggestions, model = build_suggestions(report, store, client=llm(results, seen))
    assert model == "local-model"
    assert seen[0]["model"] == "local-model"
    assert "response_format" in seen[0]
    suggestion = suggestions[0]
    assert suggestion.action == "rewrite" and suggestion.reason == "表現が近い"
    # 重複と元の文そのままの案は除き、検査に通らない案は後ろに回す
    assert [c.text for c in suggestion.candidates] == [good, bad]
    assert suggestion.candidates[0].check.passed and not suggestion.candidates[1].check.passed


def test_invalid_llm_answer_becomes_an_error_for_that_sentence(store, tmp_path, monkeypatch):
    report = report_of(store, tmp_path, SOURCE)
    suggestions, _ = build_suggestions(report, store, model="m", client=llm({"action": "delete", "reason": "", "candidates": []}))
    assert suggestions[0].action is None and "作れませんでした" in suggestions[0].error
    assert suggestions[0].candidates == []


def test_unavailable_server_is_explained():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable, match="LM Studio"):
        available_model("http://127.0.0.1:1234/v1", httpx.Client(transport=httpx.MockTransport(handler)))


def manuscript(tmp_path):
    document = docx.Document()
    document.add_paragraph("Introduction")
    paragraph = document.add_paragraph("First sentence stays. The rate was high in 2019")
    paragraph.add_run("4,6").font.superscript = True
    bold = paragraph.add_run(". Later sentence is bold.")
    bold.bold = True
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "A sentence in a table cell."
    path = tmp_path / "manuscript.docx"
    document.save(path)
    return path


def commented_text(document):
    """コメントの範囲の文字列を、コメント本文ごとに返す。"""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    result = {}
    for comment in document.comments:
        cid = str(comment.comment_id)
        for paragraph in document.element.body.iter(f"{ns}p"):
            collecting, parts = False, []
            for node in paragraph.iter():
                if node.tag == f"{ns}commentRangeStart" and node.get(f"{ns}id") == cid:
                    collecting = True
                elif node.tag == f"{ns}commentRangeEnd" and node.get(f"{ns}id") == cid:
                    collecting = False
                elif collecting and node.tag == f"{ns}t":
                    parts.append(node.text)
            if parts:
                result[comment.text] = "".join(parts)
    return result


def test_comment_covers_only_the_sentence_and_keeps_body(tmp_path):
    path = manuscript(tmp_path)
    before_text = extract_document(path).text
    before_bytes = path.read_bytes()
    out = tmp_path / "out" / "commented.docx"
    result = write_comments(path, [
        Comment("The rate was high in 2019 4,6 .", "案A"),
        Comment("Later sentence is bold.", "案B"),
        Comment("A sentence in a table cell.", "案C"),
    ], out)
    assert result["written"] == 3 and result["missed"] == []
    assert path.read_bytes() == before_bytes
    assert extract_document(out).text == before_text
    document = docx.Document(out)
    ranges = commented_text(document)
    assert ranges["案A"] == "The rate was high in 20194,6."
    assert ranges["案B"] == "Later sentence is bold."
    assert ranges["案C"] == "A sentence in a table cell."
    # 分割しても書式が残る
    runs = document.paragraphs[1].runs
    assert any(r.font.superscript and r.text == "4,6" for r in runs)
    assert all(r.bold for r in runs if "Later" in r.text)
    assert not any(r.bold for r in runs if "First" in r.text)


def test_repeated_sentence_gets_separate_anchors_and_missing_is_reported(tmp_path):
    document = docx.Document()
    document.add_paragraph("Same sentence here.")
    document.add_paragraph("Same sentence here.")
    path = tmp_path / "manuscript.docx"
    document.save(path)
    result = write_comments(path, [
        Comment("Same sentence here.", "1"), Comment("Same sentence here.", "2"), Comment("Not in the document.", "3"),
    ], tmp_path / "out.docx")
    assert result["written"] == 2 and [c.text for c in result["missed"]] == ["3"]
    ranges = commented_text(docx.Document(tmp_path / "out.docx"))
    assert set(ranges) == {"1", "2"}


def test_comments_refuse_to_overwrite_source(tmp_path):
    path = manuscript(tmp_path)
    with pytest.raises(ValueError, match="書き込みません"):
        write_comments(path, [], path)
    with pytest.raises(ValueError, match="DOCX"):
        write_comments(tmp_path / "manuscript.pdf", [], tmp_path / "out.docx")


def test_paragraph_pieces_match_extracted_text(tmp_path):
    path = manuscript(tmp_path)
    document = docx.Document(path)
    pieces, text = paragraph_pieces(document.paragraphs[1])
    assert text == "First sentence stays. The rate was high in 2019 4,6 . Later sentence is bold."
    for item, start, end in pieces:
        assert isinstance(item, Run) and text[start:end] == item.text


def test_decision_comments_state_citation_duty_and_ai_origin(store, tmp_path):
    from simcheck.revise import CandidateCheck, Suggestion, decision_comment

    target = find_targets(report_of(store, tmp_path, SOURCE))[0]
    suggestion = Suggestion(target, "rewrite", "表現が近い")
    adopted = decision_comment(suggestion, "adopt", "New wording.", CandidateCheck(), "local-model")
    assert "New wording." in adopted and "local-model" in adopted and "引用が必要" in adopted
    assert "Title of pmid:1" in decision_comment(suggestion, "cite")
    assert "引用符" in decision_comment(suggestion, "quote")
    assert "自分の論文" in decision_comment(suggestion, "own")
    assert decision_comment(suggestion, "keep") is None and decision_comment(suggestion, "none") is None


def test_harmless_rewording_is_not_flagged():
    assert check_candidate("There was a difference between the 2 observers in round 4.",
                           "The two observers differed in round four.").passed
    assert check_candidate("RESULTS: Age was 66 years in NTG.", "Age was 66 years in NTG.").passed
    assert check_candidate("Vision improved in 960 eyes of the 960 eyes.", "Vision increased in all 960 eyes.").passed
    assert check_candidate("The layer became thinner in rats.", "The layer showed thinning in rats.").passed


def test_added_direction_is_flagged_but_rephrased_direction_is_not():
    assert check_candidate("Vision was poor at month 5.", "Vision declined by month 5.").direction_changed
    assert not check_candidate("Thickness decreased by 5 um.", "Thickness showed a reduction of 5 um.").direction_changed


def test_llm_cannot_decide_to_keep(store, tmp_path, monkeypatch):
    report = report_of(store, tmp_path, SOURCE)
    answer = {"action": "keep", "reason": "引用不要", "candidates": []}
    suggestions, _ = build_suggestions(report, store, model="m", client=llm(answer))
    assert suggestions[0].action is None and suggestions[0].reason == ""


def test_keep_comes_from_corpus_frequency_without_calling_llm(tmp_path):
    from simcheck.revise import is_common_wording

    licence = "this article is licensed under a creative commons attribution license which permits use and sharing"
    with Store(tmp_path / "index") as s:
        for i in range(12):
            add_doc(s, f"pmid:{i}", f"Unique opening number {i} about topic {i * 7}. {licence}. Ending {i} varies {i * 3}.")
        add_doc(s, "pmid:99", SOURCE)
        assert is_common_wording(licence, s)
        assert not is_common_wording(SOURCE, s)
        assert not is_common_wording(ORIGINAL, s)
        calls = []
        report = report_of(s, tmp_path, f"{licence}.")
        suggestions, _ = build_suggestions(report, s, model="m", client=llm({"action": "rewrite"}, calls))
    assert suggestions[0].action == "keep" and calls == []


def test_placeholders_are_removed_from_candidates(store, tmp_path, monkeypatch):
    monkeypatch.setattr(revise, "similarities", lambda original, texts: [None] * len(texts))
    report = report_of(store, tmp_path, SOURCE)
    answer = {"action": "cite", "reason": "引用が必要", "candidates": ["A wording of the sentence [Ref]. ", "Another wording [source 1]."]}
    suggestions, _ = build_suggestions(report, store, model="m", client=llm(answer))
    assert [c.text for c in suggestions[0].candidates] == ["A wording of the sentence.", "Another wording."]
