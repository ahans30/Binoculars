import csv
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fontTools.ttLib import TTFont
from lxml import etree

from simcheck.align import align, build_query_index
from simcheck.engine import CheckOptions, check_document
from simcheck import engine, integrity, pmc, pubmed, semantic
from simcheck.flags import Flag, check_docx, check_pdf, check_text
from simcheck.integrity import check_integrity, find_retracted, find_tortured, parse_retractions, parse_tortured
from simcheck.citations import (
    _save_citations, build_title_index, check_citations, coupled_documents, index_pubmed_citations,
    reference_keys, reset_citations,
)
from simcheck.pdfcheck import check_rendered_text
from simcheck.pdfreport import render_pdf
from simcheck.pubmed import reference_ids
from simcheck.report import render_html
from simcheck.sections import quote_spans, spans_of, strip_references
from simcheck.store import Doc, DocRow, Store, compress_text
from simcheck.textnorm import fingerprints, normalize, shingle_hashes, tokenize, winnow

SOURCE_EN = (
    "Age-related macular degeneration is a leading cause of irreversible vision loss among older adults "
    "in developed countries, and anti-vascular endothelial growth factor therapy has transformed the "
    "management of the neovascular form of this disease over the past two decades of clinical practice."
)
SOURCE_JA = (
    "加齢黄斑変性は高齢者における不可逆的な視力低下の主要な原因であり、抗血管内皮増殖因子療法によって"
    "滲出型の治療成績は過去二十年の間に大きく改善したことが多くの臨床研究で報告されている。"
)
UNRELATED = (
    "Volcanic soils in the northern islands retain water differently depending on their mineral composition, "
    "which influences how farmers schedule irrigation throughout the short growing season every year."
)


@pytest.fixture(autouse=True)
def no_integrity_data(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "TORTURED_PATH", tmp_path / "no_tortured.csv")
    monkeypatch.setattr(integrity, "RETRACTION_PATH", tmp_path / "no_retraction.csv")


def add_doc(store, ext_id, text, source="pubmed", preprint=False, name=None):
    fps = fingerprints(text)
    store.add_batch(
        name or ext_id.replace(":", "_"), source,
        [DocRow(ext_id, source, ext_id, "2024", None, preprint, {}, compress_text(text))],
        np.fromiter(fps, dtype=np.uint64), np.zeros(len(fps), dtype=np.uint32),
    )


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE_EN)
        add_doc(s, "pmid:2", SOURCE_JA)
        add_doc(s, "pmid:3", UNRELATED)
        yield s


def run(store, tmp_path, text, **kwargs):
    path = tmp_path / "manuscript.txt"
    path.write_text(text, encoding="utf-8")
    return check_document(path, store, CheckOptions(use_semantic=False, **kwargs))


def test_normalize_keeps_offsets_for_highlight():
    raw = "ﬁbrosis of the ophthal-\nmology ward"
    t = tokenize(raw)
    assert t.tokens == ["fibrosis", "of", "the", "ophthalmology", "ward"]
    assert raw[t.starts[0]:t.ends[0]] == "ﬁbrosis"
    assert raw[t.starts[3]:t.ends[3]] == "ophthal-\nmology"


def test_normalize_defeats_homoglyphs_and_zero_width():
    assert normalize("mасulаr")[0] == "macular"
    assert normalize("mac​ular")[0] == "macular"


def test_shingle_length_scales_for_japanese():
    assert len(shingle_hashes(tokenize("one two three four"))) == 0
    assert len(shingle_hashes(tokenize("one two three four five"))) == 1
    assert len(shingle_hashes(tokenize("加齢黄斑変性は高齢者におけ"))) == 1


def test_winnow_guarantees_shared_fingerprint_for_ten_word_match():
    shared = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    a = fingerprints(f"completely different opening words here now {shared} and another ending appears afterwards here")
    b = fingerprints(f"nothing similar at all in this start {shared} with some other closing text following it")
    assert a & b


def test_winnow_is_deterministic_and_handles_short_input():
    assert winnow([3, 1, 2]) == {1, 2, 3}
    assert winnow([5, 4, 3, 2, 1, 6, 7, 8]) == winnow([5, 4, 3, 2, 1, 6, 7, 8])


def test_align_merges_small_edits():
    src = tokenize(SOURCE_EN)
    edited = SOURCE_EN.replace("leading cause", "major cause")
    q = tokenize(edited)
    matches = align(q, src, build_query_index(q))
    assert len(matches) == 1
    assert matches[0].q_end - matches[0].q_start == len(q)
    assert 0.9 < matches[0].similarity < 1.0


def test_sections_and_references():
    text = "Introduction\n" + "body text. " * 300 + "\nReferences\n1. Smith J. Some paper.\nFigure legends\nFig 1 shows."
    refs = spans_of(text, "references")
    assert len(refs) == 1
    assert "Smith" in text[refs[0][0]:refs[0][1]]
    assert "Fig 1" not in text[refs[0][0]:refs[0][1]]
    stripped = strip_references(text)
    assert "Smith" not in stripped and "Fig 1" in stripped


def test_spaced_japanese_heading_is_detected():
    text = "本文。" * 1000 + "\n文　献\n1) 山田太郎: 眼科の論文. 2020.\n"
    assert "山田太郎" not in strip_references(text)


def test_remove_source_keeps_other_sources(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE_EN)
        add_doc(s, "sha256:x", SOURCE_JA, source="library", name="lib1")
        s.remove_source("library")
        assert s.stats()["docs"] == {"pubmed": 1}
        assert s.stats()["segments"] == 1
        assert run(s, tmp_path, SOURCE_JA).sources == []
        assert run(s, tmp_path, SOURCE_EN).sources


def test_tables_after_references_stay_in_scope():
    text = "本文。" * 1000 + "\nReferences\n1. Smith J. A paper.\nTable 1. Baseline characteristics\nage 50\n"
    stripped = strip_references(text)
    assert "Smith" not in stripped and "Baseline characteristics" in stripped


def test_docx_tables_keep_document_order(tmp_path):
    import docx

    from simcheck.extract import extract_document

    path = tmp_path / "m.docx"
    d = docx.Document()
    d.add_paragraph("first paragraph")
    d.add_table(rows=1, cols=1).cell(0, 0).text = "cell text"
    d.add_paragraph("last paragraph")
    d.save(path)
    text = extract_document(path).text
    assert text.index("first") < text.index("cell text") < text.index("last")


def test_common_phrase_is_excluded_from_score(tmp_path):
    licence = "this article is licensed under a creative commons attribution license which permits use and sharing"
    with Store(tmp_path / "index") as s:
        for i in range(12):
            add_doc(s, f"pmid:{i}", f"Unique opening number {i} about topic {i * 7}. {licence}. Ending {i} varies {i * 3}.")
        add_doc(s, "pmid:99", SOURCE_EN)
        text = f"{licence}. {SOURCE_EN}"
        excluded = run(s, tmp_path, text, exclude_common=True)
        included = run(s, tmp_path, text, exclude_common=False)
    assert [x.doc.ext_id for x in excluded.sources] == ["pmid:99"]
    assert excluded.score == pytest.approx(1.0)
    assert len(included.sources) > 1
    assert any(label == "定型表現" for _, _, label in excluded.excluded_spans)


def test_references_heading_early_in_long_doc_is_ignored():
    text = "References\n" + "body text. " * 300
    assert spans_of(text, "references") == []


def test_quote_spans():
    assert len(quote_spans("he said “this is a quoted sentence of some length” today")) == 1
    assert len(quote_spans("「これは十分な長さの引用文である」と述べた")) == 1
    assert quote_spans("the “short” word") == []


def test_exact_copy_scores_about_half_and_finds_source(store, tmp_path):
    report = run(store, tmp_path, SOURCE_EN + "\n\n"
                 "Entirely original sentences describe our new cohort of thirty patients followed for several years "
                 "at a single referral centre, whose outcomes we summarise here without reference to earlier work "
                 "and with a deliberately unusual vocabulary chosen for this test of the scoring denominator.")
    assert [s.doc.ext_id for s in report.sources] == ["pmid:1"]
    assert 0.45 < report.score < 0.65


def test_japanese_copy_is_detected(store, tmp_path):
    report = run(store, tmp_path, SOURCE_JA)
    assert report.sources[0].doc.ext_id == "pmid:2"
    assert report.score > 0.95


def test_original_text_scores_zero(store, tmp_path):
    report = run(store, tmp_path, "Completely novel wording about marine biology and deep ocean currents, "
                 "with nothing shared with any indexed document whatsoever in this particular small test collection.")
    assert report.score == 0
    assert report.sources == []


def test_references_are_excluded_by_default(store, tmp_path):
    padding = "Fresh original narrative about local hospital logistics and staffing patterns over time. " * 40
    text = f"{padding}\nReferences\n{SOURCE_EN}"
    assert run(store, tmp_path, text).score == 0
    assert run(store, tmp_path, text, exclude_references=False).score > 0


def test_quotes_are_excluded_by_default(store, tmp_path):
    text = f"As reviewed, “{SOURCE_EN}” and we agree."
    assert run(store, tmp_path, text).score == 0
    assert run(store, tmp_path, text, exclude_quotes=False).score > 0.8


def test_min_words_filters_short_matches(store, tmp_path):
    fragment = " ".join(SOURCE_EN.split()[:12])
    text = f"Our own distinct opening statement goes here today. {fragment}. Our own distinct closing statement."
    assert run(store, tmp_path, text, min_words=10).sources
    assert not run(store, tmp_path, text, min_words=30).sources


def test_preprint_filter(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:9", SOURCE_EN, preprint=True)
        assert run(s, tmp_path, SOURCE_EN).sources
        assert not run(s, tmp_path, SOURCE_EN, exclude_preprints=True).sources


def test_identical_file_in_library_is_excluded(tmp_path):
    from simcheck.extract import sha256_file

    path = tmp_path / "paper.txt"
    path.write_text(SOURCE_EN, encoding="utf-8")
    with Store(tmp_path / "index") as s:
        add_doc(s, f"sha256:{sha256_file(path)}", SOURCE_EN, source="library", name="self")
        report = check_document(path, s, CheckOptions(use_semantic=False))
    assert report.score == 0
    assert any("除外" in n for n in report.notices)


def test_overlapping_sources_are_not_double_counted(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE_EN)
        add_doc(s, "pmid:2", SOURCE_EN)
        report = run(s, tmp_path, SOURCE_EN)
    assert report.score == pytest.approx(1.0)
    assert sorted(x.unique_weight > 0 for x in report.sources) == [False, True]


def test_homoglyph_evasion_is_still_matched_and_flagged(store, tmp_path):
    evaded = SOURCE_EN.replace("macular", "mаculаr").replace("the", "t​he")
    report = run(store, tmp_path, evaded)
    assert report.score > 0.95
    assert {"文字の置き換え", "不可視文字"} <= {f.kind for f in report.flags}


@pytest.mark.parametrize("evaded", [
    "mαcular dεgεnεratiοn ιn οur study",  # ギリシャ文字
    "macսlar degeneratiօn iո oսr stսdy",  # アルメニア文字
    "mac️ular degen️eration in our study",  # 異体字セレクタ
    "mac\U000e0061ular degen\U000e0061eration in our study",  # タグ文字
    "mac‭‬ular degen‭‬eration in our study",  # 双方向制御文字
    "mac᠎ular degen᠎eration in our study",  # モンゴル母音区切り
    "mac͏ular degen͏eration in our study",  # 結合用字
])
def test_character_tricks_are_normalized(evaded):
    assert tokenize(evaded).tokens == ["macular", "degeneration", "in", "our", "study"]


@pytest.mark.parametrize("legitimate,expected", [
    ("TGFα and IFNγ with αSMA", ["tgfα", "and", "ifnγ", "with", "αsma"]),
    ("β-blocker in anti-αvβ3 mice", ["β", "blocker", "in", "anti", "αvβ3", "mice"]),
    ("interferonα level", ["interferonα", "level"]),
    ("日本語のα波", ["日", "本", "語", "の", "α", "波"]),
])
def test_scientific_greek_letters_are_kept_and_not_flagged(legitimate, expected):
    assert tokenize(legitimate).tokens == expected
    assert check_text(legitimate) == []


def test_one_greek_word_is_not_flagged_but_many_are():
    assert check_text("The τau protein was measured.") == []
    assert [f.kind for f in check_text("mαcular dεgεnεratiοn ιn οur study")] == ["文字の置き換え"]


def test_compact_preserves_lookup(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE_EN)
        add_doc(s, "pmid:2", SOURCE_JA)
        before = {d for d, _ in s.lookup(fingerprints(SOURCE_EN), 10)}
        s.compact()
        assert s.stats()["segments"] == 1
        assert {d for d, _ in s.lookup(fingerprints(SOURCE_EN), 10)} == before
        assert run(s, tmp_path, SOURCE_JA).sources[0].doc.ext_id == "pmid:2"


def test_common_fingerprints_are_ignored_for_candidates(tmp_path):
    with Store(tmp_path / "index") as s:
        for i in range(5):
            add_doc(s, f"pmid:{i}", SOURCE_EN)
        assert s.lookup(fingerprints(SOURCE_EN), 10, max_df=3) == []
        assert len(s.lookup(fingerprints(SOURCE_EN), 10, max_df=10)) == 5


def test_report_escapes_html(store, tmp_path):
    report = run(store, tmp_path, "<script>alert(1)</script> " + SOURCE_EN)
    rendered = render_html(report, standalone=True)
    assert "<script>" not in rendered
    assert "<mark" in rendered


def test_pdf_flags_hidden_text(tmp_path):
    import pymupdf

    path = tmp_path / "hidden.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Visible body text of the manuscript goes here.")
    page.insert_text((72, 120), "white hidden keywords stuffed here", color=(1, 1, 1))
    page.insert_text((72, 160), "tiny hidden keywords stuffed here", fontsize=1)
    doc.save(path)
    doc.close()
    assert {"白文字", "極小文字"} <= {f.kind for f in check_pdf(path)}


def test_clean_pdf_has_no_flags(tmp_path):
    import pymupdf

    path = tmp_path / "clean.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "An ordinary manuscript page with normal visible text only.")
    doc.save(path)
    doc.close()
    assert check_pdf(path) == []


def test_docx_flags_hidden_run(tmp_path):
    import docx

    path = tmp_path / "hidden.docx"
    d = docx.Document()
    run = d.add_paragraph().add_run("hidden keywords stuffed in the document")
    run.font.hidden = True
    d.save(path)
    assert "隠し文字" in {f.kind for f in check_docx(path)}


def test_check_text_clean():
    assert check_text("An ordinary sentence with a Greek β-blocker and 日本語.") == []


def _highlights(page):
    return [d for d in page.get_drawings() if d.get("fill") and d.get("fill_opacity", 1) < 0.5]


def test_pdf_report_structure_and_highlights(store, tmp_path):
    import pymupdf
    from simcheck.pdfreport import render_pdf

    path = tmp_path / "manuscript.pdf"
    src = pymupdf.open()
    page = src.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 520, 400), "Bridges along the coastal highway were repainted last spring after inspectors noted minor corrosion on several steel railings." + " " + SOURCE_EN, fontsize=11)
    src.save(path)

    with_store = check_document(path, store, CheckOptions(use_semantic=False))
    assert with_store.sources
    out = pymupdf.open("pdf", render_pdf(with_store, path))
    assert "文書の詳細" in out[0].get_text()
    assert "全体的な類似性" in out[1].get_text()
    assert "PubMed抄録" in out[2].get_text()
    manuscript = out[3]
    highlights = _highlights(manuscript)
    assert highlights
    # 重ねた位置に一致した本文があり、無関係な文にはかからない
    covered = " ".join(manuscript.get_text(clip=d["rect"]) for d in highlights)
    assert "macular" in covered and "Bridges" not in covered
    assert "1/4" in out[0].get_text() and "4/4" in manuscript.get_text()


def test_pdf_report_falls_back_without_word(store, tmp_path, monkeypatch):
    import docx
    import pymupdf
    from simcheck import pdfreport

    monkeypatch.setattr(pdfreport, "_word_to_pdf", lambda path, out: False)
    path = tmp_path / "manuscript.docx"
    document = docx.Document()
    document.add_paragraph(SOURCE_EN)
    document.save(path)

    report = check_document(path, store, CheckOptions(use_semantic=False))
    out = pymupdf.open("pdf", pdfreport.render_pdf(report, path))
    assert "組み直したもの" in out[0].get_text()
    assert _highlights(out[3])


def test_docx_separates_superscript_citations(tmp_path):
    import docx
    from simcheck.extract import extract_document

    path = tmp_path / "manuscript.docx"
    document = docx.Document()
    paragraph = document.add_paragraph("The prevalence was 38.8% in 2019")
    paragraph.add_run("4,6,7").font.superscript = True
    paragraph.add_run(". Treatment changed")
    document.save(path)

    tokens = tokenize(extract_document(path).text).tokens
    assert "2019" in tokens and "20194" not in tokens
    assert tokens[-2:] == ["treatment", "changed"]


def test_short_match_found_by_default(store, tmp_path):
    # 既定の最小語数以上の一致は、原稿側を間引かず検索するので見つかる
    add_doc(store, "pmid:9", UNRELATED + " the ninth cranial nerve carries taste fibers from posterior tongue regions.")
    text = (
        "Bridges along the coastal highway were repainted last spring. "
        "the ninth cranial nerve carries taste fibers from posterior tongue regions. "
        "Inspectors noted minor corrosion on several steel railings afterwards."
    )
    assert any(s.doc.ext_id == "pmid:9" for s in run(store, tmp_path, text).sources)


def tortured_list(tmp_path, *rows):
    path = tmp_path / "tortured.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fingerprint - Tortured Phrase", "Expected Text", "Nb Retrieved Papers"])
        writer.writerows((query, expected, "1") for query, expected in rows)
    return path


def retraction_csv(tmp_path, *rows):
    path = tmp_path / "retraction.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "RetractionDate", "OriginalPaperDOI", "OriginalPaperPubMedID", "RetractionNature"])
        writer.writerows(rows)
    return path


LONG_TITLE = "Oral statin use and the risk of primary open angle glaucoma in adults"


def phrases(text, list_path):
    return [hit.phrase for hit in find_tortured(text, parse_tortured(list_path))]


def test_tortured_query_forms(tmp_path):
    path = tortured_list(
        tmp_path,
        ('"neural organization"', "neural network"),
        ('"bosom" AND "breast"', "breast"),
        ('"magnetic cell separation" NOT "with MACS"', "MACS"),
        ('"gatherings clustering"~3', "clustering"),
        ('"yield layer" AND neural', "output layer"),
        ('"enhancement calculation" AND optimiz*', "optimization algorithm"),
        ('"PC helped" AND ("be that as it" OR "on the off chance")', "computer aided"),
        ('"energy productivity”', "energy efficiency"),
    )
    assert parse_tortured(path).n_skipped == 0
    assert phrases("A Neural Organization was built.", path) == ["neural organization"]
    assert phrases("Bosom tissue.", path) == []
    assert phrases("Bosom tissue in breast cancer.", path) == ["bosom"]
    assert phrases("Magnetic cell separation was used.", path) == ["magnetic cell separation"]
    assert phrases("Magnetic cell separation with MACS was used.", path) == []
    assert phrases("Gatherings of data for clustering.", path) == ["gatherings clustering"]
    assert phrases("Gatherings " + "word " * 20 + "clustering.", path) == []
    assert phrases("The yield layer of a neural model.", path) == ["yield layer"]
    assert phrases("Enhancement calculation for optimization.", path) == ["enhancement calculation"]
    assert phrases("PC helped design, be that as it may.", path) == ["pc helped"]
    assert phrases("PC helped design.", path) == []
    assert phrases("Energy productivity rose.", path) == ["energy productivity"]


def test_tortured_query_that_cannot_be_anchored_is_skipped(tmp_path):
    path = tortured_list(tmp_path, ('("a b" OR "c d") AND "e f"', "x"), ('"a b" OR "c d"', "x"))
    assert parse_tortured(path).n_loaded == 0
    assert parse_tortured(path).n_skipped == 2


def test_tortured_phrase_does_not_cross_sentences_or_headings(tmp_path):
    path = tortured_list(tmp_path, ('"preparation data"', "training data"))
    assert phrases("Sample preparation data were kept.", path) == ["preparation data"]
    assert phrases("Sample preparation\ndata were kept.", path) == ["preparation data"]
    assert phrases("Sample preparation. Data were kept.", path) == []
    assert phrases("Sample preparation, data were kept.", path) == []
    assert phrases("Sample preparation\nData collection", path) == []
    assert phrases("Sample preparation\n\ndata collection", path) == []


def test_tortured_ignores_references(tmp_path):
    path = tortured_list(tmp_path, ('"neural organization"', "neural network"))
    text = "Body text.\n\nReferences\n1. Smith J. Neural organization of cortex. Brain. 2001."
    assert phrases(text, path) == []


def test_one_tortured_phrase_is_only_a_notice(tmp_path, monkeypatch):
    path = tortured_list(tmp_path, ('"neural organization"', "neural network"), ('"bosom peril"', "breast cancer"))
    monkeypatch.setattr(integrity, "TORTURED_PATH", path)
    flags, notices = check_integrity("The neural organization of cognition.")
    assert flags == []
    assert any("neural organization" in n for n in notices)
    flags, _ = check_integrity("The neural organization of bosom peril. A neural organization.")
    assert [f.kind for f in flags] == ["不自然な語句"]
    assert flags[0].count == 3
    assert flags[0].examples[0] == "neural organization → neural network（2回）"


def retraction_data(tmp_path):
    return parse_retractions(retraction_csv(
        tmp_path,
        (LONG_TITLE, "3/10/2020 0:00", "10.1001/retracted.1", "0", "Retraction"),
        ("Paper known only by its pubmed identifier here", "1/2/2021 0:00", "unavailable", "12345678", "Retraction"),
        ("Paper with an expression of concern about its data", "1/2/2021 0:00", "10.1001/concern.2", "", "Expression of concern"),
        ("Paper that only received a small correction later", "1/2/2021 0:00", "10.1001/corrected.3", "", "Correction"),
        ("Paper that was retracted and then later reinstated", "1/2/2021 0:00", "10.1001/back.4", "", "Retraction"),
        ("Paper that was retracted and then later reinstated", "1/2/2022 0:00", "10.1001/back.4", "", "Reinstatement"),
        ("Short retracted title", "1/2/2021 0:00", "10.1001/short.5", "", "Retraction"),
    ))


def test_retraction_match_by_doi_and_pmid(tmp_path):
    data = retraction_data(tmp_path)
    text = (
        "Body.\n\nReferences\n1. A. Something else. https://doi.org/10.1001/RETRACTED.1.\n"
        "2. B. Other. PMID: 12345678\n3. C. Other. doi:10.1001/concern.2\n"
        "4. D. doi:10.1001/corrected.3\n5. E. doi:10.1001/back.4\n6. F. doi:10.1001/unrelated.6"
    )
    by_id, by_title, found = find_retracted(text, data)
    assert found and by_title == []
    assert sorted(e.status for e in by_id) == ["懸念表明", "撤回", "撤回"]
    assert next(e for e in by_id if e.doi == "10.1001/retracted.1").date == "2020-03-10"


def test_retraction_doi_wrapped_at_line_end(tmp_path):
    by_id, _, _ = find_retracted("Body.\n\nReferences\n1. A. doi:10.1001/\nretracted.1\n2. B.", retraction_data(tmp_path))
    assert [e.doi for e in by_id] == ["10.1001/retracted.1"]


def test_retraction_match_by_title(tmp_path):
    data = retraction_data(tmp_path)
    wrapped = LONG_TITLE.replace("risk of", "risk\nof").replace("open angle", "open-angle")
    _, by_title, _ = find_retracted(f"Body.\n\nReferences\n1. Lee A. {wrapped}. JAMA. 2019.", data)
    assert [e.doi for e in by_title] == ["10.1001/retracted.1"]
    # 同じ題名でも別のDOIが書かれていれば別の論文
    _, by_title, _ = find_retracted(f"Body.\n\nReferences\n1. Lee A. {LONG_TITLE}. doi:10.9999/other.1", data)
    assert by_title == []
    # DOIで見つかった論文を題名でも重ねて数えない
    by_id, by_title, _ = find_retracted(f"Body.\n\nReferences\n1. Lee A. {LONG_TITLE}. doi:10.1001/retracted.1", data)
    assert len(by_id) == 1 and by_title == []
    # 短い題名は照合しない
    _, by_title, _ = find_retracted("Body.\n\nReferences\n1. Short retracted title. 2020.", data)
    assert by_title == []


def test_retraction_title_ignored_outside_references(tmp_path):
    data = retraction_data(tmp_path)
    by_id, by_title, found = find_retracted(f"{LONG_TITLE}. See doi:10.1001/retracted.1", data)
    assert not found and by_title == []
    assert [e.doi for e in by_id] == ["10.1001/retracted.1"]


def test_missing_integrity_data_is_reported(store, tmp_path):
    report = run(store, tmp_path, UNRELATED)
    assert any("語句リストが未取得" in n for n in report.notices)
    assert any("撤回データが未取得" in n for n in report.notices)


def pdf_text(pdf) -> str:
    # 組版で折り返しや空白の種類が変わるので、空白を除いて比べる
    return re.sub(r"\s", "", "".join(page.get_text() for page in pdf))


def test_integrity_results_reach_reports(store, tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, "TORTURED_PATH", tortured_list(
        tmp_path, ('"neural organization"', "neural network"), ('"bosom peril"', "breast cancer"),
    ))
    monkeypatch.setattr(integrity, "RETRACTION_PATH", retraction_csv(
        tmp_path, (LONG_TITLE, "3/10/2020 0:00", "10.1001/retracted.1", "0", "Retraction"),
    ))
    text = f"{UNRELATED} A neural organization of bosom peril.\n\nReferences\n1. Lee A. {LONG_TITLE}. doi:10.1001/retracted.1"
    report = run(store, tmp_path, text)
    assert {f.kind for f in report.flags} == {"不自然な語句", "撤回論文の引用"}
    html = render_html(report)
    assert "neural organization → neural network" in html
    assert "10.1001/retracted.1" in html
    with pymupdf.open(stream=render_pdf(report, tmp_path / "manuscript.txt"), filetype="pdf") as pdf:
        rendered = pdf_text(pdf)
    assert "neuralorganization→neuralnetwork" in rendered
    assert "10.1001/retracted.1" in rendered
    assert "RetractionWatch" in rendered


def test_many_flags_stay_on_the_pdf_page(store, tmp_path):
    report = run(store, tmp_path, UNRELATED)
    long_example = "A very long retracted article title about something " * 3
    report.flags = [Flag(f"種類{i}", "説明 " * 40, 3, [long_example] * 8) for i in range(12)]
    with pymupdf.open(stream=render_pdf(report, tmp_path / "manuscript.txt"), filetype="pdf") as pdf:
        for page in pdf:
            assert all(block[3] <= page.rect.height for block in page.get_text("blocks"))
        assert "ほか3件はHTMLレポートを参照" in pdf_text(pdf)


def add_cited_doc(store, ext_id, title, refs, doi=None):
    fps = fingerprints(UNRELATED + title)
    store.add_batch(
        ext_id.replace(":", "_"), "pubmed",
        [DocRow(ext_id, "pubmed", title, "2024", None, False, {"doi": doi}, compress_text(UNRELATED + title))],
        np.fromiter(fps, dtype=np.uint64), np.zeros(len(fps), dtype=np.uint32),
    )
    doc_id = store.id_for_ext(ext_id)
    _save_citations(store, doc_id, refs)
    return doc_id


def manuscript_with_refs(lines):
    body = "Body text of the manuscript. " * 200
    return body + "\n\nReferences\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def test_citation_overlap_flags_shared_reference_list(tmp_path):
    refs = [f"doi:10.1000/ref.{i}" for i in range(8)]
    with Store(tmp_path / "index") as s:
        source = add_cited_doc(s, "pmid:1", "A source article about glaucoma treatment outcomes in adults", refs)
        add_cited_doc(s, "pmid:2", "An unrelated article about something else entirely here", [f"doi:10.1000/x.{i}" for i in range(8)])
        text = manuscript_with_refs([f"Author A. Title. doi:10.1000/ref.{i}" for i in range(8)])
        flags, notices = check_citations(text, s)
        assert [f.kind for f in flags] == ["引用の重なり"]
        assert flags[0].count == 1
        assert "共通 8 件" in flags[0].examples[0] and "順序の一致 100%" in flags[0].examples[0]
        assert check_citations(text, s, {source}) == ([], [])


def test_citation_order_and_threshold(tmp_path):
    refs = [f"doi:10.1000/ref.{i}" for i in range(20)]
    with Store(tmp_path / "index") as s:
        add_cited_doc(s, "pmid:1", "A source article about glaucoma treatment outcomes in adults", refs)
        few = manuscript_with_refs([f"A. T. doi:10.1000/ref.{i}" for i in range(5)] + [f"A. T. doi:10.1000/other.{i}" for i in range(15)])
        assert check_citations(few, s)[0] == []  # 共通5件 / 20件は重なりが小さい
        reordered = manuscript_with_refs([f"A. T. doi:10.1000/ref.{i}" for i in reversed(range(20))])
        result = coupled_documents(reference_keys(reordered, s), s)[0]
        assert result["coupling"] == 1.0 and result["order_share"] == pytest.approx(0.05)


def test_citation_titles_resolve_references_without_doi(tmp_path):
    title = "Oral statin use and the risk of primary open angle glaucoma in adults"
    with Store(tmp_path / "index") as s:
        add_cited_doc(s, "pmid:7", title, [], doi="10.1000/known")
        assert build_title_index(s, progress=lambda _: None) == 1
        references = f"1. Lee A. {title.replace('risk of', 'risk' + chr(10) + 'of')}. JAMA. 2019."
        assert reference_keys(references, s) == ["doi:10.1000/known"]


def test_citation_check_reports_missing_index(tmp_path):
    with Store(tmp_path / "index") as s:
        assert "index-citations" in check_citations("body", s)[1][0]


def test_pubmed_parser_reads_reference_ids():
    xml = (
        "<PubmedArticle><PubmedData><ReferenceList>"
        "<Reference><Citation>A</Citation><ArticleIdList><ArticleId IdType='pubmed'>123</ArticleId>"
        "<ArticleId IdType='doi'>10.1/a</ArticleId></ArticleIdList></Reference>"
        "<Reference><Citation>B</Citation></Reference>"
        "</ReferenceList></PubmedData></PubmedArticle>"
    )
    assert reference_ids(etree.fromstring(xml)) == [{"pmid": "123", "doi": "10.1/a"}]


def swapped_font_pdf(path, text):
    """表示と抽出が異なるPDFを作る。字形は変えず、字形から文字への対応表（ToUnicode）だけを入れ替える。"""
    source = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
    if not source.exists():
        pytest.skip("arial.ttf がない環境")
    font = TTFont(source)
    cmap = font.getBestCmap()
    letters = "abcdefghijklmnopqrstuvwxyz"
    entries = [
        f"<{font.getGlyphID(cmap[ord(a)]):04X}> <{ord(b):04X}>"
        for a, b in zip(letters, letters[13:] + letters[:13])
    ]
    stream = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CMapName /Swapped def /CMapType 2 def\n"
        "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        f"{len(entries)} beginbfchar\n" + "\n".join(entries) + "\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end\n"
    )
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontname="swapped", fontfile=str(source), fontsize=12)
    xref = page.get_fonts()[0][0]
    cmap_xref = doc.get_new_xref()
    doc.update_object(cmap_xref, "<<>>")
    doc.update_stream(cmap_xref, stream.encode())
    doc.xref_set_key(xref, "ToUnicode", f"{cmap_xref} 0 R")
    doc.save(path)
    doc.close()


NORMAL_PAGE = (
    "Glaucoma remains among leading causes irreversible blindness worldwide, although earlier diagnosis improves visual "
    "prognosis considerably. Researchers enrolled consecutive adults attending university hospitals between January "
    "through December, excluding participants having previous ocular surgery, uveitis, diabetic retinopathy, corneal "
    "disease or incomplete records. Intraocular pressure measurements were obtained using calibrated applanation "
    "tonometry before treatment and every three months afterwards. Optical coherence tomography quantified retinal "
    "nerve fiber thickness while automated perimetry recorded functional progression. Statistical analyses applied "
    "mixed regression models adjusting baseline severity, medication adherence, socioeconomic status and comorbid "
    "conditions. Sensitivity analyses repeated these calculations after multiple imputation of missing observations. "
    "Patients receiving combination therapy demonstrated slower structural deterioration compared with monotherapy, "
    "yet adverse events such as conjunctival hyperemia occurred more frequently. These findings support individualized "
    "management strategies, particularly among younger populations whose lifetime disability burden seems substantial."
)


def test_rendered_text_check_finds_font_mapping(tmp_path):
    pytest.importorskip("rapidocr")
    path = tmp_path / "swapped.pdf"
    swapped_font_pdf(path, NORMAL_PAGE)
    with pymupdf.open(path) as doc:
        assert "glaucoma" not in doc[0].get_text().lower()  # 抽出した文字は別の語になる
    flags, notices = check_rendered_text(path)
    assert notices == []
    assert [f.kind for f in flags] == ["表示と抽出の不一致"]


def test_rendered_text_check_passes_normal_pdf(tmp_path):
    pytest.importorskip("rapidocr")
    path = tmp_path / "normal.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), NORMAL_PAGE, fontsize=12)
    doc.save(path)
    doc.close()
    assert check_rendered_text(path) == ([], [])


def test_retraction_title_allows_abbreviation_and_small_errors(tmp_path):
    path = tmp_path / "retraction.csv"
    title = "Oral statin use and the risk of primary open angle glaucoma in older adults from japan"
    path.write_text(
        "Title,RetractionDate,OriginalPaperDOI,OriginalPaperPubMedID,RetractionNature\n"
        f"{title},3/10/2020 0:00,10.1001/retracted.1,,Retraction\n",
        encoding="utf-8",
    )
    data = parse_retractions(path)
    words = title.split()
    abbreviated = " ".join(words[:12])
    typo_early = " ".join(["Oral", "statins", *words[2:]])
    typo_late = " ".join([*words[:-2], "form", "japan"])
    different = " ".join(words[:7] + ["completely", "different", "study", "about", "cataract", "surgery", "patients"])
    for reference, expected in ((abbreviated, 1), (typo_early, 1), (typo_late, 1), (different, 0)):
        doc = f"{'Body text. ' * 200}\n\nReferences\n1. Lee A. {reference}. JAMA. 2019."
        assert len(find_retracted(doc, data)[1]) == expected, reference


class FakeScorer:
    def __init__(self, gain):
        self.gain = gain

    def mean_gain(self, source, target):
        return self.gain


def rerank_pairs(text):
    return [((0, len(text)), (1, 0, 20, 0.8))]


def rerank_docs(source_text):
    return {1: Doc(1, "pmid:1", "pubmed", "t", None, None, False, {}, source_text)}


def test_rerank_keeps_only_candidates_with_enough_gain(monkeypatch):
    text = "A paraphrased sentence about glaucoma."
    docs = rerank_docs("A source sentence about glaucoma treatment.")
    options = CheckOptions(rerank_min_gain=0.6)
    monkeypatch.setattr(engine, "get_scorer", lambda: FakeScorer(1.0))
    assert len(engine._rerank(rerank_pairs(text), text, docs, options, [])) == 1
    monkeypatch.setattr(engine, "get_scorer", lambda: FakeScorer(0.1))
    assert engine._rerank(rerank_pairs(text), text, docs, options, []) == []


def test_rerank_does_not_judge_other_languages(monkeypatch):
    def fail():
        raise AssertionError("言語をまたぐ組は再評価しない")

    monkeypatch.setattr(engine, "get_scorer", fail)
    english = "A translated sentence about glaucoma."
    japanese = rerank_docs("緑内障の治療についての文章です。")
    assert len(engine._rerank(rerank_pairs(english), english, japanese, CheckOptions(), [])) == 1
    both = "緑内障の治療に関する別の文章です。"
    assert engine._rerank(rerank_pairs(both), both, japanese, CheckOptions(), []) == []  # 類似度0.8 < 既定0.85


def test_rerank_falls_back_to_threshold_without_model(monkeypatch):
    def fail():
        raise OSError("model missing")

    monkeypatch.setattr(engine, "get_scorer", fail)
    text = "A paraphrased sentence about glaucoma."
    docs = rerank_docs("A source sentence about glaucoma treatment.")
    notices = []
    assert engine._rerank(rerank_pairs(text), text, docs, CheckOptions(), notices) == []
    assert "再評価に失敗" in notices[0]
    assert len(engine._rerank(rerank_pairs(text), text, docs, CheckOptions(semantic_threshold=0.7), [])) == 1


def test_re_adding_a_document_keeps_its_id_and_fingerprints(tmp_path):
    with Store(tmp_path / "fresh") as store:
        add_doc(store, "pmid:1", SOURCE_EN, name="query")
        doc_id = store.id_for_ext("pmid:1")
        fps = fingerprints(SOURCE_EN)
        assert len(fps) > 1

        # 登録済みの文献と新しい文献が混ざっていても、新しいほうだけが入る
        text = UNRELATED * 3
        other = fingerprints(text)
        hashes = np.fromiter([*fps, *other], dtype=np.uint64)
        ids = np.array([0] * len(fps) + [1] * len(other), dtype=np.uint32)
        rows = [
            DocRow("pmid:1", "pubmed", "updated", "2024", None, False, {}, compress_text(SOURCE_EN)),
            DocRow("pmid:2", "pubmed", "new", "2024", None, False, {}, compress_text(text)),
        ]
        assert store.add_batch("query", "pubmed", rows, hashes, ids) == 1

        assert store.id_for_ext("pmid:1") == doc_id
        assert store.get_docs([doc_id])[doc_id].title == "pmid:1"
        new_id = store.id_for_ext("pmid:2")
        assert new_id > doc_id
        # 同じ名前で登録しても、前回の指紋は残り、重複もしない
        found = dict(store.lookup(fps, limit=5))
        assert doc_id in found
        assert max(store.document_frequencies(fps).values()) == 1
        assert new_id in dict(store.lookup(other, limit=5))


def test_pubmed_citations_are_indexed_in_bulk_with_pmids_resolved_to_dois(tmp_path):
    with Store(tmp_path / "fresh") as store:
        refs = [{"doi": f"10.1000/X{i}"} for i in range(8)] + [{"pmid": "77"}, {"pmid": "88"}]
        rows = [
            DocRow("pmid:1", "pubmed", "citing", "2024", None, False, {"refs": refs}, compress_text(SOURCE_EN)),
            DocRow("pmid:77", "pubmed", "cited", "2024", None, False, {"doi": "10.1000/Cited"}, compress_text(UNRELATED)),
        ]
        fps = fingerprints(SOURCE_EN)
        store.add_batch(
            "batch", "pubmed", rows, np.fromiter(fps, dtype=np.uint64), np.zeros(len(fps), dtype=np.uint32),
        )
        reset_citations(store)
        result = index_pubmed_citations(store, progress=lambda _: None)
        assert result == {"pubmed_indexed": 1, "pubmed_without_refs": 1, "citations": 10}

        keys = [f"doi:10.1000/x{i}" for i in range(8)] + ["doi:10.1000/cited", "pmid:88"]
        found = coupled_documents(keys, store)[0]
        assert found["doc_id"] == store.id_for_ext("pmid:1")
        assert (found["shared"], found["source_refs"], found["order_share"]) == (10, 10, 1.0)
        assert coupled_documents(keys[:7], store) == []


def test_semantic_index_can_be_extended_without_rebuilding(tmp_path, monkeypatch):
    def fake_encode(texts):
        vectors = np.zeros((len(texts), semantic.EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, hash(text) % semantic.EMBED_DIM] = 1.0
        return vectors

    monkeypatch.setattr(semantic, "encode", fake_encode)
    with Store(tmp_path / "fresh") as store:
        add_doc(store, "pmid:1", SOURCE_EN, name="a")
        add_doc(store, "pmid:2", UNRELATED, name="b")
        add_doc(store, "pmc:PMC3", SOURCE_EN + " " + UNRELATED, source="pmc", name="c")
        first = store.id_for_ext("pmid:1")
        root = tmp_path / "part"
        semantic.build_pubmed_index(store, root, max_docs=1)
        before = np.load(root / "chunks.npy")
        assert set(before[:, 0].tolist()) == {first}

        assert semantic.matching_pubmed_docs(store, r"PMID:2\b") == [store.id_for_ext("pmid:2")]
        wanted = [first, store.id_for_ext("pmid:2"), *store.iter_doc_ids("pmc")]
        result = semantic.extend_index(store, wanted, root, progress=lambda _: None)

        after = np.load(root / "chunks.npy")
        assert result["docs"] == 2
        assert (after[:len(before)] == before).all()
        assert set(after[:, 0].tolist()) == set(wanted)
        assert semantic.faiss.read_index(str(root / "index.faiss")).ntotal == len(after)


def test_empty_batch_is_recorded_as_ingested_without_a_segment(tmp_path):
    with Store(tmp_path / "fresh") as store:
        added = store.add_batch("empty", "pubmed", [], np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32))
        assert added == 0
        assert store.is_ingested("empty")
        assert not list((tmp_path / "fresh" / "segments").glob("empty*"))


def test_baseline_file_without_abstracts_does_not_fail(tmp_path, monkeypatch):
    gz = tmp_path / "x.xml.gz"
    gz.write_bytes(b"")
    monkeypatch.setattr(pubmed, "_download", lambda name, dest: gz)
    monkeypatch.setattr(pubmed, "iter_articles", lambda path: iter(()))
    _, rows, hashes, ids = pubmed._process_file("x.xml.gz", str(tmp_path), False)
    assert rows == [] and hashes.size == 0 and ids.size == 0


def _fake_eutils(articles, seen_ranges):
    """検索式の [pdat] の範囲で articles（出版月, PMCID, 本文）を絞る E-utilities の代わり。"""

    def matching(query):
        found = re.search(r"(\d{4})/(\d{2}):(\d{4})/(\d{2})\[pdat\]", query)
        if not found:
            return articles
        y1, m1, y2, m2 = map(int, found.groups())
        return [a for a in articles if y1 * 12 + m1 - 1 <= a[0] <= y2 * 12 + m2 - 1]

    class Response:
        def __init__(self, payload=None, content=b""):
            self.payload, self.content = payload, content

        def json(self):
            return self.payload

    keys = {}

    def fake_get(client, url, params):
        if url.endswith("esearch.fcgi"):
            webenv = str(len(keys))
            keys[webenv] = matching(params["term"])
            return Response({"esearchresult": {"count": str(len(keys[webenv])), "querykey": "1", "webenv": webenv}})
        seen_ranges.append((params["retstart"], params["retmax"]))
        chosen = keys[params["WebEnv"]][params["retstart"]:params["retstart"] + params["retmax"]]
        body = "".join(
            f'<article><front><article-meta><article-id pub-id-type="pmcid">{pmcid}</article-id>'
            f"<title-group><article-title>{pmcid}</article-title></title-group></article-meta></front>"
            f"<body><p>{text}</p></body></article>"
            for _, pmcid, text in chosen
        )
        return Response(content=f"<pmc-articleset>{body}</pmc-articleset>".encode())

    return fake_get


def test_pmc_query_over_the_limit_is_split_by_publication_month(tmp_path, monkeypatch):
    articles = [
        (2020 * 12 + i, f"PMC{i}", f"{SOURCE_EN} Sample number {i} was reported by the trial group {i * 7}.")
        for i in range(5)
    ]
    ranges = []
    monkeypatch.setattr(pmc, "_get_with_retry", _fake_eutils(articles, ranges))
    monkeypatch.setattr(pmc, "MAX_QUERY_RECORDS", 2)
    monkeypatch.setattr(pmc, "QUERY_BATCH", 2)
    monkeypatch.setattr(pmc.time, "sleep", lambda _: None)
    with Store(tmp_path / "fresh") as store:
        result = pmc.add_pmc_query(store, "glaucoma", progress=lambda _: None)
        assert result["added"] == 5 and result["found"] == 5 and result["windows"] > 1
        assert all(start + size <= 2 for start, size in ranges)
        assert len(store.iter_doc_ids("pmc")) == 5
        name = "pmc_query_" + hashlib.sha256(b"glaucoma").hexdigest()[:16]
        assert store.is_ingested(name)

        # 途中で止まった場合に当たる再実行では、登録済みの範囲を取り直さない
        ranges.clear()
        again = pmc.add_pmc_query(store, "glaucoma", progress=lambda _: None)
        assert again["added"] == 0 and ranges == []


def test_pmc_query_never_asks_past_the_eutilities_limit(tmp_path, monkeypatch):
    articles = [(2020 * 12, f"PMC{i}", f"{SOURCE_EN} Sample number {i} was reported by group {i * 7}.") for i in range(3)]
    ranges = []
    monkeypatch.setattr(pmc, "_get_with_retry", _fake_eutils(articles, ranges))
    monkeypatch.setattr(pmc, "MAX_QUERY_RECORDS", 3)
    monkeypatch.setattr(pmc, "QUERY_BATCH", 2)
    monkeypatch.setattr(pmc.time, "sleep", lambda _: None)
    with Store(tmp_path / "fresh") as store:
        assert pmc.add_pmc_query(store, "x", progress=lambda _: None)["added"] == 3
    assert ranges == [(0, 2), (2, 1)]
