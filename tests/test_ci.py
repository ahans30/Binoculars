"""GitHub Actions 版で使う部分のテスト。"""
import json

import docx
import httpx
import numpy as np
import pymupdf
import pytest
from lxml import etree

from simcheck import pdfreport, triage
from simcheck.corpus import read_queries
from simcheck.engine import CheckOptions, check_document
from simcheck.pmc import parse_article
from simcheck.report import corpus_summary, render_html
from simcheck.store import DocRow, Store, compress_text
from simcheck.textnorm import fingerprints
from simcheck.triage import ATTEMPTS, TriageItem, triage_report

SOURCE = (
    "Age-related macular degeneration is a leading cause of irreversible vision loss among older adults "
    "in developed countries, and anti-vascular endothelial growth factor therapy has transformed the "
    "management of the neovascular form of this disease over the past two decades of clinical practice."
)
ORIGINAL = "Bridges along the coastal highway were repainted last spring by the county maintenance crew."
SECRET = "Secret subscription preface sentence that must stay on this computer and never leave it at all."

PMC_XML = b"""<pmc-articleset><article article-type="research-article" xmlns:xlink="http://www.w3.org/1999/xlink"
 xmlns:ali="http://www.niso.org/schemas/ali/1.0/">
<front><journal-meta><journal-title-group><journal-title>J Test</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="pmcid">PMC123</article-id><article-id pub-id-type="doi">10.1/x</article-id>
<title-group><article-title>A <italic>study</italic> title</article-title></title-group>
<pub-date><year>2025</year></pub-date>
<permissions><license><ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref></license></permissions>
<abstract><p>Abstract sentence here.</p></abstract></article-meta></front>
<body><sec><title>Introduction</title><p>Seasonal influenza is common in Japan<xref ref-type="bibr" rid="r1"><sup>1</sup></xref> and rising.</p>
<table-wrap><table><tr><td>hidden table value</td></tr></table></table-wrap></sec></body>
<back><ref-list><ref><mixed-citation>Reference text must not appear</mixed-citation></ref></ref-list></back>
</article></pmc-articleset>"""


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


def check(store, tmp_path, text):
    path = tmp_path / "manuscript.txt"
    path.write_text(text, encoding="utf-8")
    return check_document(path, store, CheckOptions(use_semantic=False)), path


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def reply(results):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"results": results})}]}}]}


def test_pmc_article_keeps_body_only():
    record = parse_article(etree.fromstring(PMC_XML)[0])
    assert record["pmcid"] == "PMC123" and record["year"] == "2025" and record["journal"] == "J Test"
    assert record["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert record["title"] == "A study title"
    assert "Seasonal influenza is common in Japan and rising." in record["text"]
    assert "Abstract sentence here." in record["text"]
    assert "Reference text" not in record["text"] and "hidden table" not in record["text"]


def test_pmc_article_without_body_is_skipped():
    article = etree.fromstring(PMC_XML)[0]
    article.remove(article.find("body"))
    assert parse_article(article) is None


def test_read_queries(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text('# comment\n\npubmed: "a b"[Title/Abstract]\npmc: c: d\n', encoding="utf-8")
    assert read_queries(path) == [("pubmed", '"a b"[Title/Abstract]'), ("pmc", "c: d")]
    path.write_text("web: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_queries(path)


def test_project_queries_are_valid():
    assert read_queries("corpus/queries.txt")


def test_corpus_summary():
    assert corpus_summary({"pubmed": 3, "pmc": 2}) == "PubMed抄録 3 件、PMC全文 2 件"
    assert corpus_summary({"pubmed": 3, "library": 1}) == "PubMed抄録 3 件、手元のPDF 1 件"


def test_triage_sends_only_manuscript_spans_and_titles(store, tmp_path):
    add_doc(store, "lib:1", SECRET + " " + SOURCE, source="library")
    report, _ = check(store, tmp_path, ORIGINAL + " " + SOURCE)
    sent = []

    def handler(request):
        sent.append(request)
        items = json.loads(json.loads(request.content)["contents"][0]["parts"][0]["text"])["items"]
        return httpx.Response(200, json=reply([{"id": i["id"], "label": "review", "reason": "内容に固有"} for i in items]))

    items, notices = triage_report(report, api_key="test-key", client=client(handler))
    assert notices == []
    assert items and all(i.label == "review" and i.reason == "内容に固有" for i in items)
    assert sent[0].headers["x-goog-api-key"] == "test-key"
    assert "test-key" not in str(sent[0].url)
    body = sent[0].content.decode("utf-8")
    assert "Secret subscription" not in body and "repainted" not in body
    assert "macular" in body


def test_triage_request_uses_documented_format(store, tmp_path):
    report, _ = check(store, tmp_path, SOURCE)
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json=reply([]))

    triage_report(report, api_key="k", model="some-model", client=client(handler))
    assert sent[0].url.path == "/v1beta/models/some-model:generateContent"
    response_format = json.loads(sent[0].content)["generationConfig"]["responseFormat"]["text"]
    assert response_format["mimeType"] == "APPLICATION_JSON" and response_format["schema"]["type"] == "object"


def test_triage_without_key_is_reported(store, tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    report, _ = check(store, tmp_path, SOURCE)
    items, notices = triage_report(report)
    assert items == [] and "未設定" in notices[0]


def test_triage_failure_keeps_report_and_hides_response(store, tmp_path, monkeypatch):
    monkeypatch.setattr(triage.time, "sleep", lambda s: None)
    report, _ = check(store, tmp_path, SOURCE)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="server error with secret manuscript text")

    items, notices = triage_report(report, api_key="k", client=client(handler))
    assert len(calls) == ATTEMPTS
    assert items and all(i.label is None for i in items)
    assert "仕分けできません" in notices[0] and "secret" not in notices[0]


def test_triage_recovers_from_rate_limit(store, tmp_path, monkeypatch):
    monkeypatch.setattr(triage.time, "sleep", lambda s: None)
    report, _ = check(store, tmp_path, SOURCE)
    answers = [httpx.Response(429), httpx.Response(200, json=reply([{"id": 0, "label": "boilerplate", "reason": "定型"}]))]
    items, notices = triage_report(report, api_key="k", client=client(lambda r: answers.pop(0)))
    assert notices == [] and items[0].label == "boilerplate"


def test_triage_ignores_invalid_results(store, tmp_path):
    report, _ = check(store, tmp_path, SOURCE)
    bad = reply([{"id": 0, "label": "innocent", "reason": "x"}, {"id": 999, "label": "review", "reason": "y"}])
    items, _ = triage_report(report, api_key="k", client=client(lambda r: httpx.Response(200, json=bad)))
    assert items and all(i.label is None for i in items)


def test_triage_reaches_reports_without_windows_fonts(store, tmp_path, monkeypatch):
    # Linux のランナーには Yu Gothic などがないので、組み込みのフォントで描けることを確かめる
    monkeypatch.setattr(pdfreport, "_FONT_DIR", tmp_path / "no-fonts")
    report, path = check(store, tmp_path, "日本語の注記を含む文書です。\n" + SOURCE)
    s, e, rank = report.owned_spans[0]
    report.triage = [TriageItem(s, e, rank, "boilerplate", "標準的な表現"), TriageItem(s, e, rank, None, "")]
    out = pymupdf.open("pdf", pdfreport.render_pdf(report, path))
    text = "".join(page.get_text() for page in out)
    assert "Geminiによる一致箇所の仕分け" in text
    assert "定型句" in text and "未分類" in text and "標準的な表現" in text
    assert "Geminiによる一致箇所の仕分け" in render_html(report)


def test_soffice_conversion_is_used_when_word_is_unavailable(store, tmp_path, monkeypatch):
    pdf = pymupdf.open()
    pdf.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 400), SOURCE, fontsize=11)
    converted = tmp_path / "converted.pdf"
    pdf.save(converted)
    monkeypatch.setattr(pdfreport, "_word_to_pdf", lambda path, out: False)
    monkeypatch.setattr(pdfreport, "_soffice_to_pdf", lambda path: converted)
    path = tmp_path / "manuscript.docx"
    document = docx.Document()
    document.add_paragraph(SOURCE)
    document.save(path)
    report = check_document(path, store, CheckOptions(use_semantic=False))
    cover = pymupdf.open("pdf", pdfreport.render_pdf(report, path))[0].get_text()
    assert "LibreOffice" in cover and "組み直した" not in cover
