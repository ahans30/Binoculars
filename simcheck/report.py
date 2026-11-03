import html
from pathlib import Path

from .engine import Report, SourceResult

# 色だけに頼らないよう、ハイライトには出典番号も併記する
_PALETTE = ["#fecaca", "#bfdbfe", "#bbf7d0", "#fde68a", "#ddd6fe", "#fbcfe8", "#a5f3fc", "#fed7aa", "#d9f99d", "#e2e8f0"]
_DETAIL_LIMIT = 3000

_STYLE = """
<style>
.simcheck{--ink:#1f2933;--muted:#5f6c7b;--line:#d9dee5;--card:#f7f8fa;--bg:#fff;--semantic:#b45309;
  color:var(--ink);background:var(--bg);font-family:"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;font-size:14px;line-height:1.6;padding:16px;border-radius:8px}
.simcheck h2{font-size:17px;margin:22px 0 8px;color:var(--ink)}
.simcheck .cards{display:flex;flex-wrap:wrap;gap:10px}
.simcheck .card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px;min-width:120px}
.simcheck .card b{display:block;font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.simcheck .card span{color:var(--muted);font-size:12px}
.simcheck .caveat{border-left:3px solid var(--semantic);background:var(--card);padding:8px 12px;margin:12px 0;font-size:13px}
.simcheck table{border-collapse:collapse;width:100%;font-size:13px}
.simcheck th,.simcheck td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;color:var(--ink)}
.simcheck td.num,.simcheck th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.simcheck .chip{display:inline-block;min-width:20px;text-align:center;border-radius:4px;padding:0 5px;font-weight:600;color:#1f2933}
.simcheck .doc{white-space:pre-wrap;overflow-wrap:anywhere;border:1px solid var(--line);border-radius:6px;padding:14px;max-height:640px;overflow:auto;line-height:1.8}
.simcheck mark{color:#1f2933;border-radius:2px;padding:0 1px}
.simcheck sup.n{font-size:10px;font-weight:700;margin-right:2px;color:#1f2933}
.simcheck .ex{color:var(--muted);text-decoration:line-through;text-decoration-color:var(--line)}
.simcheck .para{border-bottom:2px dotted var(--semantic)}
.simcheck .page{display:block;color:var(--muted);font-size:11px;text-align:center;border-top:1px dashed var(--line);margin:8px 0}
.simcheck details{border:1px solid var(--line);border-radius:6px;margin:6px 0;padding:6px 10px}
.simcheck summary{cursor:pointer}
.simcheck .pair{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.simcheck .pair div{background:var(--card);border-radius:4px;padding:8px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
.simcheck .pair h4{margin:0 0 4px;font-size:12px;color:var(--muted);font-weight:600}
.simcheck small{color:var(--muted)}
@media (max-width:640px){.simcheck .pair{grid-template-columns:1fr}}
</style>
"""


def _e(text) -> str:
    return html.escape(str(text))


def _clip(text: str) -> str:
    return text if len(text) <= _DETAIL_LIMIT else text[:_DETAIL_LIMIT] + " …（省略）"


def _color(rank: int) -> str:
    return _PALETTE[rank % len(_PALETTE)]


def _source_label(source: SourceResult) -> str:
    doc = source.doc
    if doc.source == "pubmed":
        journal = doc.meta.get("journal") or ""
        return f"PubMed {doc.ext_id.removeprefix('pmid:')} · {journal} {doc.year or ''}".strip()
    if doc.source == "pmc":
        journal = doc.meta.get("journal") or ""
        return f"{doc.ext_id.removeprefix('pmc:')} · {journal} {doc.year or ''}".strip()
    return "手元のPDF"


def corpus_summary(docs: dict) -> str:
    parts = [f"PubMed抄録 {docs.get('pubmed', 0):,} 件"]
    if docs.get("pmc"):
        parts.append(f"PMC全文 {docs['pmc']:,} 件")
    if docs.get("library") or not docs.get("pmc"):
        parts.append(f"手元のPDF {docs.get('library', 0):,} 件")
    return "、".join(parts)


def _source_title_html(doc) -> str:
    title = _e(doc.title or doc.ext_id)
    badges = ""
    if doc.is_preprint:
        badges += " <small>[プレプリント]</small>"
    if doc.meta.get("retracted"):
        badges += " <small>[撤回論文]</small>"
    if doc.url:
        return f'<a href="{_e(doc.url)}" target="_blank" rel="noreferrer">{title}</a>{badges}'
    return title + badges


def _page_of(text: str, position: int) -> int:
    return text.count("\f", 0, position) + 1


def _render_document(report: Report) -> str:
    text = report.text
    events = []
    for s, e, rank in report.owned_spans:
        events.append((s, e, "own", rank))
    for s, e, _ in report.excluded_spans:
        events.append((s, e, "ex", None))
    for hit in report.semantic_hits:
        events.append((hit.q_start, hit.q_end, "para", None))
    # 同じ開始位置なら一致のハイライトを優先する
    priority = {"own": 0, "ex": 1, "para": 2}
    events.sort(key=lambda ev: (ev[0], priority[ev[2]]))

    parts, pos = [], 0

    def plain(chunk: str) -> str:
        return _e(chunk).replace("\f", '<span class="page">— 改ページ —</span>')

    for s, e, kind, rank in events:
        s = max(s, pos)
        if s >= e:
            continue
        parts.append(plain(text[pos:s]))
        body = plain(text[s:e])
        if kind == "own":
            parts.append(f'<mark style="background:{_color(rank)}"><sup class="n">{rank + 1}</sup>{body}</mark>')
        elif kind == "ex":
            parts.append(f'<span class="ex">{body}</span>')
        else:
            parts.append(f'<span class="para">{body}</span>')
        pos = e
    parts.append(plain(text[pos:]))
    return "".join(parts)


def render_html(report: Report, standalone: bool = False) -> str:
    score_pct = f"{report.score * 100:.1f}%"
    docs = report.index_stats.get("docs", {})
    body = [_STYLE, '<div class="simcheck">']
    body.append(f"<h2>類似度レポート: {_e(report.file_name)}</h2>")
    body.append('<div class="cards">')
    for value, label in (
        (score_pct, "類似度"),
        (f"{len(report.sources)}", "一致した出典"),
        (f"{report.words_matched:,.0f}", "一致語数"),
        (f"{report.words_checked:,.0f}", "確認対象の語数"),
        (f"{report.words_excluded:,.0f}", "除外した語数"),
        (f"{len(report.semantic_hits)}", "言い換えの疑い"),
        (f"{len(report.flags)}", "フラッグ"),
    ):
        body.append(f'<div class="card"><b>{value}</b><span>{label}</span></div>')
    body.append("</div>")

    body.append(
        '<div class="caveat">照合先は '
        f"{corpus_summary(docs)}のみです。"
        "ここにない文献（購読誌の本文、Webページなど）との一致は検出できないため、低い値でも盗用がないことの証明にはなりません。"
        "語数は日本語を2.5文字=1語として換算しています。</div>"
    )
    for notice in report.notices:
        body.append(f'<div class="caveat">{_e(notice)}</div>')

    if report.flags:
        body.append(
            "<h2>フラッグ</h2><small>自動検出した確認候補で、不正の証拠ではありません。"
            "図の中の文字やPDFの作成方法が原因で出ることもあります。</small>"
        )
        body.append("<table><tr><th>種類</th><th>内容</th><th class=num>件数</th><th>例</th></tr>")
        for flag in report.flags:
            examples = "<br>".join(_e(x) for x in flag.examples)
            body.append(
                f"<tr><td>{_e(flag.kind)}</td><td>{_e(flag.message)}</td>"
                f"<td class=num>{flag.count:,}</td><td>{examples}</td></tr>"
            )
        body.append("</table>")

    body.append("<h2>一致した出典</h2>")
    if report.sources:
        total = sum(1 for _ in report.sources)
        denominator = report.words_checked or 1
        body.append(
            "<table><tr><th>#</th><th>出典</th><th>種類</th><th class=num>寄与</th>"
            "<th class=num>一致全体</th><th class=num>箇所</th><th>ID</th></tr>"
        )
        for rank, source in enumerate(report.sources):
            contribution = source.unique_weight / 5 / denominator * 100
            overall = source.total_weight / 5 / denominator * 100
            body.append(
                f'<tr><td><span class="chip" style="background:{_color(rank)}">{rank + 1}</span></td>'
                f"<td>{_source_title_html(source.doc)}</td><td>{_e(_source_label(source))}</td>"
                f"<td class=num>{contribution:.1f}%</td><td class=num>{overall:.1f}%</td>"
                f"<td class=num>{len(source.matches)}</td><td><small>{_e(source.doc.ext_id[:24])}</small></td></tr>"
            )
        body.append(f"</table><small>寄与は重複を除いた割合で、合計が類似度になります（出典 {total} 件）。</small>")
    else:
        body.append("<p>設定した最小語数以上の一致はありませんでした。</p>")

    body.append("<h2>原稿のハイライト</h2>")
    body.append(
        "<small>色付き=出典との一致(番号は出典の#) / 取り消し線=除外範囲 / 点線=言い換えの疑い(類似度には含みません)</small>"
    )
    body.append(f'<div class="doc">{_render_document(report)}</div>')

    if report.sources:
        body.append("<h2>一致箇所の比較</h2>")
        for rank, source in enumerate(report.sources):
            doc_text = source.doc.text
            body.append(
                f'<details><summary><span class="chip" style="background:{_color(rank)}">{rank + 1}</span> '
                f"{_e(source.doc.title or source.doc.ext_id)} "
                f"<small>({len(source.matches)} 箇所)</small></summary>"
            )
            for match in source.matches:
                location = {"library": f"p.{_page_of(doc_text, match.s_start)}", "pmc": "全文"}.get(source.doc.source, "抄録")
                body.append(
                    f"<small>{match.words:.0f}語 / 一致率 {match.similarity * 100:.0f}% / "
                    f"原稿 p.{_page_of(report.text, match.q_start)} · 出典 {location}</small>"
                    '<div class="pair">'
                    f"<div><h4>原稿</h4>{_e(_clip(report.text[match.q_start:match.q_end]))}</div>"
                    f"<div><h4>出典</h4>{_e(_clip(doc_text[match.s_start:match.s_end]))}</div>"
                    "</div>"
                )
            body.append("</details>")

    if report.triage:
        body.append("<h2>Geminiによる一致箇所の仕分け</h2>")
        body.append("<small>確認の順番を決めるための補助で、精度は評価していません。類似度の値には影響しません。</small>")
        body.append("<table><tr><th>仕分け</th><th>原稿</th><th>理由</th><th>#</th></tr>")
        for item in report.triage:
            label = {"boilerplate": "定型句", "review": "要確認"}.get(item.label, "未分類")
            body.append(
                f"<tr><td>{label}</td><td>{_e(_clip(report.text[item.start:item.end]))}</td>"
                f"<td>{_e(item.reason)}</td><td>{item.rank + 1}</td></tr>"
            )
        body.append("</table>")

    if report.semantic_hits:
        body.append("<h2>言い換えの疑い</h2>")
        body.append(
            "<small>意味が近い文の候補です。一般的な表現でも挙がるので、機械的に盗用と判断せず、内容を確認してください。</small>"
        )
        for hit in sorted(report.semantic_hits, key=lambda h: -h.score):
            body.append(
                f"<details><summary>類似 {hit.score:.2f} · {_e(hit.doc.title or hit.doc.ext_id)}</summary>"
                '<div class="pair">'
                f"<div><h4>原稿</h4>{_e(_clip(report.text[hit.q_start:hit.q_end]))}</div>"
                f"<div><h4>出典 p.{_page_of(hit.doc.text, hit.s_start)}</h4>{_e(_clip(hit.doc.text[hit.s_start:hit.s_end]))}</div>"
                "</div></details>"
            )

    body.append("</div>")
    content = "".join(body)
    if standalone:
        return (
            '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>類似度レポート</title></head><body>{content}</body></html>"
        )
    return content


def write_html(report: Report, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report, standalone=True), encoding="utf-8")
    return path
