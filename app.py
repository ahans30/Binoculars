import gc
import html
import tempfile
from pathlib import Path

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from simcheck import DATA_DIR
from simcheck.docx_comments import Comment, write_comments
from simcheck.engine import CheckOptions, check_document
from simcheck.pdfreport import write_pdf
from simcheck.report import render_html, write_html
from simcheck.revise import (
    ACTIONS, DECISIONS, DEFAULT_URL as REVISE_URL, LLMUnavailable, build_suggestions, check_candidate,
    decision_comment, similarities, source_reference,
)
from simcheck.sections import strip_references
from simcheck.semantic import DEFAULT_THRESHOLD
from simcheck.store import Store

current_bino = None
# 長い文書は1トークンずつ重ねて分割し、全トークンを評価する
MAX_WINDOW_TOKENS = 512
MIN_WINDOW_TOKENS_FOR_SCORE = 64


class BinocularsDetector:
    def __init__(self, observer_path: str, performer_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(observer_path, trust_remote_code=True)
        performer_tokenizer = AutoTokenizer.from_pretrained(performer_path, trust_remote_code=True)
        if self.tokenizer.get_vocab() != performer_tokenizer.get_vocab():
            raise ValueError("Observer と Performer のトークナイザー（語彙）が一致しません。同一ファミリーのモデルを指定してください。")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map="auto" だと複数GPU環境で2モデルが別デバイスに載る可能性があるため、明示的に1デバイスへ配置する
        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_path, dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_path, dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()

    def _token_spans(self, ids, losses):
        # 日本語などはバイト単位BPEで1文字が複数トークンに分かれ、単独デコードすると「�」になるため、
        # 文字として確定するまでトークンをまとめて1つのスパンにする
        spans, buf_ids, buf_losses = [], [], []
        for tid, loss in zip(ids, losses):
            buf_ids.append(tid)
            buf_losses.append(loss)
            piece = self.tokenizer.decode(buf_ids)
            if not piece.endswith("�"):
                spans.append((piece, sum(buf_losses) / len(buf_losses)))
                buf_ids, buf_losses = [], []
        if buf_ids:
            spans.append((self.tokenizer.decode(buf_ids), sum(buf_losses) / len(buf_losses)))
        return spans

    def evaluate(self, text: str):
        if not text.strip():
            return "テキストを入力してください。", ""

        all_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]

        if all_ids.shape[0] < 5:
            return "テキストが短すぎます（最低5トークン以上必要です）。", ""

        sum_ppl = sum_cross = 0.0
        label_ids, loss_values, window_scores = [], [], []
        with torch.no_grad():
            for start in range(0, all_ids.shape[0] - 1, MAX_WINDOW_TOKENS - 1):
                input_ids = all_ids[start:start + MAX_WINDOW_TOKENS].unsqueeze(0).to(self.device)

                # bf16 のまま softmax / log_softmax を取ると精度が落ちるので float32 で計算する
                obs_logits = self.observer(input_ids).logits[:, :-1, :].float()
                perf_logits = self.performer(input_ids).logits[:, :-1, :].float()
                labels = input_ids[:, 1:]

                # パディングで語彙サイズが異なる場合に備えて共通部分に揃える
                vocab = min(obs_logits.shape[-1], perf_logits.shape[-1])
                obs_logits = obs_logits[..., :vocab]
                perf_logits = perf_logits[..., :vocab]

                # 1. log PPL（論文の公式実装と同じく Performer で計算）
                log_probs_perf = torch.nn.functional.log_softmax(perf_logits, dim=-1)
                target_log_probs = log_probs_perf.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                token_losses = -target_log_probs[0]

                # 2. log Cross-PPL（Observer の分布 × Performer の対数確率）
                probs_obs = torch.nn.functional.softmax(obs_logits, dim=-1)
                cross_entropy = -(probs_obs * log_probs_perf).sum(dim=-1)

                sum_ppl += token_losses.sum().item()
                sum_cross += cross_entropy.sum().item()
                label_ids += labels[0].tolist()
                loss_values += token_losses.tolist()
                if labels.shape[1] >= MIN_WINDOW_TOKENS_FOR_SCORE:
                    window_scores.append(token_losses.mean().item() / cross_entropy.mean().item())

        log_ppl = sum_ppl / len(label_ids)
        log_cross_ppl = sum_cross / len(label_ids)
        score = log_ppl / log_cross_ppl

        highlighted_html = '<div style="line-height: 2.2; font-family: monospace; font-size: 14px;">'

        for token, loss_val in self._token_spans(label_ids, loss_values):
            safe_token = html.escape(token).replace("\n", "<br>")
            if loss_val < 1.5:
                bg = "#c6f6d5"
            elif loss_val < 3.5:
                bg = "#e6fffa"
            elif loss_val < 6.0:
                bg = "#feebc8"
            else:
                bg = "#fed7d7"

            highlighted_html += f'<span style="background-color: {bg}; color: #1a202c; padding: 2px 4px; margin: 1px; border-radius: 3px;" title="負の対数尤度: {loss_val:.2f}">{safe_token}</span>'

        highlighted_html += '</div>'
        highlighted_html += (
            '<div style="margin-top: 10px; font-size: 12px;">'
            '凡例: <span style="background-color: #c6f6d5; color: #1a202c; padding: 2px 4px;">緑: 予測通り</span> '
            '<span style="background-color: #feebc8; color: #1a202c; padding: 2px 4px;">黄: やや意外</span> '
            '<span style="background-color: #fed7d7; color: #1a202c; padding: 2px 4px;">赤: 予想外</span>'
            '</div>'
        )

        detail = (
            f"スコア: {score:.5f}\n"
            f"平均 log PPL: {log_ppl:.3f}\n"
            f"平均 log Cross-PPL: {log_cross_ppl:.3f}\n"
            f"評価トークン数: {len(label_ids)}"
        )
        if len(window_scores) > 1:
            listed = ", ".join(f"{s:.3f}" for s in window_scores)
            detail += f"\n区間別スコア（{MAX_WINDOW_TOKENS}トークンごと）: {listed}"
        return detail, highlighted_html


def load_models(observer_path, performer_path):
    global current_bino
    # 再ロード時に古いモデルが VRAM に残り続けないよう先に解放する
    current_bino = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        current_bino = BinocularsDetector(observer_path.strip(), performer_path.strip())
        return f"モデルロード完了 ({current_bino.device}):\n{observer_path}\n{performer_path}"
    except Exception as e:
        return f"ロードエラー: {e}"


def run_prediction(text):
    if current_bino is None:
        return "先にモデルをロードしてください。", ""
    return current_bino.evaluate(text)


def run_similarity(file_path, exclude_references, exclude_quotes, exclude_methods, exclude_preprints,
                   exclude_common, min_words, use_semantic, semantic_threshold, exclude_ids):
    if not file_path:
        return "ファイルを指定してください。", "", None, gr.skip()
    options = CheckOptions(
        exclude_references=exclude_references,
        exclude_quotes=exclude_quotes,
        exclude_methods=exclude_methods,
        exclude_preprints=exclude_preprints,
        exclude_common=exclude_common,
        min_words=int(min_words),
        use_semantic=use_semantic,
        semantic_threshold=semantic_threshold,
        exclude_ext_ids={x.strip() for x in exclude_ids.splitlines() if x.strip()},
    )
    with Store() as store:
        report = check_document(file_path, store, options)
    out = Path(tempfile.mkdtemp()) / f"{Path(file_path).stem}_report.html"
    write_html(report, out)
    pdf = write_pdf(report, file_path, out.with_suffix(".pdf"))
    summary = (
        f"類似度 {report.score * 100:.1f}% / 一致した出典 {len(report.sources)} 件 / "
        f"言い換えの疑い {len(report.semantic_hits)} 件 / フラッグ {len(report.flags)} 件"
    )
    # 参考文献を除いた本文をスコアのタブへ渡す
    return summary, render_html(report), [str(pdf), str(out)], strip_references(report.text).replace("\f", "\n")


def _revise_view(state):
    """state の現在の文を、画面の各部品の値にする。"""
    empty = ("", "", "", "", gr.update(choices=[], value=None), "", "", DECISIONS["none"])
    if not state or not state["suggestions"]:
        return ("修正案はありません。", *empty[1:])
    index = state["index"]
    suggestion = state["suggestions"][index]
    target = suggestion.target
    decided = sum(1 for d in state["decisions"].values() if d["decision"] != "none")
    status = f"{index + 1} / {len(state['suggestions'])} 文（決定済み {decided}）・モデル: {state['model']}"
    source = f"{source_reference(target.source)}\n\n{target.excerpt}"
    if suggestion.error:
        advice = suggestion.error
    else:
        advice = f"提案: {ACTIONS[suggestion.action]}\n{suggestion.reason}"
    saved = state["decisions"].get(index, {})
    choices = [(f"案{i + 1}: {c.text}", i) for i, c in enumerate(suggestion.candidates)]
    selected = saved.get("candidate")
    if selected is None and choices:
        selected = 0
    edit = saved.get("text") or (suggestion.candidates[selected].text if choices else "")
    checks = suggestion.candidates[selected].check.summary() if choices else ""
    return (
        status, target.sentence, source, advice, gr.update(choices=choices, value=selected), checks, edit,
        DECISIONS[saved.get("decision", "none")],
    )


def run_revise(file_path, url, model, progress=gr.Progress()):
    if not file_path:
        return (None, "ファイルを指定してください。", *_revise_view(None)[1:])
    with Store() as store:
        report = check_document(file_path, store, CheckOptions(use_semantic=False))
        try:
            suggestions, used = build_suggestions(
                report, store, url.strip(), model.strip() or None,
                progress=lambda i, n: progress(i / max(n, 1), desc=f"{i + 1} / {n} 文"),
            )
        except LLMUnavailable as e:
            return (None, str(e), *_revise_view(None)[1:])
    state = {"file": file_path, "suggestions": suggestions, "decisions": {}, "index": 0, "model": used}
    return (state, *_revise_view(state))


def select_candidate(state, candidate):
    if not state or candidate is None:
        return gr.skip(), gr.skip()
    chosen = state["suggestions"][state["index"]].candidates[candidate]
    return chosen.text, chosen.check.summary()


def _record_decision(state, decision_label, candidate, text):
    decision = next(key for key, label in DECISIONS.items() if label == decision_label)
    state["decisions"][state["index"]] = {"decision": decision, "candidate": candidate, "text": text.strip()}


def save_and_move(state, decision_label, candidate, text, step):
    if not state or not state["suggestions"]:
        return (state, *_revise_view(state))
    _record_decision(state, decision_label, candidate, text)
    state["index"] = min(max(state["index"] + step, 0), len(state["suggestions"]) - 1)
    return (state, *_revise_view(state))


def export_revisions(state, decision_label, candidate, text):
    if not state or not state["suggestions"]:
        return None, "先に修正案を作成してください。"
    _record_decision(state, decision_label, candidate, text)
    comments, skipped = [], 0
    with Store() as store:
        for index, saved in sorted(state["decisions"].items()):
            suggestion = state["suggestions"][index]
            target = suggestion.target
            check = None
            if saved["decision"] == "adopt":
                if not saved["text"]:
                    skipped += 1
                    continue
                # 著者が書き換えた案も同じ検査にかける
                score = similarities(target.sentence, [saved["text"]])[0]
                check = check_candidate(target.sentence, saved["text"], store, target.source, score)
            body = decision_comment(suggestion, saved["decision"], saved["text"], check, state["model"])
            if body:
                comments.append(Comment(target.sentence, body))
    path = Path(state["file"])
    if path.suffix.lower() != ".docx":
        return None, "コメントを付けられるのはDOCXだけです。"
    # 原稿の隣に出すとクラウド同期の対象になりうるので、ローカルの data 配下に保存する
    result = write_comments(path, comments, DATA_DIR / "revisions" / f"{path.stem}_修正案.docx")
    status = f"コメント {result['written']} 件を付けました。元のファイルは変更していません。"
    if skipped:
        status += f" 修正案が空の {skipped} 件は除きました。"
    if result["missed"]:
        status += f" 位置を特定できなかった {len(result['missed'])} 件は付けていません。"
    return str(result["path"]), status


with gr.Blocks(title="Binoculars") as demo:
    with gr.Tabs():
        with gr.Tab("類似度チェック"):
            gr.Markdown("# 📄 類似度チェック")
            gr.Markdown(
                "PubMed抄録と手元のPDFを照合先にして、文書の一致箇所を確認します。処理はすべてこのPC内で行います。"
                "照合先にない文献との一致は検出できないため、値が低くても盗用がないことの証明にはなりません。"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    sim_file = gr.File(label="確認する文書（PDF / DOCX / TXT）", type="filepath",
                                       file_types=[".pdf", ".docx", ".txt", ".md"])
                    sim_refs = gr.Checkbox(label="参考文献を除外", value=True)
                    sim_quotes = gr.Checkbox(label="引用符内を除外", value=True)
                    sim_methods = gr.Checkbox(label="方法セクションを除外", value=False)
                    sim_preprints = gr.Checkbox(label="プレプリントを除外", value=False)
                    sim_common = gr.Checkbox(label="定型表現（ライセンス表記など）を除外", value=False)
                    sim_min_words = gr.Slider(minimum=5, maximum=50, value=8, step=1, label="一致とみなす最小語数（下げると定型句の一致が増える）")
                    sim_semantic = gr.Checkbox(label="言い換えも確認", value=True)
                    sim_semantic_threshold = gr.Slider(minimum=0.7, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.01,
                                                       label="言い換えの閾値（下げると候補が増え、誤検出も増える）")
                    sim_exclude = gr.Textbox(label="除外する出典ID（1行に1つ）", lines=2,
                                             placeholder="pmid:12345678")
                    sim_btn = gr.Button("確認を実行", variant="primary")
                with gr.Column(scale=3):
                    sim_summary = gr.Textbox(label="結果", interactive=False)
                    sim_download = gr.File(label="レポート（PDF / HTML）", file_count="multiple", interactive=False)
                    sim_html = gr.HTML()

        with gr.Tab("修正案"):
            gr.Markdown("# ✏️ 修正案")
            gr.Markdown(
                "類似度チェックで一致した文について、対応方法と修正案をこのPC内のLM Studioで作ります。原稿は外部に送りません。"
                "**書き直しても、出典の内容を使っている場合は引用が必要です**。類似度を下げることを目的にせず、1件ずつ内容を確認してください。"
                "本文は自動で変更せず、選んだ対応をコメントとして付けたWordのコピーを作ります。"
            )
            revise_state = gr.State()
            with gr.Row():
                with gr.Column(scale=1):
                    rev_file = gr.File(label="確認する文書（DOCX）", type="filepath", file_types=[".docx"])
                    rev_url = gr.Textbox(label="LM Studio のURL", value=REVISE_URL)
                    rev_model = gr.Textbox(label="モデル（空欄なら先頭のモデル）", value="")
                    rev_btn = gr.Button("修正案を作成", variant="primary")
                    rev_status = gr.Textbox(label="進み具合", interactive=False)
                    rev_export = gr.Button("Wordに書き出す")
                    rev_export_status = gr.Textbox(label="書き出しの結果", interactive=False)
                    rev_download = gr.File(label="コメント付きのWord", interactive=False)
                with gr.Column(scale=3):
                    rev_original = gr.Textbox(label="原稿の文", lines=3, interactive=False)
                    rev_source = gr.Textbox(label="出典", lines=5, interactive=False)
                    rev_advice = gr.Textbox(label="提案（AIの判定なので確認してください）", lines=2, interactive=False)
                    rev_candidates = gr.Radio(label="修正案", choices=[], type="value")
                    rev_checks = gr.Textbox(label="検査の結果", lines=2, interactive=False)
                    rev_edit = gr.Textbox(label="修正案を編集", lines=3)
                    rev_decision = gr.Radio(label="対応", choices=list(DECISIONS.values()), value=DECISIONS["none"])
                    with gr.Row():
                        rev_prev = gr.Button("前の文")
                        rev_next = gr.Button("次の文", variant="primary")

        with gr.Tab("Binocularsスコア"):
            gr.Markdown("# 🔍 Binocularsスコア")
            gr.Markdown(
                "ObserverとPerformerの2つのモデルで文章のスコアを計算します。"
                "スコアが低いほどAI生成文に近い傾向がありますが、**AI利用の有無の根拠にはなりません**。"
                "英語の医学論文の文章での評価では、AUCは抄録の長さの新規作成文で0.729、AIによる推敲で0.541、長文で0.497でした。"
                "日本語の文章は評価していません。"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    observer_input = gr.Textbox(label="Observer (Base)", value="Qwen/Qwen2.5-1.5B")
                    performer_input = gr.Textbox(label="Performer (Instruct)", value="Qwen/Qwen2.5-1.5B-Instruct")
                    load_btn = gr.Button("モデルをロード")
                    load_status = gr.Textbox(label="ステータス", interactive=False)

                with gr.Column(scale=2):
                    input_text = gr.Textbox(label="対象テキスト", lines=6, placeholder="スコアを計算するテキストを入力...")
                    detect_btn = gr.Button("スコアを計算", variant="primary")

            with gr.Row():
                with gr.Column():
                    detail_output = gr.Textbox(label="スコア", lines=5, interactive=False)
                with gr.Column():
                    html_output = gr.HTML(label="トークン単位のハイライト")

    sim_btn.click(
        fn=run_similarity,
        inputs=[sim_file, sim_refs, sim_quotes, sim_methods, sim_preprints, sim_common, sim_min_words,
                sim_semantic, sim_semantic_threshold, sim_exclude],
        outputs=[sim_summary, sim_html, sim_download, input_text],
    )
    revise_view = [rev_status, rev_original, rev_source, rev_advice, rev_candidates, rev_checks, rev_edit, rev_decision]
    revise_inputs = [revise_state, rev_decision, rev_candidates, rev_edit]
    rev_btn.click(fn=run_revise, inputs=[rev_file, rev_url, rev_model], outputs=[revise_state, *revise_view])
    rev_candidates.input(fn=select_candidate, inputs=[revise_state, rev_candidates], outputs=[rev_edit, rev_checks])
    rev_prev.click(fn=lambda *a: save_and_move(*a, -1), inputs=revise_inputs, outputs=[revise_state, *revise_view])
    rev_next.click(fn=lambda *a: save_and_move(*a, 1), inputs=revise_inputs, outputs=[revise_state, *revise_view])
    rev_export.click(fn=export_revisions, inputs=revise_inputs, outputs=[rev_download, rev_export_status])
    load_btn.click(fn=load_models, inputs=[observer_input, performer_input], outputs=[load_status])
    detect_btn.click(fn=run_prediction, inputs=[input_text], outputs=[detail_output, html_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
