import numpy as np

RERANK_MODEL = "Qwen/Qwen2.5-1.5B"
# 出典の前後も文脈として渡す
CONTEXT_CHARS = 400
# 再評価するときは、指定の類似度よりこれだけ下の候補まで見る（既定の0.85なら0.82）
RERANK_MARGIN = 0.03
# 本物の言い換えと、同じ話題の別の文を分ける値（evaluation/detection/results.md）
DEFAULT_MIN_GAIN = 0.6

_scorer = None


class GainScorer:
    """出典を読ませると文の予測がどれだけ楽になるかを測る（SCDG、arXiv:2608.03859）。

    言い換えでは語が変わっても内容の予測が楽になるので、話題が同じだけの文と区別しやすい。
    """

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        # 原稿を外部に出さないよう、ローカルのキャッシュからのみ読み込む
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, local_files_only=True)
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            RERANK_MODEL, local_files_only=True, dtype=torch.float16 if cuda else torch.float32,
        ).eval()
        if cuda:
            self.model = self.model.to("cuda")

    def _log_probs(self, ids: list[int]) -> np.ndarray:
        torch = self.torch
        tensor = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(tensor).logits[0, :-1]
        chosen = torch.log_softmax(logits.float(), dim=-1).gather(1, tensor[0, 1:].unsqueeze(1)).squeeze(1)
        return chosen.cpu().numpy()

    def gains(self, source: str, target: str) -> np.ndarray:
        separator = self.tokenizer("\n\n", add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        source_ids = self.tokenizer(source, add_special_tokens=False).input_ids
        if not target_ids:
            return np.zeros(1)
        # 両方とも同じ語を評価するよう、先頭に区切りを置く
        unconditional = self._log_probs(separator[:1] + target_ids)
        conditional = self._log_probs(source_ids + separator + target_ids)[-len(target_ids):]
        return conditional - unconditional[-len(target_ids):]

    def mean_gain(self, source: str, target: str) -> float:
        return float(self.gains(source, target).mean())


def get_scorer() -> GainScorer:
    global _scorer
    if _scorer is None:
        _scorer = GainScorer()
    return _scorer


def source_context(doc_text: str, start: int, end: int) -> str:
    return doc_text[max(0, start - CONTEXT_CHARS):end + CONTEXT_CHARS]
