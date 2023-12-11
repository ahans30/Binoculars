from typing import Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from binoculars.utils import assert_tokenizer_consistency
from config import huggingface_config
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

GLOBAL_BINOCULARS_THRESHOLD = 0.9015310749276843


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": device},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": device},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )

        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_observer_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        return self.observer_model(**encodings).logits

    @torch.inference_mode()
    def _get_performer_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        return self.performer_model(**encodings).logits

    @torch.inference_mode()
    def _get_ppl(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        performer_logits = self._get_performer_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        return ppl

    @torch.inference_mode()
    def _get_x_ppl(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self._get_observer_logits(encodings)
        performer_logits = self._get_performer_logits(encodings)
        x_ppl = entropy(observer_logits, performer_logits, encodings, self.tokenizer.pad_token_id)
        return x_ppl

    @torch.inference_mode()
    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        ppl = self._get_ppl(encodings)
        x_ppl = self._get_x_ppl(encodings)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    @torch.inference_mode()
    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = self.compute_score(input_text)
        pred = np.where(binoculars_scores < GLOBAL_BINOCULARS_THRESHOLD, "AI-Generated", "Human-Generated").tolist()
        return pred
