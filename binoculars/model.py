import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import huggingface_config

torch.set_grad_enabled(False)


class Model(object):
    def __init__(self, model_path: str, device: str, max_token_observed=512) -> None:
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map={"": device},
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16,
                                                          token=huggingface_config["TOKEN"]
                                                          )
        self.max_token_observed = max_token_observed
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text,
                              return_tensors="pt",
                              padding=False,
                              truncation=True,
                              max_length=self.max_token_observed,
                              ).input_ids.to(self.device)

    def tokenize(self, batch: list[str]) -> torch.Tensor:
        return self.tokenizer(batch, return_tensors="pt", padding=True).input_ids.to(self.device)

    def get_logits(self, text: str) -> torch.Tensor:
        return self.model(**self.tokenize(text)).logits


