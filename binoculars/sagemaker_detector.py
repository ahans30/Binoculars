# type: ignore
import torch
from sagemaker.deserializers import NumpyDeserializer
from sagemaker.huggingface.model import HuggingFacePredictor
from transformers import AutoTokenizer, BatchEncoding

from binoculars.detector import Binoculars
from binoculars.metrics import entropy, perplexity
from binoculars.utils import assert_tokenizer_consistency

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"


class SageMakerBinoculars(Binoculars):
    """
    Wrapper around the Binoculars class which allows using calls to AWS SageMaker to get the logits.
    Note: Requires modification of the SageMaker models to return logits rather than the standard string output.
    """

    def __init__(
        self,
        observer_endpoint_name: str,
        performer_endpoint_name: str,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_predictor = HuggingFacePredictor(
            observer_endpoint_name,
            deserializer=NumpyDeserializer(),
        )
        self.performer_predictor = HuggingFacePredictor(
            performer_endpoint_name,
            deserializer=NumpyDeserializer(),
        )

        self.change_mode(mode)

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def _tokenize(self, batch: list[str]) -> BatchEncoding:
        batch_size = len(batch)
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )

    def _get_logits(self, input_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = {
            "inputs": input_text,
        }
        observer_logits = torch.Tensor(self.observer_predictor.predict(data))
        performer_logits = torch.Tensor(self.performer_predictor.predict(data))

        return observer_logits, performer_logits

    def compute_score(
        self,
        input_text: str,
    ) -> float:
        encodings = self._tokenize([input_text])
        observer_logits, performer_logits = self._get_logits(input_text)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0]
