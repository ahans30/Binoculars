"""Offline device-routing tests: python -m unittest discover -s tests -v."""
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import BatchEncoding
from binoculars.detector import Binoculars, select_devices


class DeviceTests(unittest.TestCase):
    def select(self, cuda=False, count=0, mps=False, device="auto"):
        with patch("torch.cuda.is_available", return_value=cuda), \
             patch("torch.cuda.device_count", return_value=count), \
             patch("torch.backends.mps.is_available", return_value=mps):
            return select_devices(device)

    def test_cpu(self):
        self.assertEqual(self.select(), ("cpu", "cpu"))

    def test_mps(self):
        self.assertEqual(self.select(mps=True), ("mps", "mps"))

    def test_cuda_priority(self):
        self.assertEqual(self.select(cuda=True, count=2, mps=True), ("cuda:0", "cuda:1"))

    def test_single_cuda(self):
        self.assertEqual(self.select(cuda=True, count=1), ("cuda:0", "cuda:0"))

    def test_explicit_cpu(self):
        self.assertEqual(self.select(cuda=True, count=2, mps=True, device="cpu"), ("cpu", "cpu"))

    def test_explicit_mps(self):
        self.assertEqual(self.select(cuda=True, count=2, mps=True, device="mps"), ("mps", "mps"))

    def test_unavailable_backend(self):
        for device in ("cuda", "mps"):
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                self.select(device=device)

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            self.select(device="invalid")

    def test_model_loading_uses_selected_device(self):
        with patch("binoculars.detector.select_devices", return_value=("mps", "mps")), \
             patch("binoculars.detector.assert_tokenizer_consistency"), \
             patch("binoculars.detector.AutoTokenizer.from_pretrained"), \
             patch("binoculars.detector.AutoModelForCausalLM.from_pretrained") as loader:
            bino = Binoculars(device="mps")
            self.assertEqual(bino.device_1, "mps")
            self.assertEqual(loader.call_count, 2)
            for call in loader.call_args_list:
                self.assertEqual(call.kwargs["device_map"], {"": "mps"})
                self.assertEqual(call.kwargs["torch_dtype"], torch.bfloat16)

    def test_logits_disable_cache_and_synchronize_correct_backend(self):
        for devices in (("cpu", "cpu"), ("mps", "mps"), ("cuda:0", "cuda:1")):
            with self.subTest(devices=devices), \
                 patch("torch.cuda.synchronize") as cuda_sync, \
                 patch("torch.mps.synchronize") as mps_sync:
                bino = Binoculars.__new__(Binoculars)
                bino.device_1, bino.device_2 = devices
                bino.observer_model, bino.performer_model = Mock(), Mock()
                encoding = Mock()
                encoding.to.return_value = {"input_ids": "tokens"}
                bino._get_logits(encoding)
                for model in (bino.observer_model, bino.performer_model):
                    model.assert_called_once_with(input_ids="tokens", use_cache=False)
                self.assertEqual(mps_sync.call_count, int(devices[0] == "mps"))
                self.assertEqual(cuda_sync.call_count, 2 if devices[0] == "cuda:0" else 0)

    def test_cpu_score_scalar_and_batch(self):
        bino = Binoculars.__new__(Binoculars)
        bino.device_1 = bino.device_2 = "cpu"
        bino.tokenizer = Mock(pad_token_id=0)
        bino.change_mode("low-fpr")
        for text in ("example", ["one", "two"]):
            n = 1 if isinstance(text, str) else len(text)
            encoding = BatchEncoding({"input_ids": torch.ones(n, 4, dtype=torch.long),
                                      "attention_mask": torch.ones(n, 4, dtype=torch.long)})
            logits = torch.zeros(n, 4, 8)
            bino._tokenize = Mock(return_value=encoding)
            bino._get_logits = Mock(return_value=(logits, logits))
            self.assertEqual(bino.compute_score(text), 1.0 if n == 1 else [1.0, 1.0])
            self.assertEqual(bino.predict(text), "Most likely human-generated" if n == 1
                             else ["Most likely human-generated"] * n)


if __name__ == "__main__":
    unittest.main()
