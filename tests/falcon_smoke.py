"""Opt-in real-model test; requires cached/downloadable Falcon weights.

python tests/falcon_smoke.py --device mps --output artifacts/mps-tests/mps.json
Run CPU and MPS separately so four models are never resident simultaneously.
"""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import time

import torch
from binoculars import Binoculars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps":
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "Disable fallback for this test"
        assert torch.backends.mps.is_available()

    started = time.perf_counter()
    bino = Binoculars(device=args.device)
    load_seconds = time.perf_counter() - started
    forwards = []

    def check_forward(module, inputs, output):
        assert output.logits.device.type == args.device
        forwards.append(str(output.logits.device))

    for model in (bino.observer_model, bino.performer_model):
        assert all(p.device.type == args.device for p in model.parameters())
        model.register_forward_hook(check_forward)

    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
    sample = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "sample_string" for t in node.targets))
    batches = [sample, [
        "The train arrived late this morning. I left my umbrella on the platform and had to walk back in the rain.",
        "Photosynthesis allows plants to convert sunlight into chemical energy. Chlorophyll absorbs light, supporting reactions that produce sugars from carbon dioxide and water."
    ]]
    results = []
    for text in batches:
        started = time.perf_counter()
        score = bino.compute_score(text)
        if args.device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        scores = [score] if isinstance(score, float) else score
        assert len(scores) == (1 if isinstance(text, str) else len(text))
        assert all(math.isfinite(s) and s > 0 for s in scores)
        result = {"scores": scores, "seconds": elapsed,
                  "predictions": ["Most likely AI-generated" if s < bino.threshold
                                  else "Most likely human-generated" for s in scores]}
        results.append(result)
        print(json.dumps(result), flush=True)

    assert len(forwards) == 4
    report = {"device": args.device, "torch": torch.__version__, "dtype": "bfloat16",
              "load_seconds": load_seconds, "forward_devices": forwards, "results": results}
    if args.device == "mps":
        report["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
