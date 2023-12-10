import os

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ["HF_TOKEN"] if "HF_TOKEN" in os.environ else None,
}
