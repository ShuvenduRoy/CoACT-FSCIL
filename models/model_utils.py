"""Model utilities."""

from transformers import ViTModel


def get_model_args(model: str) -> dict:
    """Get model arguments."""
    args = {
        "embedding_dim": 768,
        "hf_model_class": ViTModel,
    }

    if "huge" in model:
        args["embedding_dim"] = 1280
    elif "large" in model or "hybrid" in model:
        args["embedding_dim"] = 1024
    return args
