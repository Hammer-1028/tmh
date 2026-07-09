from __future__ import annotations

import importlib.util
from pathlib import Path

from inference_core import DEFAULT_MODEL_PATH


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("Deep learning demo environment check")
    for name in ["torch", "streamlit", "numpy", "PIL", "scipy"]:
        print(f"{name}: {'OK' if has_module(name) else 'MISSING'}")
    print(f"model_path: {DEFAULT_MODEL_PATH}")
    print(f"model_exists: {Path(DEFAULT_MODEL_PATH).exists()}")
    if has_module("torch"):
        import torch

        print(f"torch_version: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
