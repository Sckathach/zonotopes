import torch as t

DEVICE: str = "cuda" if t.cuda.is_available() else "cpu"
