

def infer_dtype(s: str):  # noqa: D401
    import torch
    s = s.lower()
    if s in {"fp16", "float16", "16"}:
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    return torch.float32