import torch
from transformers import AutoTokenizer, AutoModel

class DRAMAModel:
    _instance = None
    
    @classmethod
    def get_instance(cls, device: str = None):
        if cls._instance is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance = cls._load_model(device)
        return cls._instance
    
    @staticmethod
    def _load_model(device: str):
        model_name = "facebook/drama-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        return {"model": model, "tokenizer": tokenizer, "device": device}