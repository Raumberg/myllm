from dataclasses import dataclass
from typing import Optional

from transformers import TrainingArguments
from transformers.hf_argparser import HfArg

@dataclass
class PPOScriptArguments(TrainingArguments):

    model_name: Optional[str] = HfArg(
        default="", 
        help="the model name",
    )
    
    reward_model_name: Optional[str] = HfArg(
        default="", 
        help="the reward model name",
    )
    
    dataset_name: Optional[str] = HfArg(
        default="", 
        help="the dataset name",
    )
    
    learning_rate: Optional[float] = HfArg(
        default=1.41e-5, 
        help="the learning rate",
    )
    
    max_length: Optional[int] = HfArg(
        default=512, 
        help="maximum length for input",
    )
    
    output_min_length: Optional[int] = HfArg(
        default=32, 
        help="minimum length for generation",
    )
    
    output_max_length: Optional[int] = HfArg(
        default=128, 
        help="maximum length for generation",
    )
    
    mini_batch_size: Optional[int] = HfArg(
        default=1, 
        help="the PPO minibatch size",
    )
    
    gradient_accumulation_steps: Optional[int] = HfArg(
        default=4, 
        help="the number of gradient accumulation steps",
    )

    batch_size: Optional[int] = HfArg(
        default=32, 
        help="the batch size",
    )
    
    ppo_epochs: Optional[int] = HfArg(
        default=4, 
        help="the number of ppo epochs",
    )
    
    early_stopping: Optional[bool] = HfArg(
        default=False, 
        help="whether to early stop",
    )
    
    target_kl: Optional[float] = HfArg(
        default=0.1, 
        help="kl target for early stopping",
    )
    
    reward_baseline: Optional[float] = HfArg(
        default=0.0, 
        help="a baseline value that is subtracted from the reward",
    )
    
    save_freq: Optional[int] = HfArg(
        default=None, 
        help="n steps to save the model",
    )
    
    output_dir: Optional[str] = HfArg(
        default="./checkpoints/tuning_llama_rl/", 
        help="n steps to save the model",
    )
    
    lora_r: Optional[int] = HfArg(default=16, help='Lora attention dimension (the "rank")')
    
    lora_alpha: Optional[int] = HfArg(default=16, help="The alpha parameter for Lora scaling.")
    
    lora_dropout: Optional[float] = HfArg(default=0.05, help="The dropout probability for Lora layers.")
    
    top_k: Optional[float] = HfArg(default=0.0, help="Topk for generation.")
    
    top_p: Optional[float] = HfArg(default=1.0, help="Topp for generation.")