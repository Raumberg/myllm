# MyLLM
### LLM Framework | Toolkit for various training stages
## Methods and Stages supported:
- Supervised Finetuning (Full/LoRa/QLoRa)
- Distillation (KL Divergence, MSE, Cosine and others)
- Reinforcement Learning (GRPO, DPO, PPO)
- Adapters merging
- Tokenizer extensions
## Technical details:
- Built on top of PyTorch, Transformers, TRL, Peft. No 'magic' libraries like unsloth.
- Distributed training via Accelerate, FSDP and DeepSpeed (Stage 2, 3).
- Acceleration with vLLM, FlashAttn, Liger Kernels and fusion.
- Logging options: wandb, clearml