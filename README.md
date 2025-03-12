<div align="center">
  <h1 style="background: linear-gradient(to right, black, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        MyLLM
    </h1>
    <div style="border: 10px solid; border-image: linear-gradient(to right, black, white) 1; padding: 10px; display: inline-block;">
        <img src="assets/torch.jpg" alt="gallery" width="500"/>
    </div>
    <br>
    <br>
    <p align="center">
        <img src="https://img.shields.io/github/issues/Raumberg/myllm?style=for-the-badge">
        <br>
        <img src="https://img.shields.io/github/languages/count/Raumberg/myllm?style=for-the-badge">
        <img src="https://img.shields.io/github/repo-size/Raumberg/myllm?style=for-the-badge">
        <br>
    </p>
</div>

### LLM Framework | Toolkit for various training stages
Initially derived from [Effective LLM Alignment](https://github.com/VikhrModels/effective_llm_alignment/) by VikhrModels.  
Many credits goes to the Vikhr Team.

# 🚀 New Feature Release: Reinforcement Learning with GRPO! 🎉
We are excited to announce the release of a new feature: **Reinforcement Learning with GRPO**!  
This addition allows you to leverage advanced reinforcement learning techniques to make your models excel with reasoning abilities.  
With GRPO, you can enhance the training process by incorporating reward signals to invoke reasoning capabilities of LLMs. 

## 🚀 Methods and Stages supported:
- Supervised Finetuning (Full/LoRa/QLoRa)
- Distillation (KL Divergence, MSE, Cosine and others)
- Reinforcement Learning (GRPO, DPO, PPO)
- Adapters merging
- Tokenizer extensions

## 🛠️ Technical details:
- Built on top of PyTorch, Transformers, TRL, Peft. No 'magic' libraries like unsloth.
- Distributed training via Accelerate, FSDP and DeepSpeed (Stage 2, 3).
- Acceleration with vLLM, FlashAttn, Liger Kernels and fusion.
- Logging options: wandb, clearml
- Convenient config management using TOML

## How to train?
- Everything is available from the root (MyLLM) folder. 
- What you need to do is start any desired script using accelerate:  
```bash
# ~/../myllm >
accelerate launch --config_file <path-to-cfg.yaml> <path-to-script.py> <path-to-model-cfg.toml>
# example SFT:
accelerate launch --config_file configs/accelerate/stage3_config.yaml src/train/sft.py configs/train/sft/full-sft-watari.toml
# example GRPO:
accelerate launch --config_file configs/accelerate/grpo_deepspeed.yaml src/train/grpo.py configs/train/grpo/rl-grpo-zariman-no-vllm.toml
```  
Note:  
GRPO scripts can be unstable, the work is still going on. If you encounter any errors, please, open an Issue.

> [!IMPORTANT]
> Thank you for your interest in MyLLM! We look forward to your contributions and feedback! 🚀
