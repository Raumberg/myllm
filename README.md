<div align="center">
  <h1 style="background: linear-gradient(to right, black, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        MyLLM
    </h1>
    <div style="border: 10px solid; border-image: linear-gradient(to right, black, white) 1; padding: 10px; display: inline-block;">
        <img src="assets/myllm.png" alt="gallery" width="600"/>
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

# LLM Framework | Toolkit for various training stages
Initially derived from [Effective LLM Alignment](https://github.com/VikhrModels/effective_llm_alignment/) by VikhrModels.  
Many credits goes to the Vikhr Team.

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

## How can I use the framework?
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

## Useful script:
The folder `myllm/src/helpers` contains useful scripts that you can utilize for your models:
- Merge your LoRA adapters with original model using `adapters.py` by:
```bash
cd myllm/src/helpers
python adapters.py merge --source ../../models/attn-signs-watari-32/checkpoint-5500/ --output ../../models/attn-signs-watari-32/watari-32-merged --dtype bf16
```
- Extend model tokenizer by using `tokenizer.py`

# Latest changes:
- Added lora-sft-watariv2.toml training config from [Attention Signs HuggingFace Page](https://huggingface.co/attn-signs/Watari-32b-v0)
- Added new modules: `stdout` and `data_processors` and `liger`.
    - **stdout:** print your model config, script arguments and training config in table. Example:
    ```
        Model Inspection:
    +----------------------+----------------------------+
    | Config key           | Config value               |
    +======================+============================+
    | Model Architecture   | Qwen2ForCausalLM           |
    +----------------------+----------------------------+
    | Total Parameters     | 0                          |
    +----------------------+----------------------------+
    | Trainable Parameters | 0                          |
    +----------------------+----------------------------+
    | Dtype                | torch.bfloat16             |
    +----------------------+----------------------------+
    | Device               | cuda:0                     |
    +----------------------+----------------------------+
    | Tokenizer Vocab Size | 147200                     |
    +----------------------+----------------------------+
    | Model Embedding Size | 0                          |
    +----------------------+----------------------------+
    | Padding Token        | <|endoftext|> (ID: 147075) |
    +----------------------+----------------------------+
    | EOS Token            | <|im_end|> (ID: 147077)    |
    +----------------------+----------------------------+
    | Max Sequence Length  | 32768                      |
    +----------------------+----------------------------+
    | Architecture         | Qwen2ForCausalLM           |
    +----------------------+----------------------------+
    | Hidden Size          | 5120                       |
    +----------------------+----------------------------+
    | Attention Heads      | 40                         |
    +----------------------+----------------------------+
    ```
    - **data_processors**: moved all tokenizer processing functions to a separate module. Added support for default processing and history processing.
    - **liger**: moved all liger kernels to a separate module

Overall, the training scripts are becoming more easy to read and user-friendly, outsourcing difficult tasks under the hood.

> [!IMPORTANT]
> Thank you for your interest in MyLLM! We look forward to your contributions and feedback! 🚀
