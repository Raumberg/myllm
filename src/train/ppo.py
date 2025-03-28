import logging
import warnings

from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    AutoModelForSequenceClassification, 
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from src.utils.scriptargs import PPOScriptArguments


tqdm.pandas()


def build_dataset(args, tokenizer):

    train_dataset = load_dataset(args.dataset_name, split="train")

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        #You need to specify the label (i.g. "text" here) according to your dataset
        for question in examples["text"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=24, #num of pre-process CPUs
        remove_columns=train_dataset.column_names,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train():
    
    parser = HfArgumentParser(PPOScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    #Prepare the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
        add_eos_token=True,
        use_fast=False,
        legacy=False,  # refer to the issue:https://github.com/huggingface/transformers/pull/24565
        use_cache=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        peft_config=lora_config,
    )

    #Prepare the dataset and trainer
    dataset = build_dataset(args, tokenizer)
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=args.early_stopping,
        target_kl=args.target_kl,
        ppo_epochs=args.ppo_epochs,
    )
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=None, #If no reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized with shared layers.
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    #Prepare the generation configuration
    generation_kwargs = {
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    output_length_sampler = LengthSampler(args.output_min_length, args.output_max_length)

    reward_name = args.reward_model_name
    rank_model, rank_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = []
        for q, r in zip(batch["query"], batch["response"]):
            inputs = rank_tokenizer(q, r, return_tensors='pt')
            score = rank_model(**inputs).logits[0].cpu().detach() - args.reward_baseline
            rewards.append(score)
        
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if args.save_freq and epoch and epoch % args.save_freq == 0:
            ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch}")
        ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch}")    

def init():
    set_seed(42)
    warnings.filterwarnings("ignore")
    logging.getLogger("DeepSpeed").setLevel(logging.ERROR)


if __name__ == "__main__":
    init()
    train()