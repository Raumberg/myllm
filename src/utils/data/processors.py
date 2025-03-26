from transformers import AutoTokenizer
from .extraction import extract_hash_answer

def history_row_processor(
        row: str, 
        args: dict,
        training_config: dict,
        tokenizer: AutoTokenizer,
        add_gen_prompt=False,
        ):
    system_message = [{'role': 'system', 'content': args.system_prompt}] if args.system_prompt else []
    history = row[args.conversation_field] if not add_gen_prompt else row[args.conversation_field][:-1]
    if not args.model_support_system_role and history[0]["role"] == "system":
        if len(history) > 1 and history[1]["role"] == "user":
            # add sys prompt to first user message
            history[1]["content"] = history[0]["content"] + "\n" + history[1]["content"]
            history = history[1:]
        else:
            history[0]["role"] = "user"
    
    constructed_prompt = tokenizer.apply_chat_template(
        system_message + history,
        tokenize=False,
        add_generation_prompt=add_gen_prompt
    )
    if tokenizer.bos_token is not None:
        if constructed_prompt.startswith(tokenizer.bos_token):  # Remove extra bos token
            constructed_prompt = constructed_prompt[len(tokenizer.bos_token):]
    return tokenizer(constructed_prompt, truncation=True, padding=True, max_length=training_config.max_seq_length)

def default_row_processor(
        row: str, 
        args: dict,
        training_config: dict,
        tokenizer: AutoTokenizer,
        add_gen_prompt=False,
        ):
    system_message = [{'role': 'system', 'content': args.system_prompt}] if args.system_prompt else []
    
    # Ensure we're getting a single message dictionary
    message = row[args.conversation_field]
    if isinstance(message, list):  # Handle case where conversation_field contains message list
        message = message[-1]  # Take last message in conversation
    
    # Handle system message merging
    if not args.model_support_system_role and system_message:
        if message['role'] == 'user':
            message['content'] = system_message[0]['content'] + '\n' + message['content']
        else:
            message = {'role': 'user', 'content': system_message[0]['content'] + '\n' + message['content']}

    # Construct messages list properly
    messages = system_message + [message] if args.model_support_system_role else [message]

    # Apply chat template and ensure string output
    constructed_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_gen_prompt
    )

    # Handle cases where apply_chat_template might return list of strings
    if isinstance(constructed_prompt, list):
        constructed_prompt = "".join(constructed_prompt)
    
    # Remove BOS token if present
    if tokenizer.bos_token and constructed_prompt.startswith(tokenizer.bos_token):
        constructed_prompt = constructed_prompt[len(tokenizer.bos_token):]

    return tokenizer(
        constructed_prompt, 
        truncation=True, 
        padding=True,
        max_length=training_config.max_seq_length
    )

def grpo_row_processor(row, args):
    result = {
        "prompt": [
            {'role': 'system', 'content': args.system_prompt},
            {'role': 'user', 'content': row[args.problem_field]} 
        ]
    }
    
    if args.extract_hash: 
        result["answer"] = extract_hash_answer(row[args.solution_field])
    else: 
        result["answer"] = row[args.solution_field]
    
    return result