import argparse
import transformers
from accelerate import init_empty_weights
from typing import Optional, Tuple

def calc_memory_usage(
    parameters_num: int,
    bytes_per_parameter: int = 2,
    lora_parameters_num: Optional[int] = None,
    zero_config: str = 'z3',
    gpu_num: int = 2,
    batch_size: int = 1,
    seq_length: int = 2048,
    layer_num: int = 32,
    hidden_size: int = 4096,
    vocab_size: int = 32000
) -> float:
    """
    Calculate the estimated memory usage for a model based on various parameters.

    Args:
        parameters_num (int): The number of parameters in the model.
        bytes_per_parameter (int, optional): Number of bytes per parameter. Defaults to 2.
        lora_parameters_num (Optional[int], optional): Number of LoRA parameters. Defaults to None.
        zero_config (str, optional): Zero configuration type. Defaults to 'z3'.
        gpu_num (int, optional): Number of GPUs. Defaults to 2.
        batch_size (int, optional): Batch size for the model. Defaults to 1.
        seq_length (int, optional): Sequence length for the model. Defaults to 2048.
        layer_num (int, optional): Number of layers in the model. Defaults to 32.
        hidden_size (int, optional): Size of the hidden layers. Defaults to 4096.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 32000.

    Returns:
        float: Estimated memory usage in bytes.
    
    Raises:
        ValueError: If an invalid zero configuration is provided.
    """
    if lora_parameters_num is None:
        optimizer_parameters_num = parameters_num
    else:
        parameters_num += lora_parameters_num
        optimizer_parameters_num = lora_parameters_num

    activation_memory = (12 + 2 * layer_num) * batch_size * seq_length * hidden_size + 12 * batch_size * seq_length * vocab_size

    match zero_config:
        case 'z0':
            memory = bytes_per_parameter * parameters_num + 14 * optimizer_parameters_num
            memory += max(2 * optimizer_parameters_num, activation_memory)
        case 'z1':
            memory = bytes_per_parameter * parameters_num + 2 * optimizer_parameters_num + 12 * optimizer_parameters_num / gpu_num
            memory += max(2 * optimizer_parameters_num + 2 * optimizer_parameters_num / gpu_num, activation_memory)
        case 'z2':
            memory = bytes_per_parameter * parameters_num + 14 * optimizer_parameters_num / gpu_num
            memory += max(2 * optimizer_parameters_num + 2 * optimizer_parameters_num / gpu_num, activation_memory)
        case 'z2+oo':
            memory = bytes_per_parameter * parameters_num
            memory += max(2 * optimizer_parameters_num, activation_memory)
        case 'z3':
            memory = bytes_per_parameter * parameters_num / gpu_num + 14 * optimizer_parameters_num / gpu_num
            memory += max(2 * optimizer_parameters_num / gpu_num, activation_memory)
        case 'z3+oo':
            memory = bytes_per_parameter * parameters_num / gpu_num
            memory += max(2 * optimizer_parameters_num / gpu_num, activation_memory)
        case 'z3+oo+op':
            memory = activation_memory
        case _:
            raise ValueError('Invalid zero_config!')

    return memory


def get_gpu_memory_size(gpu_name: str = 'A100') -> int:
    """
    Get the memory size of a specified GPU.

    Args:
        gpu_name (str, optional): The name of the GPU. Defaults to 'A100'.

    Returns:
        int: Memory size of the GPU in bytes.

    Raises:
        ValueError: If the specified GPU is unsupported.
    """
    match gpu_name.lower():
        case 'h100' | 'h800' | 'a100' | 'a800':
            return 80 * 2**30  # 80 GB
        case 'a100-40g' | 'a800-40g':
            return 40 * 2**30  # 40 GB
        case 'a6000' | 'a40':
            return 48 * 2**30  # 48 GB
        case 'v100':
            return 32 * 2**30  # 32 GB
        case '3090' | '4090' | 'rtx-3090' | 'rtx-4090':
            return 24 * 2**30  # 24 GB
        case _:
            raise ValueError('Unsupported GPU!')


def get_bytes_per_parameter(model_dtype: str) -> float:
    """
    Get the number of bytes per parameter based on the model's data type.

    Args:
        model_dtype (str): The data type of the model (e.g., 'bf16', 'fp16', 'int8', etc.).

    Returns:
        float: Number of bytes per parameter.

    Raises:
        ValueError: If the specified model data type is unsupported.
    """
    if model_dtype.lower() in ('bf16', 'fp16'):
        return 2.0
    elif model_dtype.lower() in ('int8', 'fp8'):
        return 1.0
    elif model_dtype.lower() == 'int4':
        return 0.5
    else:
        raise ValueError('Unsupported model_dtype!')


def calc_gpu_num(
    model_name: str = 'meta-llama/Llama-2-7b-hf',
    model_dtype: str = 'bf16',
    lora_module: str = 'q,k,v,o',
    lora_rank: int = 8,
    seq_length: int = 2048,
    zero_config: str = 'z3',
    gpu_name: str = 'A100',
    batch_size: int = 1
) -> Tuple[int, float]:
    """
    Calculate the number of GPUs required for a given model configuration.

    Args:
        model_name (str, optional): The name of the model. Defaults to 'meta-llama/Llama-2-7b-hf'.
        model_dtype (str, optional): The data type of the model. Defaults to 'bf16'.
        lora_module (str, optional): The LoRA module configuration. Defaults to 'q,k,v,o'.
        lora_rank (int, optional): The rank for LoRA. Defaults to 8.
        seq_length (int, optional): The sequence length for the model. Defaults to 2048.
        zero_config (str, optional): Zero configuration type. Defaults to 'z3'.
        gpu_name (str, optional): The name of the GPU. Defaults to 'A100'.
        batch_size (int, optional): The batch size for the model. Defaults to 1.

    Returns:
        Tuple[int, float]: A tuple containing the number of GPUs and the estimated memory usage.

    Raises:
        ValueError: If a suitable number of GPUs cannot be found for the configuration.
    """
    config = transformers.AutoConfig.from_pretrained(model_name)
    layer_num = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    gpu_memory = get_gpu_memory_size(gpu_name)
    bytes_per_parameter = get_bytes_per_parameter(model_dtype)

    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config)
        parameters_num = model.num_parameters()
        print(f'parameters_num = {parameters_num:e}')

    if lora_module is not None:
        lora_module = lora_module.split(',')
        lora_parameters_num = 2 * hidden_size * lora_rank * layer_num * len(lora_module)
        print(f'lora_parameters_num = {lora_parameters_num:e}')
    else:
        lora_parameters_num = None

    for gpu_num in range(1, 100):
        memory_usage = calc_memory_usage(
            parameters_num,
            bytes_per_parameter=bytes_per_parameter,
            lora_parameters_num=lora_parameters_num,
            zero_config=zero_config,
            gpu_num=gpu_num,
            batch_size=batch_size,
            seq_length=seq_length,
            layer_num=layer_num,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        if memory_usage <= 0.95 * gpu_memory:
            return gpu_num, memory_usage

    raise ValueError('Cannot find suitable gpu_num for this config!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="The name of the model to use."
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
        help="The data type of the model (e.g., bf16, fp16)."
    )
    parser.add_argument(
        "--zero_config",
        type=str,
        default="",
        help="Zero configuration type."
    )
    parser.add_argument(
        "--lora_module",
        type=str,
        default=None,
        help="LoRA module configuration."
    )
    parser.add_argument(
        "--gpu_name",
        type=str,
        default="",
        help="The name of the GPU to use."
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="The sequence length for the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for the model."
    )
    args = parser.parse_args()

    gpu_num, memory_usage = calc_gpu_num(
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        lora_module=args.lora_module,
        seq_length=args.seq_length,
        zero_config=args.zero_config,
        gpu_name=args.gpu_name,
        batch_size=args.batch_size,
    )
    print(f'GPU: {gpu_num}, Memory: {memory_usage / 2**30:.2f} GB')