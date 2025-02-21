import numpy as np
import warnings
from accelerate import PartialState
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from typing import Optional, Union

def _load_dataset_from_path(path: str, test_size: Optional[float] = None) -> DatasetDict:
    """
    Load a dataset from a specified path. If the path points to a JSONL file, 
    it loads it using the JSON loader. Optionally splits the dataset into 
    training and testing sets based on the specified test size.

    Args:
        path (str): The path to the dataset file or dataset name.
        test_size (Optional[float]): The proportion of the dataset to include in the test split. 
                                      If None, no split is performed.

    Returns:
        DatasetDict: A dictionary containing the loaded dataset, with 'train' and 'test' splits if applicable.
    """
    if path.endswith('jsonl'):
        dataset = load_dataset("json", data_files=path)
    else:
        dataset = load_dataset(path)
    
    if test_size is not None:
        dataset = dataset['train'].train_test_split(test_size, seed=42, load_from_cache_file=True)
    
    return dataset


def _get_subset_from_dataset(dataset: Dataset, dataset_ratio: Optional[float] = None) -> Dataset:
    """
    Get a random subset of a dataset based on the specified ratio.

    Args:
        dataset (Dataset): The dataset from which to select a subset.
        dataset_ratio (Optional[float]): The ratio of the dataset to retain. 
                                          If None, the entire dataset is returned.

    Returns:
        Dataset: A subset of the original dataset.
    """
    if dataset_ratio is None:
        return dataset  # Return the entire dataset if no ratio is specified
    
    indices = np.random.choice(range(len(dataset)), int(dataset_ratio * len(dataset)), replace=False)
    dataset = dataset.select(indices)
    
    return dataset


def _get_subset_from_dataset_dict(dataset: DatasetDict, dataset_ratio: Optional[float] = None) -> DatasetDict:
    """
    Get subsets of the training and testing datasets within a DatasetDict.

    Args:
        dataset (DatasetDict): The DatasetDict containing 'train' and 'test' datasets.
        dataset_ratio (Optional[float]): The ratio of the datasets to retain.

    Returns:
        DatasetDict: A DatasetDict containing the subsets of the original datasets.
    """
    dataset["train"] = _get_subset_from_dataset(dataset["train"], dataset_ratio)
    dataset["test"] = _get_subset_from_dataset(dataset["test"], dataset_ratio)
    
    return dataset


def load_datasets(path: Union[str, list], test_size: Optional[float] = None, dataset_ratio: Optional[Union[float, list]] = None) -> DatasetDict:
    """
    Load one or more datasets from specified paths, optionally splitting them into 
    training and testing sets and selecting subsets based on specified ratios.

    Args:
        path (Union[str, list]): The path(s) to the dataset(s). Can be a single path or a list of paths.
        test_size (Optional[float]): The proportion of the dataset to include in the test split. 
                                      If None, no split is performed.
        dataset_ratio (Optional[Union[float, list]]): The ratio(s) of the dataset(s) to retain. 
                                                      If None, the entire dataset is used.

    Returns:
        DatasetDict: A DatasetDict containing the combined training and testing datasets.

    Raises:
        ValueError: If the number of dataset paths does not match the number of dataset ratios.
    """
    with PartialState().local_main_process_first():
        if dataset_ratio is None:
            warnings.warn("You haven't set dataset ratio for your datasets. Assuming that it's 1 for all datasets.")
            dataset_ratio = [1] * len(path) if isinstance(path, list) else 1
        
        if isinstance(path, list) and not isinstance(dataset_ratio, list):
            raise ValueError("You should pass dataset ratio for all of your datasets.")
        
        if not isinstance(path, list) and isinstance(dataset_ratio, list):
            raise ValueError("You should pass datasets for all of your dataset ratios.")
        
        if isinstance(path, list) and isinstance(dataset_ratio, list) and len(path) != len(dataset_ratio):
            raise ValueError(f"You have set {len(path)} datasets and {len(dataset_ratio)} dataset ratios, but they should be equal.")
        
        if isinstance(path, list):
            all_datasets = [_load_dataset_from_path(d, test_size) for d in path]
            truncated_datasets = [_get_subset_from_dataset_dict(d, ratio) for d, ratio in zip(all_datasets, dataset_ratio)]
            ds = DatasetDict()
            ds['train'] = concatenate_datasets([d['train'] for d in truncated_datasets])
            ds['test'] = concatenate_datasets([d['test'] for d in truncated_datasets])
        else:
            ds = _load_dataset_from_path(path, test_size)
            ds = _get_subset_from_dataset_dict(ds, dataset_ratio)
    
    return ds


def prepare_generative_row(row: dict, tokenizer, max_length: int) -> dict:
    """
    Prepare a row for generative tasks by applying a chat template and tokenizing the prompt.

    Args:
        row (dict): A dictionary containing the data for a single row, including the prompt.
        tokenizer: The tokenizer to use for processing the prompt.
        max_length (int): The maximum length for the tokenized output.

    Returns:
        dict: A dictionary containing the tokenized prompt ready for input into a model.
    """
    constructed_prompt = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True
    )
    
    return tokenizer(constructed_prompt, truncation=True, padding=True, max_length=max_length, add_special_tokens=False)