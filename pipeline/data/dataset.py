"""Copyright (2025) Tilde Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union, Callable
from datasets import load_dataset, Dataset, IterableDataset
import logging

logger = logging.getLogger(__name__)


def load_dataset_config() -> Dict[str, Any]:
    """Load dataset configurations from datasets.json.

    The configuration defines how each dataset should be loaded,
    including templates and filtering conditions.

    Returns:
        dict: Mapping of dataset names to their configurations

    Raises:
        FileNotFoundError: If datasets.json is not found
        json.JSONDecodeError: If datasets.json is invalid
    """
    config_path = Path(__file__).parent / "datasets.json"
    with open(config_path) as f:
        return json.load(f)


def apply_filter_conditions(
    dataset: Union[Dataset, IterableDataset], conditions: Dict[str, Any]
) -> Union[Dataset, IterableDataset]:
    """Apply filtering conditions to a dataset.

    Args:
        dataset: Input dataset to filter
        conditions: Dictionary mapping column names to required values

    Returns:
        Dataset: Filtered dataset containing only matching rows

    Example:
        ```python
        filtered_ds = apply_filter_conditions(ds, {"language": "en"})
        ```
    """
    for column, value in conditions.items():
        dataset = dataset.filter(lambda x: x[column] == value)
    return dataset


def deduplicate_dataset(
    dataset: Union[Dataset, IterableDataset], conditions: Dict[str, Any]
) -> Union[Dataset, IterableDataset]:
    """Deduplicate dataset rows based on specified columns.

    Args:
        dataset: Input dataset to deduplicate
        conditions: Dictionary specifying columns to check for duplicates

    Returns:
        Dataset: Deduplicated dataset

    Example:
        ```python
        deduped_ds = deduplicate_dataset(ds, {"url": True})
        ```
    """
    for column, value in conditions.items():
        seen_values = set()

        def dedup_filter(example):
            value = example[column]
            if value in seen_values:
                return False
            seen_values.add(value)
            return True

        dataset = dataset.filter(dedup_filter)
    return dataset


def make_template_fn(template: str, columns: List[str], dataset_name: str) -> Callable:
    """Create a function that formats examples using a template.

    Args:
        template: String template with {column} placeholders
        columns: List of column names to use in template
        dataset_name: Name of dataset (for error messages)

    Returns:
        Callable: Function that formats examples using the template

    Raises:
        KeyError: If required columns are missing
        ValueError: If template produces empty text

    Example:
        ```python
        template_fn = make_template_fn(
            "Question: {question}\nAnswer: {answer}",
            ["question", "answer"],
            "qa_dataset"
        )
        ```
    """

    def apply_template(example):
        try:
            text = template.format(**{col: example[col] for col in columns})
            if not text or text.isspace():
                raise ValueError(f"Empty text generated for {dataset_name}")
            return {"text": text}
        except KeyError as e:
            raise KeyError(f"Missing required column {e} for dataset {dataset_name}")

    return apply_template


def make_chat_template_fn(template: List[str], columns: List[str], dataset_name: str) -> Callable:
    """Create a function that formats examples into chat format.

    Args:
        template: List of role names ("user", "assistant", etc.)
        columns: List of column names containing messages
        dataset_name: Name of dataset (for error messages)

    Returns:
        Callable: Function that formats examples into chat format

    Raises:
        KeyError: If required columns are missing

    Example:
        ```python
        chat_fn = make_chat_template_fn(
            ["user", "assistant"],
            ["question", "answer"],
            "dialogue_dataset"
        )
        ```
    """

    def apply_chat_template(example):
        try:
            entry = []
            for role, column in zip(template, columns):
                entry.append({"role": role, "content": example[column]})
            return {"text": entry}
        except KeyError as e:
            raise KeyError(f"Missing required column {e} for dataset {dataset_name}")

    return apply_chat_template


def load_dataset_by_key(
    dataset_key: str, split: str = "train", streaming: bool = True
) -> IterableDataset:
    """Load a single dataset according to its configuration.

    This function loads a dataset and applies templates and filters as configured.

    Args:
        dataset_key: Name of the dataset configuration to use
        split: Dataset split to load ("train", "validation", etc.)
        streaming: Whether to use streaming mode

    Returns:
        IterableDataset: Processed dataset

    Raises:
        KeyError: If dataset_key is not found
        Exception: If dataset loading or processing fails

    Example:
        ```python
        dataset = load_dataset_by_key(
            dataset_key="web_instruct",
            split="train",
            streaming=True
        )
        ```
    """
    config = load_dataset_config()[dataset_key]
    logger.info(f"\nLoading dataset: {dataset_key}")

    # Load dataset
    ds = load_dataset(config["name"], config.get("subset"), split=split, streaming=streaming)

    # Apply filters if specified
    if "filter" in config:
        ds = apply_filter_conditions(ds, config["filter"])

    if "deduplicate" in config:
        ds = deduplicate_dataset(ds, config["deduplicate"])

    # Handle different column structures
    if "template" in config:
        # Apply template if specified
        ds = ds.map(make_template_fn(config["template"], config["columns"], dataset_key))
    elif "chat_template" in config:
        ds = ds.map(make_chat_template_fn(config["chat_template"], config["columns"], dataset_key))
    else:
        # If no template, map the specified column to 'text'
        column = config["columns"][0]
        if column != "text":
            ds = ds.map(lambda x: {"text": x[column]})

    # Select only the text column
    ds = ds.select_columns(["text"])
    logger.info(f"Successfully loaded {dataset_key}")

    return ds
