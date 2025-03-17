import torch
import json
import os
from typing import Dict, Tuple, Any, List
from transformers import AutoTokenizer


def load_pt_file(file_path: str) -> Dict[str, torch.Tensor]:
    """Load a .pt file and return its contents."""
    return torch.load(file_path)


def check_tensor_validity(tensor: torch.Tensor) -> Tuple[bool, bool, float, float]:
    """Check tensor for NaNs and Infs."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = float(tensor.min())
    max_val = float(tensor.max())
    return has_nan, has_inf, min_val, max_val


def format_tensor_preview(tensor: torch.Tensor, batch_idx: int) -> str:
    """Format a preview of the states tensor for a specific batch."""
    # Get the batch
    batch = tensor[batch_idx]
    # Convert to float to ensure consistent formatting
    batch = batch.float()
    # Get a small preview of actual values (first few elements)
    preview_size = 5
    preview_values = batch.flatten()[:preview_size].tolist()
    preview_str = ", ".join(f"{x:.3f}" for x in preview_values)

    return f"First {preview_size} values: [{preview_str}, ...]"


def decode_input_ids(input_ids: torch.Tensor, model_name: str) -> List[str]:
    """Decode input_ids tensor into text using the correct tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Decode each batch separately
        texts = []
        for batch_idx in range(input_ids.shape[0]):
            batch_ids = input_ids[batch_idx]
            text = tokenizer.decode(batch_ids)
            texts.append(text)
        return texts
    except Exception as e:
        return [f"Error decoding text: {str(e)}"] * input_ids.shape[0]


def inspect_tensor_file(
    bucket: str, file_path: str, s3_path: str
) -> Tuple[Dict[str, Any], str, List[str], Dict[str, Any]]:
    """
    Inspect a .pt file and return shapes, model info, and decoded texts.
    Args:
        file_path: Local path to the .pt file
        s3_path: Full S3 path to the file (for finding cfg.json)
    Returns:
        (shapes_dict, tokenizer_name, decoded_texts, tensor_info)
    """
    print(f"Inspecting file: {file_path}")
    # Load the tensor file
    data = load_pt_file(file_path)

    # Get shapes and check tensors
    shapes = {}
    tensor_info = {}
    if "states" in data:
        states = data["states"]
        shapes["states"] = list(states.shape)
        has_nan, has_inf, min_val, max_val = check_tensor_validity(states)
        tensor_info["states"] = {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "min_val": min_val,
            "max_val": max_val,
            "tensor": states,  # Store for later use in previews
        }
    if "input_ids" in data:
        shapes["input_ids"] = list(data["input_ids"].shape)

    # Get model info from cfg.json in activault root
    try:
        # Get the activault job name (first component of the path)
        activault_job = s3_path.split("/")[0]
        cfg_path = f"/tmp/{activault_job}_cfg.json"

        # Download and read cfg.json
        from s3.utils import create_s3_client

        s3_client = create_s3_client()
        s3_client.download_file(bucket, f"{activault_job}/cfg.json", cfg_path)

        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            model_name = cfg["transformer_config"]["model_name"]

        # Clean up cfg file
        import os

        if os.path.exists(cfg_path):
            os.remove(cfg_path)
    except:
        model_name = "Unknown"

    # Decode input_ids if available
    decoded_texts = []
    if "input_ids" in data and model_name != "Unknown":
        decoded_texts = decode_input_ids(data["input_ids"], model_name)

    return shapes, model_name, decoded_texts, tensor_info
