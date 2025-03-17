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

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from typing import Dict
from pipeline.data.dataloader import DataLoader
from pipeline.config import Config
from pipeline.vault import HookUploader
from s3.utils import create_s3_client
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


def generate_activations(
    model: AutoModelForCausalLM,
    loader: DataLoader,
    config: Config,
    uploaders: Dict[str, HookUploader] = None,
    hook_activations: Dict[str, Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Main activation generation loop.

    Args:
        model: The transformer model (already on correct device)
        loader: DataLoader instance
        config: Configuration object
        uploaders: Dictionary mapping hook names to their uploaders
    """
    # Set d_model in config
    config.d_model = model.config.hidden_size

    # Create S3 client for config saving
    s3_client = create_s3_client()

    # Load existing config and stats if available
    existing_config = Config.load_from_s3(s3_client, config.data_config["bucket_name"])
    if existing_config:
        logger.info(f"Resuming run {config.run_name} from {existing_config.total_tokens} tokens")
        config.total_tokens = existing_config.total_tokens
        config.n_total_files = existing_config.n_total_files
        config.batches_processed = existing_config.batches_processed

        # Skip tokens based on existing total tokens with an offset
        tokens_to_skip = existing_config.total_tokens + 3000
        loader.skip_tokens(tokens_to_skip)

    # Initialize statistics tracking
    hooks = config.upload_config["hooks"]
    means = {hook: torch.zeros(model.config.hidden_size, device=model.device) for hook in hooks}

    # M2 stores sum of squared differences from the mean (for Welford's algorithm)
    M2s = {hook: torch.zeros(model.config.hidden_size, device=model.device) for hook in hooks}
    counts = {hook: 0 for hook in hooks}  # Track number of samples per dimension
    norm_sums = {hook: torch.zeros(1, device=model.device) for hook in hooks}  # Track sum of norms
    norm_counts = {hook: 0 for hook in hooks}  # Track count for norms

    # Load existing statistics if available
    for hook in hooks:
        stats = Config.load_hook_statistics(
            s3_client, config.run_name, hook, config.data_config["bucket_name"]
        )
        if stats:
            logger.info(f"Loading existing statistics for {hook}")
            means[hook] = torch.tensor(stats["mean"], device=model.device)
            if "M2" in stats:
                M2s[hook] = torch.tensor(stats["M2"], device=model.device)
            else:
                # If M2 not available, approximate from std (for backward compatibility)
                std = torch.tensor(stats["std"], device=model.device)
                M2s[hook] = std * std * config.batches_processed

            counts[hook] = config.batches_processed
            norm_sums[hook] = stats.get("norm", 0.0) * config.batches_processed
            norm_counts[hook] = config.batches_processed

    # Prepare for activation collection
    layers = {hook: int(hook.split(".")[1]) for hook in hooks}

    # Initialize batches processed from config
    batches_processed = config.batches_processed

    # Calculate tokens to skip based on batches processed
    tokens_to_skip = (
        config.batches_processed
        * config.data_config["batch_size"]
        * config.data_config["seq_length"]
    )
    loader.skip_tokens(tokens_to_skip)

    # Main loop
    model.eval()
    with torch.no_grad():
        total_batches = (
            loader.batches_per_machine
            if loader.batches_per_machine is not None
            else config.data_config["n_batches"]
        )
        pbar = tqdm(total=total_batches)
        pbar.update(batches_processed)

        # Generate a new UUID for each batch group
        current_group_uuid = str(uuid4())
        group_batch_count = 0

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= config.data_config["n_batches"]:
                break

            # Move batch to model's device
            batch = {k: v.to(device=model.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch, output_hidden_states=True)

            # Extract activations
            activations = {}
            for hook in hooks:
                if hook in hook_activations:
                    activations[hook] = {
                        "states": hook_activations[hook]["states"],
                        "input_ids": batch["input_ids"],
                    }
                else:
                    activations[hook] = {
                        "states": outputs.hidden_states[layers[hook] + 1],
                        "input_ids": batch["input_ids"],
                    }

            try:
                # Clean special tokens from activations (e.g. BOS)
                cleaned_activations = {}
                for hook in hooks:
                    cleaned_input_ids, cleaned_states = loader.clean_batch(
                        activations[hook]["input_ids"], activations[hook]["states"]
                    )
                    cleaned_activations[hook] = {
                        "states": cleaned_states,
                        "input_ids": cleaned_input_ids,
                    }
            except Exception as e:
                logger.warning(f"SKIPPING BATCH {batch_idx} due to error: {e}")
                continue

            # Update total tokens
            config.total_tokens += (
                config.data_config["batch_size"] * config.data_config["seq_length"]
            )

            # Compute statistics and move to CPU
            if uploaders:
                any_file_uploaded = False
                for hook in hooks:
                    # Get current batch activations
                    states = cleaned_activations[hook]["states"]
                    N, T = states.shape[0], states.shape[1]
                    total_tokens = N * T

                    # Update statistics using Welford's online algorithm
                    counts[hook] += total_tokens

                    # Calculate deltas for Welford's algorithm (vectorized)
                    delta = states.mean(dim=(0, 1)) - means[hook]
                    means[hook] += delta * (total_tokens / counts[hook])

                    # Calculate contribution to M2
                    delta2 = states.mean(dim=(0, 1)) - means[hook]
                    M2s[hook] += total_tokens * delta * delta2

                    # Update norm statistics
                    norm_sums[hook] += torch.norm(states, dim=2).sum().item()
                    norm_counts[hook] += total_tokens

                    # Move to CPU for saving
                    cpu_activations = {
                        "states": states.to(device="cpu", non_blocking=True),
                        "input_ids": cleaned_activations[hook]["input_ids"].to(
                            device="cpu", non_blocking=True
                        ),
                    }

                    # Append to uploader with the current group UUID
                    file_id = uploaders[hook].append(cpu_activations, current_group_uuid)
                    if file_id:
                        any_file_uploaded = True

                if any_file_uploaded:
                    config.n_total_files += 1
                    # Generate new UUID for next group since we just uploaded
                    current_group_uuid = str(uuid4())
                    group_batch_count = 0
                else:
                    group_batch_count += 1

            # Update batches processed
            batches_processed += 1

            # Save config periodically only from machine index 0
            if (
                config.machine_index == 0
                and batches_processed % config.upload_config["batches_per_upload"] == 0
            ):
                # Update config with the current state
                config.batches_processed = batches_processed

                # Save config in a non-blocking way
                config.save_to_s3(s3_client, blocking=False)

                # Save statistics
                if uploaders:
                    for hook in hooks:
                        # Extract final statistics from running calculations
                        mean = means[hook].cpu()

                        # Calculate standard deviation from M2
                        variance = M2s[hook] / counts[hook]
                        std = torch.sqrt(variance).cpu()

                        # Calculate average norm
                        norm = norm_sums[hook] / norm_counts[hook] if norm_counts[hook] > 0 else 0.0

                        # Also save M2 for future resumption
                        uploaders[hook].save_stats(mean, std, norm, M2=M2s[hook].cpu())

            pbar.update(1)

        pbar.close()

        # Save final config and statistics only from machine index 0
        if config.machine_index == 0:
            config.batches_processed = batches_processed
            config.save_to_s3(s3_client, blocking=True)  # Block on final save

            if uploaders:
                for hook in hooks:
                    # Extract final statistics from running calculations
                    mean = means[hook].cpu()

                    # Calculate standard deviation from M2
                    variance = M2s[hook] / counts[hook]
                    std = torch.sqrt(variance).cpu()

                    # Calculate average norm
                    norm = norm_sums[hook] / norm_counts[hook] if norm_counts[hook] > 0 else 0.0

                    # Also save M2 for future resumption
                    uploaders[hook].save_stats(mean, std, norm, M2=M2s[hook].cpu())
                    uploaders[hook].finalize()
