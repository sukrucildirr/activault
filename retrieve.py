"""
RCache efficiently streams transformer activation data from S3 for training interpreter models.
It maintains a small buffer of megabatch files (each containing multiple batches concatenated together during uploads)
and asynchronously downloads the next files while the current ones are being processed.

After a brief initial load (<30s), training should never bottlenecked by downloads since they happen asynchronously in the background.

Copyright (2025) Tilde Research Inc.

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

from transformers import AutoTokenizer
from pipeline.vault import S3RCache
from s3.utils import create_s3_client
import os
import json
import logging

# Constants
RUN_NAME = "llama3.3_70b"  # Base run name without hook
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "main")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_first_hook_prefix(run_name, bucket_name):
    """Get the first available hook prefix for the run."""
    s3_client = create_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{run_name}/", Delimiter="/")
    if "CommonPrefixes" in response:
        # Get first hook directory
        first_hook = response["CommonPrefixes"][0]["Prefix"].rstrip("/")
        return first_hook
    return None


def get_model_name_from_config(run_name, bucket_name):
    """Get model name from the run's config file."""
    s3_client = create_s3_client()
    cfg_path = f"/tmp/{run_name}_cfg.json"
    s3_client.download_file(bucket_name, f"{run_name}/cfg.json", cfg_path)
    with open(cfg_path, "r") as f:
        model_name = json.load(f)["transformer_config"]["model_name"]
    os.remove(cfg_path)
    return model_name


def inspect_batch(states, input_ids, tokenizer):
    """Helper function to inspect a batch of activations and tokens."""
    logger.info(f"States shape: {states.shape}")
    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"\nStats: mean={states.mean().item():.4f}, std={states.std().item():.4f}")
    logger.info(f"Sample text: {tokenizer.decode(input_ids[0])[:100]}...")


def main():
    logger.info("Demo: Reading transformer activations from S3 cache")

    # Get first available hook prefix
    prefix = get_first_hook_prefix(RUN_NAME, BUCKET_NAME)
    if not prefix:
        logger.error(f"No hooks found for run {RUN_NAME}")
        return
    logger.info(f"Using hook prefix: {prefix}")

    # Initialize tokenizer
    model_name = get_model_name_from_config(RUN_NAME, BUCKET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize cache reader
    cache = S3RCache.from_credentials(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_prefix=prefix,
        bucket_name=BUCKET_NAME,
        device="cpu",
        buffer_size=2,
        return_ids=True,
    )

    logger.info("\nReading first two megabatch files from S3...")
    logger.info("Each file contains n_batches_per_file batches concatenated together")
    logger.info("Format: [n_batches_per_file, sequence_length, hidden_dim]\n")

    # Inspect a few batches
    for batch_idx, batch in enumerate(cache):
        if batch_idx >= 2:
            break
        inspect_batch(batch["states"], batch["input_ids"], tokenizer)

    cache.finalize()


if __name__ == "__main__":
    main()
