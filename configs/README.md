# Configuration Files

This directory contains YAML configuration files for the ActiVault pipeline, which is used to extract and store model activations.

## Configuration Structure

Each configuration file follows this general structure:

```yaml
run_name: model_name_dataset       # Format: model_family.model_size.dataset[.template]
num_runs: 1                        # Number of parallel runs to split work across. If you are running distributed, set this to the number of jobs.
transformer_config:                # Model-specific configurations
  model_name: path/to/model        # HuggingFace model ID or path
  dtype: float16                   # Precision (float16, bfloat16, float32)
  cache_dir: /cache                # Local cache directory for models
  max_per_device_memory: 55GB      # Maximum GPU memory per device
  kwargs:                          # Additional model-specific configurations
    load_in_8bit: True
data_config:                       # Dataset and processing configurations
  bucket_name: main                # S3 bucket for storing results
  data_key: dataset:TEMPLATE       # Dataset key with optional template
  n_batches: 100                   # Number of batches to process
  seq_length: 2048                 # Maximum sequence length
  batch_size: 4                    # Batch size for processing
  seed: 42                         # Random seed for reproducibility
  skip_cache: False                # Skip caching very long texts
  start_batch: 0                   # Starting batch number
  clean_added_tokens: True         # Whether to skip hidden states for special tokens cleaned from outputs
  clean_default_system_prompt: True # Whether to skip hidden states for default system prompt (will ignore custom system prompts)
upload_config:                     # Upload behavior and hook specifications
  batches_per_upload: 8            # Number of batches to accumulate before upload
  hooks:                           # List of activation hooks to capture
    - "blocks.24.hook_resid_post"
    - "blocks.36.hook_mlp_post"
```

## Configuration Fields

### Root Level

- `run_name` (required): Unique identifier for this run. Format: `model_family.model_size.dataset[.template]`
- `num_runs` (required): Number of parallel runs to split the work across. For distributed processing.

### Transformer Config

- `model_name` (required): HuggingFace model identifier or path
- `dtype` (optional, default: float16): Model precision (float16, bfloat16, float32)
- `cache_dir` (optional): Directory to cache downloaded models
- `max_per_device_memory` (optional): Maximum GPU memory allocated per device

### Data Config

- `bucket_name` (required): S3 bucket name for storing results
- `data_key` (required): Dataset key with optional template format
- `n_batches` (required): Total number of batches to process
- `seq_length` (required): Maximum sequence length for processing
- `batch_size` (required): Batch size for processing
- `seed` (optional, default: 42): Random seed for reproducibility
- `skip_cache` (optional, default: False): Whether to skip caching very long texts
- `start_batch` (optional, default: 0): Starting batch index (for resuming or parallel jobs)
- `clean_added_tokens` (optional, default: False): Whether to ignore special tokens in the output

### Upload Config

- `batches_per_upload` (required): Number of batches to accumulate before uploading to S3
- `hooks` (required): List of activation hooks to capture (format: "blocks.{layer}.hook_resid_post")

## Example Configurations

1. **Llama 3.3 70B** ([llama3.3_70b.yaml](llama3.3_70b.yaml))
   - 70 billion parameter model from Meta
   - FP16 precision
   - Captures activations from 4 different layers

2. **Gemma 3 27B** ([gemma3_27b.yaml](gemma3_27b.yaml))
   - 27 billion parameter model from Google
   - BF16 precision 

## Creating a New Configuration

To create a new configuration:

1. Copy an existing configuration file that most closely matches your needs
2. Modify the following fields:
   - `run_name`: Change to match your new model and dataset
   - `model_name`: Update to the correct HuggingFace model ID
   - Adjust batch sizes, sequence lengths, and other parameters as needed
   - Update hook positions based on the model's architecture

## Usage

Configuration files are used with the ActiVault pipeline:

```bash
python stash.py --config configs/your_config.yaml
```

For distributed processing across multiple machines:

```bash
python stash.py --config configs/your_config.yaml --machine 0
python stash.py --config configs/your_config.yaml --machine 1
# ... and so on
``` 