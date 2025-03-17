# Activault
**Activault** is an activation data engine that dramatically reduces costs for training interpreter models on frontier LLMs:
- Collects and stores model activations (the model's "mental state") efficiently using S3 object storage, reducing activation management costs by 4-8x.
- Enables reproducible and shareable interpretability research through standardized object storage.
- Maintains peak efficiency and throughput while handling petabyte-scale activation datasets.


You can read about Activault in our [blog post](https://www.tilderesearch.com/blog/activault).

> ⚠️ **CRITICAL WARNING**  
> Streaming/storing activations with Activault can be expensive ($$$) and slow if care is not taken before launching large-scale jobs. We recommend users set up their compute environment in the same region/data center as their s3 solution to ensure minimal latency and avoid egress fees. We also **strongly** recommend users consult the pricing page for their s3 solution to ensure they understand the costs associated with their jobs.

## 💾 Activault principles

When designing Activault, we considered the tradeoffs between computing activations on-the-fly vs storing them on disk vs storing them in S3. Here's how the approaches compare:

| Aspect | On-the-fly | Local Cache | Naive S3 | Activault |
|--------|------------|-------------|----------|-----------|
| Setup Complexity | ✅ Easy | ✅ Easy | ❌ Hard | ✅ Easy |
| Write Performance | ✅ Fast | ✅ Fast | ❌ Slow | ✅ Fast |
| Read Performance | ✅ Fast | ✅ Fast | ❌ Slow | ✅ Fast |
| Efficiency | ❌ Enormously inefficient as activations must be regenerated across runs | ✅ Efficient | ✅ Efficient | ✅ Efficient |
| Reproducibility | ❌ Poor | ✅ Good | ✅ Guaranteed | ✅ Guaranteed |
| Token Context | ❌ Autointerp requires recomputing | ✅ Good | ❌ Poor (no tokens saved) | ✅ Tokens saved with data |
| Shareability | ❌ Vanishes after training | ❌ Terrible | ✅ Guaranteed | ✅ Guaranteed |
| Storage Cost | ✅ None | ❌ Very expensive | ✅ Cheap | ✅ Cheap |
| Storage Availability | ✅ N/A | ❌ Very low | ✅ High | ✅ High |

## Table of Contents

- [🔧 Setup](#setup) - Installation and AWS credential configuration
- [📊 Collecting Activations](#collecting-activations) - Core pipeline for gathering model activations
- [🚀 Running Collection Jobs](#running-collection-jobs)
  - [Using Slurm](#using-slurm-traditional-method) - Run on HPC clusters
  - [Using Ray](#using-ray-distributed-computing) - Distributed computing setup
  - [Running Locally](#running-locally) - Single machine execution
- [🔍 Checking the Outputs: S3 Shell](#checking-the-outputs-s3-shell) - Tools for inspecting collected data
- [📈 Using Activations with RCache](#using-your-activations-with-rcache) - Efficient streaming interface
- [❓ FAQs](#faqs) - Common questions and answers
- [💾 Local vs S3 Storage](#local-storage-vs-s3-storage) - Storage approach comparisons
- [👥 Credits](#credits) - Attribution and inspiration

## 🔧 Setup

```
pip install uv
uv sync --no-build-isolation
uv pip install -e .
```

Make sure your AWS credentials are set.
```
export AWS_ACCESS_KEY_ID=<your_key>
export AWS_SECRET_ACCESS_KEY=<your_secret>
export S3_ENDPOINT_URL=<your_endpoint_url>
```

## 📊 Collecting Activations

Use one of the pre-existing configs in `configs/` or create your own. We provide configs for several frontier open-weight models out-of-box. 

The collection pipeline:

1. Loads a transformer model and hooks into the specified layers and modules
2. Streams text data according to a specified data_key (mappings defined in `pipeline/data/datasets.json`) through the model in batches
3. For each hook (e.g., residual stream, attention outputs):
   - Collects activations and their corresponding input tokens
   - Concatenates multiple batches into "megabatch" files
   - Computes running statistics (mean, std, norm)
   - Uploads to S3 asynchronously

Each hook's data is stored in its own directory:
```
s3://{bucket}/{run_name}/
  ├── cfg.json              # Collection config and model info
  └── {hook_name}/
      ├── metadata.json     # Shape and dtype info
      ├── statistics.json   # Running statistics
      └── {uuid}--{n}.pt   # Megabatch files
```

## 🚀 Running Collection Jobs

> 📢 **IMPORTANT**  
> Ensure `n_runs` in the config file is set to the total number of runs you want to launch before runnign large-scale distributed jobs. if this is not done, you will generate redundant data.

### Using Slurm (Traditional method)

#### 1. Basic Single-Node Usage

For a simple job on a single Slurm node:
```bash
sbatch scripts/collect.slurm configs/your_config.yaml
```

#### 2. Running Distributed Jobs

To run multiple distributed jobs across different nodes:
```bash
./scripts/run_slurm_jobs.sh configs/your_config.yaml 8 0 7
```

**Key Arguments:**
- `configs/your_config.yaml`: Path to configuration file
- `8`: Total number of workers to spawn
- `0 7`: Start and end indices for worker assignment (will launch jobs for indices 0-7)

The script will generate a log file mapping machine indices to Slurm job IDs.

#### 3. Configuration

Slurm job parameters (CPUs, GPUs, memory, etc.) can be adjusted by editing `scripts/collect.slurm`. Important parameters:
```bash
#SBATCH --cpus-per-task=16     # CPUs per task
#SBATCH --gres=gpu:1           # GPUs per node
#SBATCH --mem=250G             # Memory per node
```

### Using Ray

Be sure to start a Ray cluster.
```bash
# Start Ray locally
ray start --head

# On head node
ray start --head --port=6379

# On worker nodes
ray start --address=<head-node-ip>:6379
```

Running a single worker:
```bash
python scripts/run_ray_jobs.py configs/your_config.yaml 1 0 0 --resources '{"CPU": 32, "GPU": 2}' --wait
```

Running distributed jobs (8 workers from index 0-7):
```bash
python scripts/run_ray_jobs.py configs/your_config.yaml 8 0 7 --resources '{"CPU": 32, "GPU": 2}' --wait
```

**Key Arguments:**
- `configs/your_config.yaml`: Path to configuration file
- `8`: Total number of workers to spawn
- `0 7`: Start and end indices for worker assignment
- `--resources`: CPU and GPU allocation per worker (JSON format)
- `--address`: Optional Ray cluster address (if not using environment variable)
- `--wait`: Wait for all jobs to complete and show results

Check Ray's dashboard periodically (typically at http://localhost:8265) for cluster status.

### Running locally

To run the pipeline locally, you can use the Activault CLI:
```bash
activault collect --config configs/your_config.yaml
```

Alternatively, you can run it directly:
```bash
python stash.py --config configs/your_config.yaml
```

For distributed execution, specify the machine index:
```bash
activault collect --config configs/your_config.yaml --machine 0
```

## 🔍 Checking the Outputs: S3 Shell

After running the pipeline, you can check the outputs by using our S3 shell.

First, make sure your S3 bucket name is set:
```bash
export S3_BUCKET_NAME=<your_bucket>
```

Then, launch the S3 shell using the Activault CLI:
```bash
activault s3
```

In the S3 shell, navigate to your run directory and use these commands:
- `ls` - List files and directories 
- `cd directory_name` - Change directory
- `filecount` - Count the number of files in the current directory and subdirectories
- `sizecheck` - Calculate the total size of files in the current directory
- `inspect <file_index>` - Inspect a specific megabatch file

Example inspection output:

```
s3://main/testing/blocks.24.hook_resid_post> inspect 1
Inspecting file: /tmp/0f909221-ff28-4a94-a43f-cfe973e835cf--5_0.saved.pt

PT File Inspection:
----------------------------------------
Model: meta-llama/Llama-3.3-70B-Instruct

Tensor Shapes:
  states: [32, 2048, 8192]
  input_ids: [32, 2048]

States Tensor Check:
  No NaNs: ✅
  No Infs: ✅
  Value range: [-6.941, 4.027]

First 4 batches (first 250 chars each):
----------------------------------------
Batch 0: Given a triangle... (truncated)
Batch 1: Neonatal reviewers indicated... (truncated)
Batch 2: Is there a method... (truncated)
Batch 3: John visits three different... (truncated)

Enter batch number (0-31) to view full text, or 'q' to quit:
```

## 📈 Using Activations with RCache

RCache provides a simple interface for efficiently streaming large activation datasets from S3 without memory or I/O bottlenecks.

1. RCache maintains a small buffer (default: 2 files) in memory
2. While you process the current megabatch, the next ones are downloaded asynchronously
3. After a brief initial load (<30s), processing should never be bottlenecked by the downloads/streamings

### Quick usage:
```python
cache = S3RCache.from_credentials(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    s3_prefix="run_name/hook_name",
    device="cuda",  # or "cpu"
    return_ids=True  # if you need the input tokens
)

for batch in cache:
    states = batch["states"]        # shape: [n_batches, seq_len, d_model]
    input_ids = batch["input_ids"]  # shape: [n_batches, seq_len]
    # ... process batch ...

cache.finalize()  # clean up
```

See `retrieve.py` for a complete example.

## ❓ FAQs

### Can I use this to train my own sparse autoencoder?

Yes! This is the intended goal. RCache can be used out of box in SAE training workflows. It supports blazing fast throughput to ensure training is always FLOP-bottlencked, not IO-bottlencked.

### What platforms is Activault built for?

Activault is designed to be compatible with any S3-style object storage solution. We performed most of our testing on Nebius S3 and have also tested on AWS S3. It is possible that other platforms may encounter issues, and we welcome contributions to expand support.

### Why does Activault use `transformers` instead of a more efficient inference library such as `vllm`?

A few reasons:
1. The main reason is that the bottleneck is upload speed not throughput. We experimented with using much faster internal serving engines but the main process ran far ahead of the save processes and there was no real gain in overall time.
2. Activault does not use the `generate` method and prefill speeds are more comparable between the public libraries.
3. Activault should be compatible with as many models as possible.
4. `vllm` does not play nice with procuring internal states.

That said, we welcome contributions to expand Activault's support for more efficient inference libraries.

### Why does Activault not use `nnsight` or `transformer-lens`?

We do not use libraries such as `nnsight` or `transformer-lens` to minimize dependencies and potential failure points, and to ensure maximal compatibility with a wide range of models.

### Activault doesn't (support vision models/get the activations I need/work for my storage solution)!

We welcome contributions! Please open an issue or PR. We are releasing Activault as a community tool to enable low-resource users to collect activations, run experiments, and share data to analyze frontier open-weight models. 


## 👥 Credits

This repo was originally inspired by [Lewington-pitsos/sache](https://github.com/Lewington-pitsos/sache), which is linked in the LessWrong post [here](https://www.lesswrong.com/posts/AtJdPZkMdsakLc6mB/training-a-sparse-autoencoder-in-less-than-30-minutes-on).

## 📄 License

Activault is licensed under the [Apache License 2.0](LICENSE).

This is a permissive license that allows you to:
- Use the code commercially
- Modify the code
- Distribute your modifications
- Use patent claims of contributors (if applicable)
- Sublicense and/or distribute the code

Key requirements:
- Include a copy of the license in any redistribution
- Clearly mark any changes you make to the code
- Include the original copyright notices

The full text of the license is available in the [LICENSE](LICENSE) file.
