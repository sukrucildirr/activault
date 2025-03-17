#!/bin/bash
#SBATCH --job-name=activault           # Job name
#SBATCH --output=logs/activault_%j.out # Output file (%j = job ID)
#SBATCH --error=logs/activault_%j.err  # Error file (%j = job ID)
#SBATCH --time=200:00:00             # Time limit (200 hours)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --gres=gpu:2                # Number of GPUs (2 in this case)
#SBATCH --mem=250G                   # Memory per node
#SBATCH --partition=main             # Partition/queue name
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications

# Activate virtual environment
source .venv/bin/activate

#Create cache directory
mkdir -p /tmp/.cache/huggingface

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Default to default.yaml and machine index 0
CONFIG_PATH=${1:-configs/default.yaml}
MACHINE_INDEX=${2:-0}

# Run the data generation script with machine index
srun python stash.py --config $CONFIG_PATH --machine $MACHINE_INDEX

# Optional: Copy output files to a backup location
# rsync -av outputs/ /scratch/$USER/activault_outputs/ 
