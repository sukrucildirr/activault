#!/bin/bash

# Usage: ./run_all.sh <config_path> <num_total_runs> <start_run_idx> <end_run_idx>
# Example: ./run_all.sh configs/default.yaml 8 2 5
#   - Will run jobs for indices 2,3,4,5 out of total 8 runs

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <config_path> <num_total_runs> <start_run_idx> <end_run_idx>"
    echo "Example: $0 configs/llama3.3_70b.yaml 8 2 5"
    exit 1
fi

CONFIG_PATH=$1
NUM_TOTAL_RUNS=$2
START_IDX=$3
END_IDX=$4

# Validate indices
if [ $START_IDX -ge $NUM_TOTAL_RUNS ] || [ $END_IDX -ge $NUM_TOTAL_RUNS ] || [ $START_IDX -gt $END_IDX ]; then
    echo "Error: Invalid index range. Ensure start_idx and end_idx are within [0, num_total_runs-1]"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/run_all_${TIMESTAMP}.txt"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "IMPORTANT: Please ensure `n_runs` in the config file is set to $NUM_TOTAL_RUNS."

echo "Launching jobs $START_IDX to $END_IDX (of total $NUM_TOTAL_RUNS runs) using config: $CONFIG_PATH"
echo "Logging job mapping to: $LOG_FILE"

# Create header for log file
echo "MACHINE | JOB" > "$LOG_FILE"
echo "---------------" >> "$LOG_FILE"

# Launch jobs for the specified range
for ((i=$START_IDX; i<=$END_IDX; i++)); do
    echo "Launching job $((i+1))/$NUM_TOTAL_RUNS with machine_index $i"
    # Capture the job ID from sbatch output
    JOB_ID=$(sbatch scripts/collect.slurm "$CONFIG_PATH" "$i" | grep -o '[0-9]\+')
    echo "$i | $JOB_ID" >> "$LOG_FILE"
    sleep 1  # Small delay to prevent potential race conditions
done

echo -e "\nAll jobs submitted. Use 'squeue' to check their status."
echo "Job mapping saved to: $LOG_FILE" 