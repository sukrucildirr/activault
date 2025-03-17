#!/usr/bin/env python
"""
Run activation collection jobs on a Ray cluster.

This script provides a parallel alternative to the Slurm-based run_jobs.sh,
allowing Activault to be deployed on Ray clusters without modifying the core functionality.

Usage:
    python scripts/run_ray_jobs.py configs/llama3.3_70b.yaml 8 0 7 --address ray://ray-head:10001 --resources '{"CPU": 32, "GPU": 2}' --wait

Arguments:
    config_path: Path to configuration file
    num_total_runs: Total number of workers to run
    start_idx: Index of first worker (default: 0)
    end_idx: Index of last worker (default: num_total_runs-1)

Options:
    --address: Ray cluster address (default: auto-detect)
    --resources: JSON string with resource requirements per worker
    --wait: Wait for jobs to complete before exiting
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Ray utilities
from pipeline.utils.ray_utils import (
    launch_ray_jobs,
    monitor_ray_jobs,
    calculate_resources_per_worker,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run activation collection jobs on a Ray cluster")

    parser.add_argument("config_path", type=str, help="Path to configuration file")
    parser.add_argument("num_total_runs", type=int, help="Total number of workers to run")
    parser.add_argument(
        "start_idx", type=int, default=0, nargs="?", help="Index of first worker (default: 0)"
    )
    parser.add_argument(
        "end_idx",
        type=int,
        default=None,
        nargs="?",
        help="Index of last worker (default: num_total_runs-1)",
    )

    parser.add_argument("--address", type=str, default=None, help="Ray cluster address")
    parser.add_argument(
        "--resources",
        type=str,
        default=None,
        help='JSON string with resource requirements (e.g. \'{"CPU": 4, "GPU": 1}\')',
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for jobs to complete before exiting"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")


def main():
    """Main entry point for Ray job submission."""
    args = parse_args()

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Configure file logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/ray_jobs_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    # Set end_idx to (num_total_runs - 1) if not specified
    end_idx = args.end_idx if args.end_idx is not None else args.num_total_runs - 1

    # Load config to get resource requirements
    config = load_config(args.config_path)

    # Parse resources JSON if provided
    resources_per_worker = None
    if args.resources:
        try:
            resources_per_worker = json.loads(args.resources)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse resources JSON: {e}")
            sys.exit(1)
    else:
        # Get resources from config or use defaults
        ray_config = config.get("ray_config", {})
        resources_per_worker = calculate_resources_per_worker(ray_config)

    logger.info(
        f"Launching jobs {args.start_idx} to {end_idx} (of total {args.num_total_runs} runs)"
    )
    logger.info(f"Using config: {args.config_path}")
    logger.info(f"Ray address: {args.address or 'auto-detect'}")
    logger.info(f"Resources per worker: {resources_per_worker}")

    # Launch jobs
    try:
        job_refs = launch_ray_jobs(
            args.config_path,
            args.num_total_runs,
            args.start_idx,
            end_idx,
            args.address,
            resources_per_worker,
        )

        logger.info(f"Launched {len(job_refs)} Ray jobs")

        # Write job information to log file
        job_log_file = f"logs/ray_jobs_info_{timestamp}.txt"
        with open(job_log_file, "w") as f:
            f.write("MACHINE | JOB_ID\n")
            f.write("---------------\n")
            for i, job_ref in enumerate(job_refs, start=args.start_idx):
                f.write(f"{i} | {job_ref}\n")

        logger.info(f"Job mapping saved to: {job_log_file}")

        # Wait for jobs to complete if requested
        if args.wait:
            logger.info("Waiting for jobs to complete...")
            results = monitor_ray_jobs(job_refs)

            # Log completion status
            success_count = sum(1 for r in results if r.get("status") == "completed")
            failed_count = sum(1 for r in results if r.get("status") == "failed")

            logger.info(f"Job completion: {success_count} succeeded, {failed_count} failed")

            # Write detailed results
            results_file = f"logs/ray_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Detailed results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Error launching Ray jobs: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
