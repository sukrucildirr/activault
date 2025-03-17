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

import os
import ray
import logging
import time
import yaml
import tempfile
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


def init_ray(address: Optional[str] = None) -> None:
    """Initialize Ray with the given address or environment configuration.

    Args:
        address: Ray cluster address, or None to use environment variables
    """
    if not ray.is_initialized():
        try:
            # Use address if provided, otherwise Ray will use RAY_ADDRESS env var
            ray.init(address=address, ignore_reinit_error=True)
            logger.info(f"Ray initialized: {ray.cluster_resources()}")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise


@ray.remote
class RayActivationWorker:
    """Ray actor for distributed activation collection.

    This worker encapsulates the main processing logic in a Ray-friendly way
    without modifying the core processing functions.
    """

    def __init__(self, machine_index: int, config_path: str):
        """Initialize the worker with its machine index and config path.

        Args:
            machine_index: Index of this worker (equivalent to Slurm machine_index)
            config_path: Path to the configuration file
        """
        self.machine_index = machine_index
        self.config_path = config_path
        self.results = {"status": "initialized"}

    def run(self) -> Dict[str, Any]:
        """Run the activation collection process.

        This method imports and calls the main function to run the collection process,
        capturing logs and results.

        Returns:
            Dict containing status and results
        """
        import sys
        import os
        import json
        from importlib import import_module

        # Set machine index environment variable for compatibility
        os.environ["MACHINE_INDEX"] = str(self.machine_index)

        try:
            # Handle config processing - ray_config section isn't supported by Config class
            if self.config_path.endswith(".yaml"):
                # Create a clean config without ray_config
                with open(self.config_path, "r") as f:
                    config_dict = yaml.safe_load(f)

                # Remove ray_config section if present
                if "ray_config" in config_dict:
                    del config_dict["ray_config"]

                # Create temporary config file
                with tempfile.NamedTemporaryFile(
                    suffix=".yaml", mode="w", delete=False
                ) as temp_file:
                    temp_config_path = temp_file.name
                    yaml.dump(config_dict, temp_file)
            else:
                temp_config_path = self.config_path

            # Use the existing main function from stash.py with temporary config
            sys.argv = [
                "stash.py",
                "--config",
                temp_config_path,
                "--machine",
                str(self.machine_index),
            ]

            # Import and run the main function
            stash = import_module("stash")
            stash.main()

            # Clean up temp file
            if temp_config_path != self.config_path:
                os.remove(temp_config_path)

            self.results = {
                "status": "completed",
                "machine_index": self.machine_index,
                "error": None,
            }

        except Exception as e:
            import traceback

            error_trace = traceback.format_exc()
            logger.error(f"Worker {self.machine_index} failed: {e}\n{error_trace}")
            self.results = {
                "status": "failed",
                "machine_index": self.machine_index,
                "error": str(e),
                "traceback": error_trace,
            }

        return self.results


def calculate_resources_per_worker(config: Dict[str, Any]) -> Dict[str, float]:
    """Calculate Ray resources required per worker.

    Args:
        config: Configuration dictionary with resource requirements

    Returns:
        Dict of Ray resources (e.g., {"GPU": 1, "CPU": 4})
    """
    resources = {}

    # Get resources from config or use defaults
    if "resources" in config:
        resources = config["resources"]
    else:
        # Default resource allocation
        resources = {
            "CPU": 4,
            "GPU": 1,
        }

    return resources


def launch_ray_jobs(
    config_path: str,
    num_total_runs: int,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    ray_address: Optional[str] = None,
    resources_per_worker: Optional[Dict[str, float]] = None,
) -> List[ray.ObjectRef]:
    """Launch Ray workers for activation collection.

    Args:
        config_path: Path to configuration file
        num_total_runs: Total number of runs to execute
        start_idx: Starting machine index
        end_idx: Ending machine index (inclusive)
        ray_address: Ray cluster address
        resources_per_worker: Dict of resources per worker

    Returns:
        List of Ray ObjectRefs for the running jobs
    """
    # Initialize Ray
    init_ray(ray_address)

    # Set end_idx to (num_total_runs - 1) if not specified
    if end_idx is None:
        end_idx = num_total_runs - 1

    # Validate indices
    if start_idx >= num_total_runs or end_idx >= num_total_runs or start_idx > end_idx:
        raise ValueError(
            f"Invalid index range: {start_idx} to {end_idx} (total runs: {num_total_runs})"
        )

    # Create actor options with resource requirements
    actor_options = {}
    if resources_per_worker:
        actor_options["num_cpus"] = resources_per_worker.get("CPU", 4)
        actor_options["num_gpus"] = resources_per_worker.get("GPU", 1)

        # Add custom resources if any
        for k, v in resources_per_worker.items():
            if k not in ["CPU", "GPU"]:
                actor_options[f"resources"] = {k: v}

    # Launch jobs for the specified range
    job_refs = []
    for i in range(start_idx, end_idx + 1):
        logger.info(f"Launching Ray worker {i+1}/{num_total_runs}")
        worker = RayActivationWorker.options(**actor_options).remote(i, config_path)
        job_ref = worker.run.remote()
        job_refs.append(job_ref)

    return job_refs


def monitor_ray_jobs(
    job_refs: List[ray.ObjectRef], poll_interval: int = 10
) -> List[Dict[str, Any]]:
    """Monitor Ray jobs until completion.

    Args:
        job_refs: List of Ray ObjectRefs to monitor
        poll_interval: How often to check job status in seconds

    Returns:
        List of results from completed jobs
    """
    all_results = []
    remaining_refs = list(job_refs)

    while remaining_refs:
        # Check for any completed jobs
        done_refs, remaining_refs = ray.wait(remaining_refs, timeout=poll_interval)

        # Process completed jobs
        for job_ref in done_refs:
            try:
                result = ray.get(job_ref)
                all_results.append(result)
                logger.info(
                    f"Job completed: machine_index={result.get('machine_index')}, status={result.get('status')}"
                )
            except Exception as e:
                logger.error(f"Error getting job result: {e}")
                all_results.append({"status": "error", "error": str(e)})

        # Log progress
        logger.info(f"Progress: {len(all_results)}/{len(job_refs)} jobs completed")

    return all_results
