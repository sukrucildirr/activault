from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from s3.utils import create_s3_client

BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "main")


@dataclass
class Progress:
    current: int = 0
    total: int = 0
    done: bool = False


class S3Operations:
    def __init__(self, bucket: str = BUCKET_NAME):
        self.s3_client = create_s3_client()
        self.bucket = bucket

    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def list_objects(self, prefix: str) -> Tuple[List[Dict], List[str]]:
        """List objects and prefixes at a path."""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        files = []
        folders = set()

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
                # Get folders
                if "CommonPrefixes" in page:
                    for p in page["CommonPrefixes"]:
                        folder = p["Prefix"].rstrip("/").split("/")[-1]
                        folders.add(folder)

                # Get files
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key == prefix:  # Skip the prefix itself
                            continue
                        name = key[len(prefix) :].split("/")[0]
                        if "/" not in name:  # Only direct files
                            files.append({"name": name, "size": obj["Size"], "key": key})
        except Exception as e:
            print(f"\nError listing objects: {str(e)}")

        return files, sorted(list(folders))

    def list_all_objects(self, prefix: str) -> List[Dict[str, Any]]:
        """List all objects under a prefix without delimiter (recursive)."""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                objects.extend(page["Contents"])
        return objects

    def delete_batch(self, batch: List[Dict[str, str]], progress: Optional[Progress] = None) -> int:
        """Delete a batch of objects."""
        if not batch:
            return 0

        try:
            self.s3_client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True}
            )
            if progress:
                progress.current += len(batch)
            return len(batch)
        except Exception as e:
            print(f"\nError deleting batch: {str(e)}")
            return 0

    def count_objects(self, prefix: str) -> int:
        """Count objects with a prefix."""
        count = 0
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                count += len(page["Contents"])
        return count

    def delete_objects(self, prefix: str, progress_callback=None) -> Tuple[int, int]:
        """Delete objects with a prefix recursively."""
        # List all objects first
        all_objects = self.list_all_objects(prefix)
        total_objects = len(all_objects)

        if total_objects == 0:
            return 0, 0

        # Setup progress
        progress = Progress(total=total_objects) if progress_callback else None
        if progress_callback:
            progress_thread = threading.Thread(target=progress_callback, args=(progress,))
            progress_thread.start()

        try:
            # Delete in batches
            batch_size = 1000
            deleted_count = 0
            current_batch = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []

                # Create batches of objects to delete
                for obj in all_objects:
                    current_batch.append({"Key": obj["Key"]})

                    if len(current_batch) >= batch_size:
                        futures.append(
                            executor.submit(self.delete_batch, current_batch.copy(), progress)
                        )
                        current_batch = []

                # Handle remaining objects
                if current_batch:
                    futures.append(executor.submit(self.delete_batch, current_batch, progress))

                # Wait for all deletions to complete
                for future in futures:
                    deleted_count += future.result()

            return deleted_count, total_objects

        finally:
            if progress:
                progress.done = True
                progress_thread.join()

    def read_file(self, key: str) -> str:
        """Read a file's contents."""
        response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def download_file(self, key: str, local_path: str):
        """Download a file from S3 to a local path."""
        try:
            self.s3_client.download_file(self.bucket, key, local_path)
        except Exception as e:
            print(f"Error downloading file {key}: {str(e)}")
