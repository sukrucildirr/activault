from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
import os
from pathlib import Path
from .s3_operations import S3Operations, Progress
from .sanity import inspect_tensor_file, format_tensor_preview
from .history import CommandHistory


@dataclass
class IndexedItem:
    index: int
    name: str
    is_folder: bool
    stats: Dict[str, Any]


class S3Shell:
    def __init__(self):
        self.s3 = S3Operations()
        self.current_path: List[str] = []
        self.indexed_items: List[IndexedItem] = []
        self.last_output: str = ""

        # Setup command history
        histfile = os.path.expanduser("~/.s3shell_history")
        self.history = CommandHistory(histfile)

        # ANSI color codes
        self.BLUE = "\033[34m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.RESET = "\033[0m"

    def check_mark(self, condition: bool) -> str:
        """Return emoji check mark or X based on condition."""
        if condition:
            return f"{self.GREEN}✅{self.RESET}"
        return f"{self.RED}❌{self.RESET}"

    def get_current_prefix(self) -> str:
        """Get the current S3 prefix based on path."""
        return "/".join(self.current_path + [""] if self.current_path else [])

    def list_items(self) -> List[IndexedItem]:
        """List items in current path."""
        prefix = self.get_current_prefix()
        files, folders = self.s3.list_objects(prefix)

        # Create indexed items
        items = []
        self._index_map = {}  # For O(1) index lookup
        self._folder_map = {}  # For O(1) folder name lookup
        idx = 1

        # Add folders first
        for folder in folders:
            item = IndexedItem(index=idx, name=folder, is_folder=True, stats={})
            items.append(item)
            self._index_map[idx] = item
            self._folder_map[folder] = item
            idx += 1

        # Add files
        for file in files:
            item = IndexedItem(
                index=idx, name=file["name"], is_folder=False, stats={"size": file["size"]}
            )
            items.append(item)
            self._index_map[idx] = item
            idx += 1

        self.indexed_items = items
        return items

    def format_listing(self, items: List[IndexedItem]) -> str:
        """Format items listing."""
        output = []

        # Current path
        path = "/" + "/".join(self.current_path)
        output.append(f"\nCurrent path: {path}")
        output.append("=" * 80)

        # List items or show empty directory message
        if not items:
            output.append("(empty directory)")
        else:
            for item in items:
                prefix = "📁" if item.is_folder else "📄"
                if item.is_folder:
                    output.append(f"{item.index:3d}. {prefix} {item.name}")
                else:
                    output.append(
                        f"{item.index:3d}. {prefix} {item.name} ({self.s3.format_size(item.stats['size'])})"
                    )

        return "\n".join(output)

    def cmd_filecount(self, args: List[str]) -> str:
        """Count files in folders at current level."""
        items = self.indexed_items
        folders = [item for item in items if item.is_folder]

        if not folders:
            return "No folders in current directory"

        output = ["\nFile Count Check:", "-" * 40]

        # Count files in each folder
        folder_counts = {}
        for folder in folders:
            folder_prefix = self.get_current_prefix() + folder.name + "/"
            count = self.s3.count_objects(folder_prefix)
            folder_counts[folder.name] = count
            output.append(f"{folder.name}: {count} files")

        # Check if all counts are the same
        counts = list(folder_counts.values())
        same_count = len(set(counts)) == 1
        output.append(f"\nAll folders have same file count? {self.check_mark(same_count)}")

        return "\n".join(output)

    def cmd_sizecheck(self, args: List[str]) -> str:
        """Check .pt file sizes in current directory and one level down."""
        items = self.indexed_items
        output = ["\nPT File Size Check:", "-" * 40]

        # Check current directory first
        pt_files = [
            (item.name, item.stats["size"])
            for item in items
            if not item.is_folder and item.name.endswith(".pt")
        ]

        if not pt_files:
            output.append("No .pt files in current directory")
            output.append("Checking subdirectories...")
            output.append("")
        else:
            # Check current directory files
            sizes = {size for _, size in pt_files}
            same_size = len(sizes) == 1

            if same_size:
                size = next(iter(sizes))
                output.append(
                    f"Current directory - All .pt files same size: {self.check_mark(True)} ({self.s3.format_size(size)})"
                )
            else:
                output.append(
                    f"Current directory - All .pt files same size: {self.check_mark(False)}"
                )
                output.append("Size groups:")
                size_groups = {}
                for name, size in pt_files:
                    if size not in size_groups:
                        size_groups[size] = []
                    size_groups[size].append(name)
                for size, files in sorted(size_groups.items()):
                    output.append(f"  {self.s3.format_size(size)}: {', '.join(files)}")
            output.append("")

        # Check subdirectories
        folders = [item for item in items if item.is_folder]
        if folders:
            for folder in folders:
                folder_prefix = self.get_current_prefix() + folder.name + "/"
                files, _ = self.s3.list_objects(folder_prefix)
                pt_files = [(f["name"], f["size"]) for f in files if f["name"].endswith(".pt")]

                if pt_files:
                    sizes = {size for _, size in pt_files}
                    same_size = len(sizes) == 1

                    if same_size:
                        size = next(iter(sizes))
                        output.append(
                            f"{folder.name} - All .pt files same size: {self.check_mark(True)} ({self.s3.format_size(size)})"
                        )
                    else:
                        output.append(
                            f"{folder.name} - All .pt files same size: {self.check_mark(False)}"
                        )
                        output.append("Size groups:")
                        size_groups = {}
                        for name, size in pt_files:
                            if size not in size_groups:
                                size_groups[size] = []
                            size_groups[size].append(name)
                        for size, files in sorted(size_groups.items()):
                            output.append(f"  {self.s3.format_size(size)}: {', '.join(files)}")
                else:
                    output.append(f"{folder.name} - No .pt files")
        else:
            output.append("No subdirectories to check")

        return "\n".join(output)

    def cmd_help(self, args: List[str]) -> str:
        """Show help message."""
        return """Available commands:
  ls              List current directory contents
  cd <idx>        Change directory (cd .. for parent, cd for root)
  cat <idx>       View file contents (> file.txt to save)
  rm <idx...>     Remove files/folders
  filecount       Compare file counts across folders
  sizecheck       Check .pt file sizes in current and child dirs
  inspect <idx>   View tensor shapes and decode text from .pt file
  help            Show this help
  exit            Exit shell"""

    def handle_redirect(self, output: str, args: List[str]) -> str:
        """Handle output redirection."""
        if len(args) >= 2 and args[-2] == ">":
            filename = args[-1]
            with open(filename, "w") as f:
                f.write(output)
            return f"Output saved to {filename}"
        return output

    def show_progress(self, progress: Progress):
        """Show a spinning progress indicator."""
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not progress.done:
            sys.stdout.write(
                f"\r{spinner[idx]} Deleted {progress.current}/{progress.total} objects..."
            )
            sys.stdout.flush()
            idx = (idx + 1) % len(spinner)
            import time

            time.sleep(0.1)
        sys.stdout.write(f"\rDeleted {progress.current}/{progress.total} objects.   \n")
        sys.stdout.flush()

    def get_item_by_index(self, idx: int) -> Optional[IndexedItem]:
        """Get item by its index in O(1) time."""
        return self._index_map.get(idx)

    def get_folder_by_name(self, name: str) -> Optional[IndexedItem]:
        """Get folder by exact name match in O(1) time."""
        return self._folder_map.get(name)

    def get_prompt(self) -> str:
        """Get the current prompt with path."""
        path = f"{self.BLUE}s3://{self.s3.bucket}"
        if self.current_path:
            path += "/" + "/".join(self.current_path)
        return f"{path}{self.RESET}> "

    def cmd_ls(self, args: List[str]) -> str:
        """Handle ls command."""
        items = self.list_items()
        return self.format_listing(items)

    def cmd_cd(self, args: List[str]) -> str:
        """Handle cd command."""
        if not args:
            self.current_path = []
            self.list_items()
            return "Changed to root directory"

        if args[0] == "..":
            if self.current_path:
                self.current_path.pop()
                self.list_items()
                return "Changed to parent directory"
            return "Already at root"

        # Try to convert to index first - fast path
        try:
            idx = int(args[0])
            item = self.get_item_by_index(idx)
            if not item:
                return f"No item with index {idx}"
            if not item.is_folder:
                return "Not a folder"
            self.current_path.append(item.name)
            self.list_items()
            return f"Changed to {item.name}"
        except ValueError:
            # Only try folder name lookup if index conversion fails
            item = self.get_folder_by_name(args[0])
            if not item:
                return f"No folder named '{args[0]}' found"
            self.current_path.append(item.name)
            self.list_items()
            return f"Changed to {item.name}"

    def cmd_cat(self, args: List[str]) -> str:
        """Handle cat command."""
        if not args:
            return "Usage: cat <index> [> filename]"

        try:
            idx = int(args[0])
            item = self.get_item_by_index(idx)
            if not item or item.is_folder:
                return "Not a file"

            key = self.get_current_prefix() + item.name
            content = self.s3.read_file(key)
            self.last_output = content
            return content

        except ValueError:
            return "Invalid index"
        except Exception as e:
            return f"Error: {str(e)}"

    def cmd_rm(self, args: List[str]) -> str:
        """Handle rm command - removes files or folders."""
        if not args:
            return "Usage: rm <index> [<index2> <index3> ...]"

        results = []
        total_deleted = 0
        total_objects = 0

        # Pre-validate all indices first to avoid partial deletions
        items_to_delete = []
        for arg in args:
            try:
                idx = int(arg)
                item = self.get_item_by_index(idx)
                if not item:
                    return f"Invalid index: {idx}"
                items_to_delete.append(item)
            except ValueError:
                return f"Invalid index: {arg}"

        # All indices are valid, proceed with deletion
        for item in items_to_delete:
            try:
                prefix = self.get_current_prefix() + item.name
                if item.is_folder:
                    prefix += "/"

                deleted_count, objects_count = self.s3.delete_objects(
                    prefix, progress_callback=self.show_progress
                )

                total_deleted += deleted_count
                total_objects += objects_count

                if objects_count == 0:
                    results.append(f"No objects found with prefix {prefix}")
                else:
                    what = "folder" if item.is_folder else "file"
                    results.append(f"Deleted {what} '{item.name}'")

            except Exception as e:
                results.append(f"Error deleting {item.name}: {str(e)}")

        # Add summary if multiple items were processed
        if len(args) > 1:
            results.append(f"\nSummary: Deleted {total_deleted} objects across {len(args)} items")

        # Refresh the directory listing after deletion
        self.list_items()

        return "\n".join(results)

    def cmd_inspect(self, args: List[str]) -> str:
        """Handle inspect command for .pt files."""
        if not args:
            return "Usage: inspect <index>"

        try:
            idx = int(args[0])
            item = self.get_item_by_index(idx)
            if not item or item.is_folder:
                return "Not a file"

            if not item.name.endswith(".pt"):
                return "Not a .pt file"

            # Download the file to a temporary location
            key = self.get_current_prefix() + item.name
            local_path = f"/tmp/{item.name}"
            self.s3.download_file(key, local_path)

            try:
                # Get the S3 path relative to bucket root
                s3_path = "/".join(self.current_path + [item.name])

                # Inspect the file
                shapes, model_name, decoded_texts, tensor_info = inspect_tensor_file(
                    self.s3.bucket, local_path, s3_path
                )

                # Format initial output
                output = ["\nPT File Inspection:", "-" * 40]
                output.append(f"Model: {model_name}")
                output.append("\nTensor Shapes:")
                for key, shape in shapes.items():
                    output.append(f"  {key}: {shape}")

                # Add tensor validity checks
                if "states" in tensor_info:
                    states_info = tensor_info["states"]
                    output.append("\nStates Tensor Check:")
                    output.append(f"  No NaNs: {self.check_mark(not states_info['has_nan'])}")
                    output.append(f"  No Infs: {self.check_mark(not states_info['has_inf'])}")
                    output.append(
                        f"  Value range: [{states_info['min_val']:.3f}, {states_info['max_val']:.3f}]"
                    )

                if decoded_texts:
                    output.append("\nFirst 4 batches (first 250 chars each):")
                    output.append("-" * 40)
                    for i, text in enumerate(decoded_texts[:4]):
                        # Clean up text: remove multiple newlines and leading/trailing whitespace
                        text = "\n".join(line for line in text.splitlines() if line.strip())
                        preview = text[:250] + "..." if len(text) > 250 else text
                        output.append(f"Batch {i}: {preview}")

                    # Print initial output
                    print("\n".join(output))

                    # Interactive mode for viewing full batches
                    num_batches = len(decoded_texts)
                    while True:
                        print(
                            f"\nEnter batch number (0-{num_batches-1}) to view full text, or 'q' to quit:"
                        )
                        try:
                            choice = input().strip()
                            if choice.lower() == "q":
                                return ""

                            batch_idx = int(choice)
                            if 0 <= batch_idx < num_batches:
                                text = "\n".join(
                                    line
                                    for line in decoded_texts[batch_idx].splitlines()
                                    if line.strip()
                                )
                                print(f"\nFull text of batch {batch_idx}:\n{'-'*40}\n{text}")

                                # Show states tensor preview for this batch
                                if "states" in tensor_info:
                                    states_preview = format_tensor_preview(
                                        tensor_info["states"]["tensor"], batch_idx
                                    )
                                    print(
                                        f"\nStates tensor preview for batch {batch_idx}:\n{'-'*40}\n{states_preview}"
                                    )
                            else:
                                print(
                                    f"Invalid batch number. Please enter a number between 0 and {num_batches-1}"
                                )
                        except ValueError:
                            print(
                                f"Invalid input. Please enter a number between 0 and {num_batches-1}"
                            )

                return ""
            finally:
                # Clean up
                if os.path.exists(local_path):
                    os.remove(local_path)

        except ValueError:
            return "Invalid index"
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        """Run the shell loop."""
        print("S3 Shell - Type 'help' for commands")
        print(self.cmd_ls([]))  # List items on startup

        while True:
            try:
                command = input(self.get_prompt()).strip()
                if not command:
                    continue

                # Add command to history
                self.history.add_command(command)

                parts = command.split()
                cmd, args = parts[0], parts[1:]

                if cmd == "exit":
                    self.history.save()  # Save history before exiting
                    break

                # Handle commands
                output = ""
                if cmd == "ls":
                    output = self.cmd_ls(args)
                elif cmd == "cd":
                    output = self.cmd_cd(args)
                elif cmd == "cat":
                    output = self.cmd_cat(args[0:1])  # Only pass the index
                    output = self.handle_redirect(output, args)
                elif cmd == "rm":
                    output = self.cmd_rm(args)
                elif cmd == "filecount":
                    output = self.cmd_filecount(args)
                elif cmd == "sizecheck":
                    output = self.cmd_sizecheck(args)
                elif cmd == "inspect":
                    output = self.cmd_inspect(args)
                elif cmd == "help":
                    output = self.cmd_help(args)
                else:
                    output = f"Unknown command: {cmd}"

                if output:
                    print(output)

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {str(e)}")

        # Final save of history
        self.history.save()
