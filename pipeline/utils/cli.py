"""Copyright (2025) Tilde Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

This module provides the main entry point for the activault CLI and handles subcommands.
"""

import sys
import os
import argparse
import logging
import random
from pathlib import Path

# Add the project root to sys.path so we can import stash.py
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TILDE_LOGO = """                           
                         ___     
                        / _ \_/\ 
                        \/ \___/                             
"""

ACTIVAULT_BANNER = """
╔════════════════════════════════════════════════════════╗
║                   ACTIVAULT v0.1.0                     ║
║                                                        ║
║   A pipeline for collecting LLM activations and        ║
║      storing them for efficient retrieval              ║
╚════════════════════════════════════════════════════════╝
"""


def show_welcome():
    """Display the welcome message with ASCII art."""
    os.system("cls" if os.name == "nt" else "clear")
    print("\033[94m" + TILDE_LOGO + "\033[0m")
    print("\033[92m" + ACTIVAULT_BANNER + "\033[0m")
    print("\033[1mAvailable Commands:\033[0m")
    print("  \033[96mactivault collect\033[0m - Run activation collection")
    print("  \033[96mactivault s3\033[0m      - Launch the S3 shell")
    print("\nFor more information on a command, run: activault <command> --help")
    print("\nVisit: \033[4mhttps://github.com/tilde-research/activault\033[0m for documentation")
    print()


def s3_command():
    """Launch the S3 shell."""
    from s3.shell.shell import S3Shell

    shell = S3Shell()
    shell.run()


def collect_command(args):
    """Run the main activation collection pipeline."""
    try:
        import stash

        # Pass command line arguments through to stash.py
        sys.argv = ["stash.py"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.machine is not None:
            sys.argv.extend(["--machine", str(args.machine)])
        stash.main()
    except ImportError as e:
        logger.error(f"Failed to import stash module: {e}")
        logger.error("Make sure you're running from the project root directory.")
        sys.exit(1)


def main():
    """Main entry point for the activault CLI."""
    parser = argparse.ArgumentParser(
        description="Activault - A tool for collecting and processing neural network activations."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add the 's3' subcommand
    s3_parser = subparsers.add_parser("s3", help="Launch the S3 shell")

    # Add the 'collect' subcommand
    collect_parser = subparsers.add_parser("collect", help="Run activation collection")
    collect_parser.add_argument(
        "--config", type=str, help="Path to configuration file", required=False
    )
    collect_parser.add_argument(
        "--machine", type=int, help="Machine index for distributed processing", required=False
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle subcommands
    if args.command == "s3":
        s3_command()
    elif args.command == "collect":
        collect_command(args)
    elif args.command is None:
        show_welcome()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
