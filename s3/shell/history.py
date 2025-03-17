import readline
import os
from typing import Optional


class CommandHistory:
    def __init__(self, histfile: Optional[str] = None):
        """Initialize command history with optional history file."""
        self.histfile = histfile

        # Configure readline
        if histfile:
            # Create history file directory if it doesn't exist
            os.makedirs(os.path.dirname(histfile), exist_ok=True)

            # Set history file
            readline.set_history_length(1000)
            try:
                readline.read_history_file(histfile)
            except FileNotFoundError:
                pass

        # Enable tab completion and better key bindings
        readline.parse_and_bind("tab: complete")

        # Set better key bindings for history navigation
        readline.parse_and_bind('"\e[A": previous-history')  # Up arrow
        readline.parse_and_bind('"\e[B": next-history')  # Down arrow

    def add_command(self, cmd: str):
        """Add a command to history."""
        if cmd.strip():  # Only add non-empty commands
            readline.add_history(cmd)

    def save(self):
        """Save history to file if histfile was specified."""
        if self.histfile:
            try:
                readline.write_history_file(self.histfile)
            except Exception as e:
                print(f"Error saving history: {str(e)}")
