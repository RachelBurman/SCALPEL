"""
Shared Rich console for SCALPEL.

Configured to handle Windows terminal encoding issues.
"""

import sys
from rich.console import Console

# Force UTF-8 on Windows to avoid encoding errors with special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True)
