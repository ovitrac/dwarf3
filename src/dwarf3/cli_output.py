"""
Colored CLI output utilities for dwarf3.

Provides styled terminal output with colors, progress bars, and status indicators.

Author: Olivier Vitrac, PhD, HDR
        Generative Simulation Initiative
        olivier.vitrac@gmail.com
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from colorama import Fore, Style, init as colorama_init
from tqdm import tqdm

# Initialize colorama for cross-platform support
colorama_init(autoreset=True)


# Color scheme for dwarf3 CLI
class Colors:
    """Color constants for consistent styling."""

    # Stage and headers
    HEADER = Fore.CYAN + Style.BRIGHT
    STAGE = Fore.BLUE + Style.BRIGHT
    SUBSTAGE = Fore.BLUE

    # Status indicators
    SUCCESS = Fore.GREEN + Style.BRIGHT
    WARNING = Fore.YELLOW
    ERROR = Fore.RED + Style.BRIGHT
    INFO = Fore.WHITE

    # Values and metrics
    VALUE = Fore.YELLOW + Style.BRIGHT
    METRIC = Fore.MAGENTA
    PATH = Fore.CYAN

    # Progress
    PROGRESS = Fore.GREEN

    # Dim for inactive/disabled items
    DIM = Style.DIM

    # Reset
    RESET = Style.RESET_ALL


# Unicode symbols (with ASCII fallbacks)
class Symbols:
    """Unicode symbols for status indicators."""

    CHECK = "\u2714"  # ✔
    CROSS = "\u2718"  # ✘
    ARROW = "\u2192"  # →
    BULLET = "\u2022"  # •
    STAR = "\u2605"   # ★
    TELESCOPE = "\U0001F52D"  # 🔭
    GALAXY = "\U0001F30C"     # 🌌
    ROCKET = "\U0001F680"     # 🚀
    SPARKLE = "\u2728"        # ✨
    TIMER = "\u23F1"          # ⏱
    FOLDER = "\U0001F4C1"     # 📁
    FILE = "\U0001F4C4"       # 📄
    CHART = "\U0001F4CA"      # 📊
    WRENCH = "\U0001F527"     # 🔧

    @classmethod
    def use_ascii(cls):
        """Switch to ASCII-only fallbacks."""
        cls.CHECK = "[OK]"
        cls.CROSS = "[X]"
        cls.ARROW = "->"
        cls.BULLET = "*"
        cls.STAR = "*"
        cls.TELESCOPE = "[T]"
        cls.GALAXY = "[G]"
        cls.ROCKET = "[>]"
        cls.SPARKLE = "*"
        cls.TIMER = "[T]"
        cls.FOLDER = "[D]"
        cls.FILE = "[F]"
        cls.CHART = "[C]"
        cls.WRENCH = "[W]"


def print_banner(version: str) -> None:
    """Print the dwarf3 startup banner."""
    banner = f"""
{Colors.HEADER}╔══════════════════════════════════════════════════════════════╗
║  {Symbols.TELESCOPE}  DWARF3 Astrophotography Pipeline  {Symbols.GALAXY}                    ║
║     Reproducible stacking for DWARF 3 smart telescope        ║
║     Version: {version:<20}                            ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def print_header(text: str, width: int = 60) -> None:
    """Print a styled section header."""
    line = "═" * width
    print(f"\n{Colors.HEADER}{line}")
    print(f"  {text}")
    print(f"{line}{Colors.RESET}")


def print_stage(stage_num: int, text: str, emoji: str = "") -> None:
    """Print a pipeline stage header."""
    if emoji:
        print(f"\n{Colors.STAGE}[Stage {stage_num}] {emoji}  {text}{Colors.RESET}")
    else:
        print(f"\n{Colors.STAGE}[Stage {stage_num}] {text}{Colors.RESET}")


def print_substage(text: str) -> None:
    """Print a substage or detail line."""
    print(f"  {Colors.SUBSTAGE}{Symbols.ARROW} {text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.SUCCESS}{Symbols.CHECK} {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}! {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.ERROR}{Symbols.CROSS} {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.INFO}{Symbols.BULLET} {text}{Colors.RESET}")


def print_metric(name: str, value: str | int | float, unit: str = "") -> None:
    """Print a metric with value."""
    if unit:
        print(f"  {Colors.METRIC}{name}: {Colors.VALUE}{value}{Colors.RESET} {unit}")
    else:
        print(f"  {Colors.METRIC}{name}: {Colors.VALUE}{value}{Colors.RESET}")


def print_path(label: str, path: str) -> None:
    """Print a file path."""
    print(f"  {Colors.INFO}{label}: {Colors.PATH}{path}{Colors.RESET}")


def print_summary_box(lines: list[str], title: str = "Summary") -> None:
    """Print a summary box with multiple lines."""
    width = max(len(line) for line in lines) + 4
    width = max(width, len(title) + 4)

    border_top = "╔" + "═" * width + "╗"
    border_mid = "╟" + "─" * width + "╢"
    border_bot = "╚" + "═" * width + "╝"

    print(f"\n{Colors.SUCCESS}{border_top}")
    print(f"║ {title:^{width-2}} ║")
    print(border_mid)
    for line in lines:
        print(f"║  {line:<{width-3}}║")
    print(f"{border_bot}{Colors.RESET}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_bytes(n_bytes: int) -> str:
    """Format byte count in human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


@dataclass
class ProgressConfig:
    """Configuration for progress bars."""

    bar_format: str = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ncols: int = 80
    colour: str = "green"
    leave: bool = True


def create_progress_bar(
    total: int,
    desc: str,
    unit: str = "frame",
    config: ProgressConfig | None = None,
    disable: bool = False,
) -> tqdm:
    """
    Create a styled progress bar.

    Parameters
    ----------
    total : int
        Total number of items.
    desc : str
        Description text.
    unit : str, default "frame"
        Unit name for items.
    config : ProgressConfig, optional
        Progress bar configuration.
    disable : bool, default False
        Disable the progress bar.

    Returns
    -------
    tqdm
        Configured progress bar.
    """
    if config is None:
        config = ProgressConfig()

    return tqdm(
        total=total,
        desc=f"{Colors.PROGRESS}{desc}{Colors.RESET}",
        unit=unit,
        bar_format=config.bar_format,
        ncols=config.ncols,
        colour=config.colour,
        leave=config.leave,
        disable=disable,
    )


class PipelineProgress:
    """
    Track and display progress for multi-stage pipeline.

    Example
    -------
    >>> progress = PipelineProgress(total_stages=7)
    >>> progress.start_stage(1, "Discovery", emoji="🔍")
    >>> progress.update_detail("Found 510 frames")
    >>> progress.complete_stage()
    """

    def __init__(self, total_stages: int = 7, quiet: bool = False):
        self.total_stages = total_stages
        self.current_stage = 0
        self.quiet = quiet
        self._stage_start_time = None

    def start_stage(self, stage_num: int, name: str, emoji: str = "") -> None:
        """Start a new stage."""
        import time

        if self.quiet:
            return

        self.current_stage = stage_num
        self._stage_start_time = time.time()

        stage_text = f"Stage {stage_num}/{self.total_stages}: {name}"
        if emoji:
            print(f"\n{Colors.STAGE}{emoji}  {stage_text}{Colors.RESET}")
        else:
            print(f"\n{Colors.STAGE}▶ {stage_text}{Colors.RESET}")

    def update_detail(self, text: str) -> None:
        """Update with a detail message."""
        if self.quiet:
            return
        print(f"   {Colors.INFO}{text}{Colors.RESET}")

    def complete_stage(self, message: str = "") -> None:
        """Mark current stage as complete."""
        import time

        if self.quiet:
            return

        if self._stage_start_time:
            elapsed = time.time() - self._stage_start_time
            elapsed_str = format_duration(elapsed)
            if message:
                print(f"   {Colors.SUCCESS}{Symbols.CHECK} {message} ({elapsed_str}){Colors.RESET}")
            else:
                print(f"   {Colors.SUCCESS}{Symbols.CHECK} Complete ({elapsed_str}){Colors.RESET}")
        else:
            if message:
                print(f"   {Colors.SUCCESS}{Symbols.CHECK} {message}{Colors.RESET}")
            else:
                print(f"   {Colors.SUCCESS}{Symbols.CHECK} Complete{Colors.RESET}")

    def fail_stage(self, message: str) -> None:
        """Mark current stage as failed."""
        if self.quiet:
            return
        print(f"   {Colors.ERROR}{Symbols.CROSS} {message}{Colors.RESET}")

    def warn(self, message: str) -> None:
        """Show a warning message."""
        if self.quiet:
            return
        print(f"   {Colors.WARNING}⚠ {message}{Colors.RESET}")


def detect_terminal_capabilities() -> dict:
    """
    Detect terminal capabilities for optimal display.

    Returns
    -------
    dict
        Capabilities dict with 'unicode', 'color', 'width' keys.
    """
    import os
    import shutil

    caps = {
        "unicode": True,
        "color": True,
        "width": 80,
    }

    # Check for color support
    if not sys.stdout.isatty():
        caps["color"] = False
    elif os.environ.get("NO_COLOR"):
        caps["color"] = False
    elif os.environ.get("TERM") == "dumb":
        caps["color"] = False
        caps["unicode"] = False

    # Check terminal width
    try:
        caps["width"] = shutil.get_terminal_size().columns
    except Exception:
        pass

    # Check for unicode support (simple heuristic)
    if os.environ.get("LANG", "").lower().find("utf") == -1:
        # Try to detect Windows console without UTF-8
        if sys.platform == "win32":
            import ctypes
            try:
                # Check if console supports UTF-8
                kernel32 = ctypes.windll.kernel32
                if kernel32.GetConsoleOutputCP() != 65001:
                    caps["unicode"] = False
            except Exception:
                caps["unicode"] = False

    return caps


def setup_terminal() -> dict:
    """
    Setup terminal for optimal display.

    Returns
    -------
    dict
        Terminal capabilities that were configured.
    """
    caps = detect_terminal_capabilities()

    if not caps["unicode"]:
        Symbols.use_ascii()

    return caps
