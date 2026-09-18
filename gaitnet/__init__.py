"""GaitNet.

Importing this package has no side effects. Entry points call `setup_logging()` to
attach handlers and open the run's log file.
"""

from __future__ import annotations

import inspect
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_git_commit: str | None = None


def __getattr__(name: str):
    # GIT_COMMIT is resolved on first use, so importing the package never shells out
    if name == "GIT_COMMIT":
        global _git_commit
        if _git_commit is None:
            try:
                _git_commit = subprocess.check_output(
                    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode("ascii").strip()
            except Exception:
                _git_commit = "unknown"
        return _git_commit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ProjectRelativeFormatter(logging.Formatter):
    """Custom formatter that shows file path relative to project root."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            path = Path(record.pathname).resolve()
            record.relpath = path.relative_to(PROJECT_ROOT)
        except Exception:
            record.relpath = record.pathname  # fallback to absolute
        return super().format(record)


class AlignedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, path_column=100):
        super().__init__(fmt, datefmt)
        self.path_column = path_column

    def format(self, record):
        original_message = super().format(record)
        lines = original_message.splitlines()

        # pad only the first line to align the path info
        path_info = f"[{record.pathname}:{record.lineno}]"
        line = lines[0]
        padding = max(self.path_column - len(line), 1)
        lines[0] = f"{line}{' ' * padding}{path_info}"

        return "\n".join(lines)


def setup_logging(log_to_file: bool | None = None, max_logs: int = 10) -> Path | None:
    """Attach console and file handlers to the "gaitnet" logger.

    Call this from entry points, before anything else logs. Calling it again is a no-op.

    Args:
        log_to_file: Write a timestamped log under `PROJECT_ROOT/logs`. If None, a
            `--no-log-file` flag in sys.argv turns it off (and is removed from argv
            so the entry point's own argument parser doesn't see it).
        max_logs: Oldest log files beyond this count are deleted.

    Returns:
        The log file path, or None when not logging to a file.
    """
    logger = logging.getLogger("gaitnet")
    if logger.handlers:
        return None

    if log_to_file is None:
        log_to_file = "--no-log-file" not in sys.argv
        if not log_to_file:
            sys.argv.remove("--no-log-file")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        ProjectRelativeFormatter(
            "%(asctime)s | %(relpath)s:%(lineno)d | %(levelname)s |  %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console)

    log_file = None
    if log_to_file:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)

        # delete old log files, leaving room for this one
        log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)
        while len(log_files) >= max_logs:
            oldest = log_files.pop(0)
            try:
                oldest.unlink()
            except Exception as e:
                logger.debug(f"failed to delete {oldest}: {e}")

        timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
        log_file = log_dir / f"{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            AlignedFormatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                path_column=120,  # column at which [pathname:lineno] should start
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"log file: {log_file}")

    try:
        logger.debug(f"running '{' '.join(sys.orig_argv)}'")
    except AttributeError:
        pass

    return log_file


def get_logger():
    """Get a logger with name relative to the gaitnet directory."""
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return logging.getLogger("gaitnet.unknown")
    frame = frame.f_back
    filename = frame.f_code.co_filename
    try:
        rel_path = Path(filename).relative_to(PROJECT_ROOT)
    except ValueError:
        rel_path = Path(filename)
    module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
    return logging.getLogger("gaitnet." + module_name)
