import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import inspect

# Define project root (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    GIT_COMMIT = subprocess.check_output(
        ['git', '-C', str(PROJECT_ROOT), 'rev-parse', '--short', 'HEAD']
    ).decode('ascii').strip()
except Exception:
    GIT_COMMIT = "unknown"


class ProjectRelativeFormatter(logging.Formatter):
    """Custom formatter that shows file path relative to project root."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            path = Path(record.pathname).resolve()
            record.relpath = path.relative_to(PROJECT_ROOT)
        except Exception:
            # logger.debug(f"failed to make path relative: {record.pathname}")
            record.relpath = record.pathname  # fallback to absolute
        return super().format(record)


class AlignedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, path_column=100):
        super().__init__(fmt, datefmt)
        self.path_column = path_column

    def format(self, record):
        # Format the original message
        original_message = super().format(record)

        # Split into lines
        lines = original_message.splitlines()

        # Path info to append
        path_info = f"[{record.pathname}:{record.lineno}]"

        # Pad only the last line to align path_info
        line = lines[0]
        padding = max(self.path_column - len(line), 1)
        lines[0] = f"{line}{' ' * padding}{path_info}"

        # Rejoin all lines
        return "\n".join(lines)


# Check for --no-log-file flag
no_log_file = "--no-log-file" in sys.argv
if no_log_file:
    sys.argv.remove("--no-log-file")

# Timestamped log file
timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
"""Holds the timestamp of when the program started."""

# Ensure logs directory exists
log_dir = PROJECT_ROOT / "logs"

if not no_log_file:
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{timestamp}.log"
else:
    log_file = None

# Root logger
logger = logging.getLogger("src")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(
    ProjectRelativeFormatter(
        "%(asctime)s | %(relpath)s:%(lineno)d | %(levelname)s |  %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)

# File handler (only if logging to file is enabled)
fh = None
if not no_log_file and log_file is not None:
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        AlignedFormatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            path_column=120,  # column at which [pathname:lineno] should start
        )
    )

# --- Add handlers ---
if not logger.handlers:
    logger.addHandler(ch)
    if fh is not None:
        logger.addHandler(fh)
    logger.propagate = False

if not no_log_file:
    logger.info(f"log file: {log_file}")

launch_str = " ".join(sys.orig_argv)
logger.debug(f"running '{launch_str}'")


def get_logger():
    """Get a logger with name relative to the src directory."""
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return logging.getLogger("src.unknown")
    frame = frame.f_back
    filename = frame.f_code.co_filename
    rel_path = Path(filename).relative_to(PROJECT_ROOT / "src")
    module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
    return logging.getLogger("src." + module_name)


# Delete old log files (only if logging to file)
if not no_log_file:
    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)
    max_logs = 10
    while len(log_files) >= max_logs:
        oldest = log_files.pop(0)
        try:
            oldest.unlink()
            logger.debug(f"deleted old log file: {oldest}")
        except Exception as e:
            logger.debug(f"failed to delete {oldest}: {e}")
        del oldest
    del log_files, max_logs

del (
    ch,
    fh,
    launch_str,
    no_log_file,
    ProjectRelativeFormatter,
    AlignedFormatter,
    datetime,
    # Path,
    sys,
    # logging,
    # inspect,
)

logger.debug("initialized")
