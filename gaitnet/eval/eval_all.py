"""Run the full evaluation sweep.

This used to launch one subprocess per (difficulty, velocity) configuration and kill
each one off, because `eval_gaitnet` could only evaluate a single configuration and
would not exit on its own. It now sweeps the whole grid in one process, so all that
is left here is the entry point: everything below forwards to `eval_gaitnet`, whose
defaults are the full sweep.

Pass through any `eval_gaitnet` argument to narrow the sweep, e.g.

    python -m gaitnet.eval.eval_all --difficulties 0.2 0.4 --velocities 0.1
"""

import os
import subprocess
import sys

from gaitnet import PROJECT_ROOT, get_logger, setup_logging

logger = get_logger()


def main():
    setup_logging()
    os.chdir(PROJECT_ROOT)
    subprocess_args = [
        sys.executable,
        "-m",
        "gaitnet.eval.eval_gaitnet",
        "--headless",
        "--no-log-file",
        *sys.argv[1:],
    ]
    logger.debug(f"Running command: {' '.join(subprocess_args)}")
    raise SystemExit(subprocess.run(subprocess_args).returncode)


if __name__ == "__main__":
    main()
