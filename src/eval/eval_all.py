import os
import sys
from src import PROJECT_ROOT, get_logger
import subprocess
import itertools
import signal
import time
import psutil
import threading
import queue
import argparse
parser = argparse.ArgumentParser(description="Run all evaluations")
parser.add_argument(
    "--resume-from",
    type=int,
    default=0,
    help="Resume from iteration number (0 = start from beginning)"
)
args, unused_args = parser.parse_known_args()

logger = get_logger()

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                logger.debug(f"Killing child process {child.pid}")
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        try:
            logger.debug(f"Killing parent process {pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        
        # Wait for processes to die
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # If any are still alive, try SIGKILL again
        for p in alive:
            try:
                logger.warning(f"Process {p.pid} still alive, sending SIGKILL again")
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        logger.debug(f"Process {pid} already dead")
    except Exception as e:
        logger.error(f"Error killing process tree: {e}")

def read_output(pipe, output_queue):
    """Read from pipe and put lines into queue."""
    try:
        for line in iter(pipe.readline, ''):
            output_queue.put(line)
    finally:
        pipe.close()

def run_with_timeout(cmd, timeout_after_completion=5):
    """
    Run a subprocess and force kill it after a timeout period following completion signal.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True
    )
    
    # Create a queue for output and a thread to read it
    output_queue = queue.Queue()
    reader_thread = threading.Thread(target=read_output, args=(process.stdout, output_queue))
    reader_thread.daemon = True
    reader_thread.start()
    
    completion_signaled = False
    completion_time = None
    
    try:
        while True:
            # Check if process has terminated
            if process.poll() is not None:
                # Drain remaining output
                while not output_queue.empty():
                    try:
                        line = output_queue.get_nowait()
                        print(line, end='')
                    except queue.Empty:
                        break
                
                # If we saw completion signal, give it a moment then force kill if needed
                if completion_signaled:
                    time.sleep(timeout_after_completion)
                    if psutil.pid_exists(process.pid):
                        logger.warning(f"Process still exists after completion, force killing")
                        kill_process_tree(process.pid)
                break
            
            # Try to get output (non-blocking)
            try:
                line = output_queue.get(timeout=0.1)
                print(line, end='')
                
                # Look for completion signal
                if "Evaluation complete." in line:
                    if not completion_signaled:
                        completion_signaled = True
                        completion_time = time.time()
                        logger.info(f"Completion signal detected, will force kill in {timeout_after_completion}s if needed")
            except queue.Empty:
                pass
            
            # If we've signaled completion and timeout has passed, force kill
            if completion_signaled and completion_time:
                elapsed = time.time() - completion_time
                if elapsed > timeout_after_completion:
                    logger.warning(f"Process hung after completion ({elapsed:.1f}s elapsed), force killing process tree (PID: {process.pid})")
                    kill_process_tree(process.pid)
                    process.wait(timeout=2)
                    return
    
    except KeyboardInterrupt:
        logger.info("Interrupted, killing subprocess tree")
        kill_process_tree(process.pid)
        raise
    
    finally:
        # Ensure process is dead
        try:
            if process.poll() is None:
                logger.warning("Process still running in finally block, force killing")
                kill_process_tree(process.pid)
                process.wait(timeout=2)
        except:
            pass

def main():
    
    os.chdir(PROJECT_ROOT)
    eval_scripts = ["src.eval.eval_gaitnet"]
    difficulties = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    speeds = [0.05, 0.1, 0.15, 0.2]
    
    # Calculate total iterations and create enumeration
    total_iterations = len(difficulties) * len(speeds) * len(eval_scripts)
    logger.info(f"Total iterations: {total_iterations}")
    
    if args.resume_from > 0:
        logger.info(f"Resuming from iteration {args.resume_from}")
    
    for iteration, (difficulty, speed, script) in enumerate(itertools.product(
        difficulties, speeds, eval_scripts
    )):
        # Skip iterations before resume point
        if iteration < args.resume_from:
            continue
        
        logger.info(f"[Iteration {iteration}/{total_iterations-1}] Running {script} with difficulty {difficulty} and speed {speed}")
        subprocess_args = [
            sys.executable,
            "-m",
            script,
            "--difficulty",
            str(difficulty),
            "--velocity",
            str(speed),
            "--headless",
            "--trials",
            "1",
            "--num_envs",
            "50",
            "--no-log-file",
        ]
        logger.debug(f"Running command: {' '.join(subprocess_args)}")
        
        run_with_timeout(subprocess_args, timeout_after_completion=5)
        logger.info(f"Completed iteration {iteration}\n")
        
        # Extra safety: wait a moment between runs
        time.sleep(1)

if __name__ == "__main__":
    main()