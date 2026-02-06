from __future__ import annotations
import inspect
import traceback
from multiprocessing import Pipe, Process, cpu_count
from multiprocessing.connection import Connection
from typing import Any, Callable, Generic, Type, TypeVar

import numpy as np

import enum

from gaitnet import get_logger
logger = get_logger()

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nptyping import NDArray, Shape, Bool, Number


class Message:
    class _ManagerMessages(enum.Enum):
        PING = enum.auto()
        CALL_FUNCTION = enum.auto()
        SHUTDOWN = enum.auto()

    class _WorkerMessages(enum.Enum):
        PONG = enum.auto()
        SUCCESS = enum.auto()
        EXCEPTION = enum.auto()

    Manager = _ManagerMessages
    Worker = _WorkerMessages


T = TypeVar("T")


class VectorPool(Generic[T]):
    """Provides a vectorized wrapper around a objects using multiprocessing.

    useful over a multiprocessing.Pool when the objects cannot be serialized

    classes are distributed across worker processes to handle CPU-bound
    computations in parallel while maintaining state persistence across calls.
    """

    def __init__(
        self,
        instances: int,
        cls: Type[T],
        num_workers: None | int = None,
        **kwargs,
    ) -> None:
        """

        Args:
            instances: Number of object instances to create
            cls: The Class to instantiate
            num_workers: Number of worker processes (default: cpu_count())
                automatically set to the number of instances if that is lower
                than the desired number of workers
            **kwargs: Additional arguments passed to Class constructor
        """
        self.instances = instances
        self.num_workers = min(num_workers or cpu_count(), instances)

        self.workers: list[Process] = []
        self.pipes: list[Connection] = []

        # Setup workers
        self._setup_workers(cls, **kwargs)

    def _batch_data(self, data: np.ndarray) -> list[np.ndarray]:
        """Batches data to be sent to the workers"""
        return np.array_split(data, self.num_workers)

    def _setup_workers(self, cls: Type[T], **kwargs) -> None:
        """Setup worker processes and verify they initialized successfully."""
        object_assignments = self._batch_data(np.arange(self.instances))
        # figure out how many objects each worker has
        worker_objects = [assignment.shape[0] for assignment in object_assignments]

        for worker_id, num_objects in zip(range(self.num_workers), worker_objects):
            parent_conn, child_conn = Pipe()

            worker = Process(
                target=self._worker_loop,
                args=(child_conn, worker_id, cls, num_objects, kwargs),
            )
            worker.start()

            self.workers.append(worker)
            self.pipes.append(parent_conn)

        # Verify all workers started successfully
        for pipe in self.pipes:
            try:
                pipe.send((Message.Manager.PING, None))
                response_type, result = pipe.recv()
                if response_type != Message.Worker.PONG:
                    raise RuntimeError(f"Worker failed to initialize: {result}")
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to verify worker readiness: {e}")

    def _verify_workers_ready(self) -> None:
        """Verify all worker processes initialized successfully."""
        for pipe in self.pipes:
            try:
                # Send initialization check
                pipe.send((Message.Manager.PING, None))
                response_type, result = pipe.recv()
                if response_type != Message.Worker.PONG:
                    raise RuntimeError(f"Worker failed to initialize: {result}")
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to verify worker readiness: {e}")

    @staticmethod
    def _worker_loop(
        conn: Connection,
        worker_id: int,
        cls: Type[T],
        num_objects: int,
        kwargs: dict,
    ) -> None:
        """Main loop for worker process. Creates and maintains multiple Class instances."""
        try:
            # Initialize all objects for this worker
            objects = [cls(**kwargs) for _ in range(num_objects)]

            logger.debug(f"started worker id {worker_id} with {num_objects} objects")

            # Main processing loop
            while True:
                try:
                    command, data = conn.recv()

                    if command == Message.Manager.PING:
                        conn.send((Message.Worker.PONG, None))

                    elif command == Message.Manager.CALL_FUNCTION:
                        function_name, mask, batch_args = data

                        # Process each object in this worker's batch
                        results_list = []
                        for i in range(len(objects)):
                            if not mask[i]:
                                # nan here still allows the output array to be typed
                                results_list.append(np.nan)
                                continue
                            function = getattr(objects[i], function_name)
                            args = [batch_arg[i] for batch_arg in batch_args]
                            result = function(*args)
                            results_list.append(result)

                        # Stack results for this worker's batch
                        batch_results = np.stack(results_list, axis=0)
                        conn.send((Message.Worker.SUCCESS, batch_results))

                    elif command == Message.Manager.SHUTDOWN:
                        conn.send((Message.Worker.SUCCESS, None))
                        break

                    else:
                        conn.send(
                            (Message.Worker.EXCEPTION, f"Unknown command: {command}")
                        )

                except Exception as e:
                    # Send error back to main process
                    error_msg = (
                        f"Worker {worker_id} error: {str(e)}\n{traceback.format_exc()}"
                    )
                    logger.error(error_msg)
                    conn.send((Message.Worker.EXCEPTION, error_msg))

        except Exception as e:
            # Initialization failed
            error_msg = f"Worker {worker_id} initialization failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            conn.send((Message.Worker.EXCEPTION, error_msg))
        finally:
            conn.close()

    @staticmethod
    def _expect_success(pipe: Connection) -> Any:
        """Helper method that returns the value when a pipe returns sucess
        throws a runtime error otherwise

        Args:
            pipe (Connection): The pipe to communicate with

        Raises:
            RuntimeError: Whever something goes wrong

        Returns:
            Any: The result result of the pipe's communciation
        """
        try:
            response_type, result = pipe.recv()
            if response_type == Message.Worker.SUCCESS:
                return result
            elif response_type == Message.Worker.EXCEPTION:
                raise RuntimeError(f"Worker exception: {result}")
            else:
                raise RuntimeError(f"Worker sent unexpected response: {response_type}")
        except Exception as e:
            logger.error(f"Failed to receive result from worker: {e}")
            raise RuntimeError(f"Failed to receive result from worker: {e}")

    def call(
        self,
        function: Callable,
        mask: None | NDArray[Shape["*"], Bool],
        **kwargs: NDArray[Shape["*, ..."], Number],
    ) -> NDArray[Shape["*, ..."], Any]:
        """Calls a function on all of the underlying objects

        Note: kwargs get masked, so it should have the same number of rows
        as the number of instances in the pool, but the values of masked rows
        do not matter.

        Args:
            function (Callable): function to call (should be a handle to a function from T)
            mask (None | NDArray[Shape["*"], Bool]): mask to apply to the function inputs
                masked values result in nan outputs
            kwargs (np.ndarray): arguments to pass to function. same type but with dimensionality
                one higher than the function's input

        Returns:
            np.ndarray: a numpy array of results with a dimensionality one higher than the 
                function return type
        """
        # validate kwargs against the function signature
        sig = inspect.signature(function)
        try:
            sig.bind_partial(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Invalid arguments for {function.__name__}: {e}"
            )

        # validate kwarg shapes
        for kw, arg in kwargs.items():
            assert (
                arg.shape[0] == self.instances
            ), f"Expected {self.instances} rows for {kw}, got {arg.shape[0]}"
        if mask is not None:
            assert (
                mask.shape[0] == self.instances
            ), f"Expected {self.instances} rows for mask, got {mask.shape[0]}"
        else:
            mask = np.full((self.instances,), True, dtype=bool)

        function_name = function.__name__
        all_batched_mask = self._batch_data(mask)
        all_batched_args = [self._batch_data(arg) for arg in kwargs.values()]
        if all_batched_args:
            batched_args_iter = zip(*all_batched_args)
        else:
            # need this edge case to keep the outer zip happy
            # here we just pass an empty tuple which eventually gets
            # splatted (*...) into nothingness by the workers
            batched_args_iter = [() for _ in range(len(self.pipes))]  # type: ignore

        # send function calls
        for pipe, batched_mask, batched_args in zip(
            self.pipes, all_batched_mask, batched_args_iter
        ):
            try:
                pipe_args = (function_name, batched_mask, batched_args)
                pipe.send((Message.Manager.CALL_FUNCTION, pipe_args))
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to send data to worker: {e}")

        # gather results
        results = []
        for pipe in self.pipes:
            batched_torques = VectorPool._expect_success(pipe)
            results.append(batched_torques)

        return np.concatenate(results)

    def _cleanup(self) -> None:
        """Clean up worker processes and pipes."""
        # Send shutdown signal to all responsive workers
        for i, pipe in enumerate(self.pipes, start=1):
            try:
                if pipe is not None:
                    pipe.send((Message.Manager.SHUTDOWN, None))
                    pipe.recv()  # Wait for acknowledgment
                    pipe.close()
                    logger.debug(f"cleaned up pipe {i}/{len(self.pipes)}")
            except:
                logger.info(f"pipe {i} unresponsive during cleanup")
                pass  # Worker may already be dead

        # Terminate any remaining worker processes
        for i, worker in enumerate(self.workers, start=1):
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.1)
                if worker.is_alive():
                    worker.kill()  # Force kill if terminate didn't work
                    logger.info(f"force killed worker {i}/{len(self.workers)}")
                else:
                    logger.debug(f"terminated worker {i}/{len(self.workers)}")

        self.workers.clear()
        self.pipes.clear()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
