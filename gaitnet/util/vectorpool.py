from __future__ import annotations
import inspect
import traceback
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process, cpu_count, resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
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
        OPEN_CHANNEL = enum.auto()
        SET_CHANNEL_OUTPUT = enum.auto()
        CALL_CHANNEL = enum.auto()

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

    @classmethod
    def _worker_loop(
        pool_cls,
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
            # per-worker storage that persists between messages
            state: dict[str, Any] = {}

            logger.debug(f"started worker id {worker_id} with {num_objects} objects")

            # Main processing loop
            while True:
                try:
                    command, data = conn.recv()

                    if command == Message.Manager.SHUTDOWN:
                        conn.send((Message.Worker.SUCCESS, None))
                        break

                    conn.send(pool_cls._handle_message(objects, state, command, data))

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
    def _handle_message(
        objects: list, state: dict[str, Any], command: Message.Manager, data: Any
    ) -> tuple[Message.Worker, Any]:
        """Handle one message from the manager inside a worker process.

        Args:
            objects: This worker's object instances
            state: Per-worker storage that persists between messages
            command: The message type
            data: The message payload

        Returns:
            The (response type, result) to send back to the manager
        """
        if command == Message.Manager.PING:
            return (Message.Worker.PONG, None)

        if command == Message.Manager.CALL_FUNCTION:
            function_name, mask, batch_args = data
            results_list = VectorPool._call_objects(
                objects, function_name, mask, batch_args
            )
            # Stack results for this worker's batch
            return (Message.Worker.SUCCESS, np.stack(results_list, axis=0))

        return (Message.Worker.EXCEPTION, f"Unknown command: {command}")

    @staticmethod
    def _call_objects(
        objects: list, function_name: str, mask: np.ndarray, batch_args: list
    ) -> list:
        """Call a function on each of a worker's objects.

        Args:
            objects: This worker's object instances
            function_name: Name of the method to call
            mask: (num_objects,) which objects to call
            batch_args: Arguments with one row per object

        Returns:
            One result per object, nan for masked objects
        """
        results_list = []
        for i in range(len(objects)):
            if not mask[i]:
                # nan here still allows the output array to be typed
                results_list.append(np.nan)
                continue
            function = getattr(objects[i], function_name)
            args = [batch_arg[i] for batch_arg in batch_args]
            results_list.append(function(*args))
        return results_list

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

    def _validate_call(
        self, function: Callable, mask: None | np.ndarray, kwargs: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Check call arguments against the function signature and pool size.

        Returns:
            np.ndarray: the mask, with None replaced by all True
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
            return mask
        return np.full((self.instances,), True, dtype=bool)

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
        mask = self._validate_call(function, mask, kwargs)

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


###############################################################################
###############################################################################

_SharedArraySpec = tuple[str, tuple[int, ...], str]
"""(shared memory block name, shape, dtype string) for an array another process can attach to"""


def _create_shared_array(
    shape: tuple[int, ...], dtype: np.dtype | type
) -> tuple[SharedMemory, np.ndarray]:
    """Allocate an array in a new shared memory block owned by this process."""
    dtype = np.dtype(dtype)
    # zero sized blocks aren't allowed
    size = max(int(np.prod(shape)) * dtype.itemsize, 1)
    block = SharedMemory(create=True, size=size)
    return block, np.ndarray(shape, dtype=dtype, buffer=block.buf)


def _shared_array_spec(block: SharedMemory, array: np.ndarray) -> _SharedArraySpec:
    """Describe a shared array so another process can attach to it."""
    return (block.name, array.shape, array.dtype.str)


def _attach_shared_array(spec: _SharedArraySpec) -> tuple[SharedMemory, np.ndarray]:
    """Attach to a shared array created by another process.

    SharedMemory registers every block it opens with the resource tracker, which
    unlinks it when the tracker shuts down even if another process owns it
    (Python 3.13 adds track=False for this). Only the manager should own blocks.
    """
    name, shape, dtype = spec
    register = resource_tracker.register
    resource_tracker.register = lambda *args, **kwargs: None
    try:
        block = SharedMemory(name=name)
    finally:
        resource_tracker.register = register
    return block, np.ndarray(shape, dtype=dtype, buffer=block.buf)


def _is_shareable(arg: Any) -> bool:
    """Whether an argument can be passed through a shared numeric buffer."""
    return isinstance(arg, np.ndarray) and arg.dtype.kind in "biufc"


@dataclass
class _Channel:
    """Manager side shared buffers for one function signature."""

    id: int
    mask: np.ndarray
    inputs: dict[str, np.ndarray]
    blocks: list[SharedMemory]
    output: np.ndarray | None = None


@dataclass
class _WorkerChannel:
    """Worker side view of a _Channel."""

    function_name: str
    start: int
    """Row of this worker's first object in the shared arrays"""
    mask: np.ndarray
    inputs: list[np.ndarray]
    blocks: list[SharedMemory]
    output: np.ndarray | None = None
    fill: Any = np.nan
    """Output value for masked rows"""


class SharedMemoryVectorPool(VectorPool[T]):
    """VectorPool that passes call arguments and results through shared memory.

    The pipe based VectorPool pickles every argument and result on every call.
    Here each function signature (function name, argument names, shapes and dtypes)
    gets shared buffers the first time it is called. Later calls write their
    arguments into those buffers in place, and workers only receive a small
    message naming the buffers.

    Results also go through shared memory once a call with no masked rows shows
    they are a numeric array. Other results (e.g. None from reset) and
    non-numeric arguments go through the pipes, the same as VectorPool.

    Unlike VectorPool, masked rows of a shared integer or bool result are 0 instead of nan.
    """

    def __init__(
        self,
        instances: int,
        cls: Type[T],
        num_workers: None | int = None,
        **kwargs,
    ) -> None:
        # set before workers start, since a failed start calls _cleanup
        self._channels: dict[tuple, _Channel] = {}
        super().__init__(instances, cls, num_workers, **kwargs)
        worker_objects = [len(rows) for rows in self._batch_data(np.arange(self.instances))]
        self._worker_starts: list[int] = np.cumsum([0] + worker_objects[:-1]).tolist()

    def call(
        self,
        function: Callable,
        mask: None | NDArray[Shape["*"], Bool],
        **kwargs: NDArray[Shape["*, ..."], Number],
    ) -> NDArray[Shape["*, ..."], Any]:
        mask = self._validate_call(function, mask, kwargs)
        if not all(_is_shareable(arg) for arg in kwargs.values()):
            return super().call(function, mask, **kwargs)

        key = (function.__name__,) + tuple(
            (name, arg.shape, arg.dtype.str) for name, arg in kwargs.items()
        )
        channel = self._channels.get(key)
        if channel is None:
            channel = self._open_channel(key, function.__name__, kwargs)

        channel.mask[:] = mask
        for name, arg in kwargs.items():
            channel.inputs[name][:] = arg

        results = self._send_to_workers(
            Message.Manager.CALL_CHANNEL, [channel.id] * len(self.pipes)
        )
        if channel.output is not None:
            # copy, since the next call overwrites the buffer
            return channel.output.copy()

        result = np.concatenate(results)
        if mask.all() and result.dtype.kind in "biufc":
            self._set_channel_output(channel, result)
        return result

    def _send_to_workers(self, command: Message.Manager, payloads: list) -> list:
        """Send one payload to each worker, then gather their results in order."""
        for pipe, payload in zip(self.pipes, payloads):
            try:
                pipe.send((command, payload))
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to send data to worker: {e}")
        return [VectorPool._expect_success(pipe) for pipe in self.pipes]

    def _open_channel(
        self, key: tuple, function_name: str, kwargs: dict[str, np.ndarray]
    ) -> _Channel:
        """Allocate shared input buffers for a function signature and share them with the workers."""
        mask_block, mask = _create_shared_array((self.instances,), bool)
        channel = _Channel(id=len(self._channels), mask=mask, inputs={}, blocks=[mask_block])
        # register before sending so _cleanup frees the blocks if sending fails
        self._channels[key] = channel

        input_specs = []
        for name, arg in kwargs.items():
            block, array = _create_shared_array(arg.shape, arg.dtype)
            channel.blocks.append(block)
            channel.inputs[name] = array
            input_specs.append(_shared_array_spec(block, array))

        mask_spec = _shared_array_spec(mask_block, mask)
        self._send_to_workers(
            Message.Manager.OPEN_CHANNEL,
            [
                (channel.id, function_name, start, mask_spec, input_specs)
                for start in self._worker_starts
            ],
        )
        return channel

    def _set_channel_output(self, channel: _Channel, result: np.ndarray) -> None:
        """Allocate a shared output buffer shaped like result and share it with the workers."""
        block, output = _create_shared_array(result.shape, result.dtype)
        channel.blocks.append(block)
        # masked rows can only hold nan if the output is floating point
        fill = np.nan if result.dtype.kind in "fc" else 0
        self._send_to_workers(
            Message.Manager.SET_CHANNEL_OUTPUT,
            [(channel.id, _shared_array_spec(block, output), fill)] * len(self.pipes),
        )
        channel.output = output

    @staticmethod
    def _handle_message(
        objects: list, state: dict[str, Any], command: Message.Manager, data: Any
    ) -> tuple[Message.Worker, Any]:
        channels: dict[int, _WorkerChannel] = state.setdefault("channels", {})

        if command == Message.Manager.OPEN_CHANNEL:
            channel_id, function_name, start, mask_spec, input_specs = data
            mask_block, mask = _attach_shared_array(mask_spec)
            blocks, inputs = [mask_block], []
            for spec in input_specs:
                block, array = _attach_shared_array(spec)
                blocks.append(block)
                inputs.append(array)
            channels[channel_id] = _WorkerChannel(
                function_name=function_name,
                start=start,
                mask=mask,
                inputs=inputs,
                blocks=blocks,
            )
            return (Message.Worker.SUCCESS, None)

        if command == Message.Manager.SET_CHANNEL_OUTPUT:
            channel_id, spec, fill = data
            channel = channels[channel_id]
            block, channel.output = _attach_shared_array(spec)
            channel.blocks.append(block)
            channel.fill = fill
            return (Message.Worker.SUCCESS, None)

        if command == Message.Manager.CALL_CHANNEL:
            channel = channels[data]
            rows = slice(channel.start, channel.start + len(objects))
            mask = channel.mask[rows]
            # copy this worker's rows, since objects may hold on to their arguments
            batch_args = [array[rows].copy() for array in channel.inputs]
            results_list = VectorPool._call_objects(
                objects, channel.function_name, mask, batch_args
            )
            if channel.output is None:
                return (Message.Worker.SUCCESS, np.stack(results_list, axis=0))

            output = channel.output[rows]
            for i, result in enumerate(results_list):
                output[i] = result if mask[i] else channel.fill
            return (Message.Worker.SUCCESS, None)

        return VectorPool._handle_message(objects, state, command, data)

    def _cleanup(self) -> None:
        """Clean up worker processes, pipes, and shared memory."""
        super()._cleanup()
        blocks = [block for channel in self._channels.values() for block in channel.blocks]
        # drop the array views first, since a block can't be closed while they exist
        self._channels.clear()
        for block in blocks:
            try:
                block.close()
            except BufferError:
                pass
            try:
                block.unlink()
            except FileNotFoundError:
                pass
