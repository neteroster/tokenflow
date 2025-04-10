import logging
from typing import Callable, Awaitable, Any, Literal, Optional, Self
import asyncio
from dataclasses import dataclass, field
import traceback

logger = logging.getLogger("tokenflow")

@dataclass(frozen=True)
class ActionDone:
    pass

@dataclass(frozen=True)
class ActionContinue:
    continue_params: Any

type TokenFlowAction = ActionDone | ActionContinue

@dataclass
class TokenFlowRequestTrajectoryElement:
    request_params: Any
    result: Any
    action: Optional[TokenFlowAction] = None

@dataclass
class TokenFlowRequestTrajectory:
    trajectory: list[TokenFlowRequestTrajectoryElement] = field(default_factory=list)

    def empty(self) -> bool:
        return len(self.trajectory) == 0

    def first(self) -> TokenFlowRequestTrajectoryElement:
        return self.trajectory[0]
    
    def last(self) -> TokenFlowRequestTrajectoryElement:
        return self.trajectory[-1]
    
    def length(self) -> int:
        return len(self.trajectory)
    
    def mark_last_done(self, result) -> None:
        self.trajectory[-1].result = result
        self.trajectory[-1].action = ActionDone()
    
    def mark_last_continue(self, result, continue_params) -> None:
        self.trajectory[-1].result = result
        self.trajectory[-1].action = ActionContinue(continue_params)
    
    def add(self, element: TokenFlowRequestTrajectoryElement) -> None:
        self.trajectory.append(element)

@dataclass
class TokenFlowRequest:
    id: int
    request_trajectory: TokenFlowRequestTrajectory

class TokenFlowTask:
    request_fn: Callable[[TokenFlowRequestTrajectory], Awaitable[tuple[TokenFlowAction, Any]]]
    request_params: list
    task_queue: asyncio.Queue[TokenFlowRequest]
    result_queue: asyncio.Queue[TokenFlowRequest]
    target_stay: bool
    workers: list[asyncio.Task]
    done_tasks: list[TokenFlowRequest]
    result_handler: Optional[asyncio.Task]
    todo_tasks: list[TokenFlowRequest]
    finish_event: asyncio.Event
    n_tasks: int

    SOFT_STOP_SIGNAL: object
    RESULT_HANDLER_DONE_SIGNAL: object
    UNEXPECTED_USER_EXCEPTION: object

    def __init__(
            self,
            request_fn: Callable[[TokenFlowRequestTrajectory], Awaitable[tuple[TokenFlowAction, Any]]],
            request_params: list,
        ) -> None:
        self.request_fn = request_fn
        self.request_params = request_params
        self.target_stay = False
        self.done_tasks = []
        self.todo_tasks = []
        self.finish_event = asyncio.Event()

        self.SOFT_STOP_SIGNAL = object()
        self.RESULT_HANDLER_DONE_SIGNAL = object()
        self.UNEXPECTED_USER_EXCEPTION = object()
        
        self.todo_tasks = [
            TokenFlowRequest(
                id=i,
                request_trajectory=TokenFlowRequestTrajectory(
                    trajectory=[
                        TokenFlowRequestTrajectoryElement(
                            request_params=params,
                            result=None,
                            action=None
                        )
                    ]
                )
            )
            for i, params in enumerate(request_params)
        ]

        self.n_tasks = len(self.todo_tasks)

    def run(self, n_workers: int) -> None:
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers = []

        self.target_stay = False

        if self.todo_tasks:
            for request in self.todo_tasks:
                self.task_queue.put_nowait(request)
            self.todo_tasks = []

        self.result_handler = asyncio.create_task(self._result_handler())

        for _ in range(n_workers):
            worker = asyncio.create_task(self._worker())
            self.workers.append(worker)
            logger.debug(f"Worker {worker.get_name()} started.")

    def _done_request(self, result: TokenFlowRequest) -> bool:
        self.done_tasks.append(result)
        
        if len(self.done_tasks) == self.n_tasks:
            return True
        
        return False

    async def wait_for_done(self) -> list[TokenFlowRequest]:
        await self.finish_event.wait()
        logger.debug("All tasks finished.")

        ordered_results = sorted(
            self.done_tasks,
            key=lambda x: x.id
        )

        return ordered_results

    async def soft_stop(self) -> None:
        self.target_stay = True

        for _ in self.workers:
            # send stop signal to workers
            self.task_queue.put_nowait(self.SOFT_STOP_SIGNAL)
        logger.debug("Stop signal sent to workers.")

        for worker in self.workers:
            await worker

        logger.debug("All workers stopped.")

        self.result_queue.put_nowait(self.RESULT_HANDLER_DONE_SIGNAL)
        logger.debug("Stop signal sent to result handler.")
        await self.result_handler
        logger.debug("Result handler stopped.")

        # Handle left elements in the task queue
        while not self.task_queue.empty():
            request = self.task_queue.get_nowait()
            if request is self.SOFT_STOP_SIGNAL: continue
            # We can also assert that the signal, if present, is present
            # behind all normal requests

            self.todo_tasks.append(request)

        # Clean up

        self.workers = []
        self.result_handler = None

    async def hard_stop(self) -> None:
        self.target_stay = True

        for worker in self.workers:
            worker.cancel() # force cancel the worker

        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.debug("All workers stopped.")

        self.result_queue.put_nowait(self.RESULT_HANDLER_DONE_SIGNAL)
        logger.debug("Stop signal sent to result handler.")
        await self.result_handler
        logger.debug("Result handler stopped.")

        # Handle left elements in the task queue
        while not self.task_queue.empty():
            request = self.task_queue.get_nowait()
            if request is self.SOFT_STOP_SIGNAL: 
                # This should never happen
                assert False, "Task queue should not contain stop signal when hard stop"
                continue
            
            self.todo_tasks.append(request)

        # Clean up

        self.workers = []
        self.result_handler = None

    async def _result_handler(self) -> None:
        while True:
            req = await self.result_queue.get()
            
            if req is self.RESULT_HANDLER_DONE_SIGNAL:
                logger.debug("Result handler received stop signal.")
                break

            match req.request_trajectory.last().action:
                case ActionDone():
                    is_all_done = self._done_request(req)
                    if is_all_done:
                        self.target_stay = True
                        for _ in self.workers: self.task_queue.put_nowait(self.SOFT_STOP_SIGNAL)
                        for worker in self.workers: await worker
                        self.workers = []
                        self.finish_event.set()
                        logger.debug("All tasks finished.")
                        break
                case None:
                    self.todo_tasks.append(req)
                case _:
                    raise ValueError("Should never happen")

    async def _worker(self) -> None:
        while True:
            if self.target_stay: break

            try:
                request = None
                request = await self.task_queue.get()

                if request is self.SOFT_STOP_SIGNAL:
                    logger.debug("Worker received stop signal.")
                    break
        
                action, result = await self.request_fn(request.request_trajectory)
                
                match action:
                    case ActionDone():
                        request.request_trajectory.mark_last_done(result)
                        self.result_queue.put_nowait(request)
                    case ActionContinue(continue_params):
                        request.request_trajectory.mark_last_continue(result, continue_params)
                        request.request_trajectory.add(TokenFlowRequestTrajectoryElement(
                            request_params=continue_params,
                            result=None,
                            action=None
                        ))
                        self.task_queue.put_nowait(request)
                    case _:
                        raise ValueError("Should never happen")

            except asyncio.CancelledError:
                if request is None:
                    logger.debug("Worker cancelled, no request is being processed.")
                    break

                logger.debug(f"Force stopping worker, the worker was working on request {request.id}.")

                # If hard stop, we need to retry the request next time started
                # and we can do this by directly putting the request in the result queue.
                # Since when the action of the last element of the trajectory is None
                # the result handler will put it back to the todo_tasks.
                self.result_queue.put_nowait(request)

                break

            except Exception as e:
                # this means that the user's function raised an exception
                # which is disallowed in TokenFlow
                logger.error(f"Worker raised exception: {e}")
                logger.error(traceback.format_exc())

                request.request_trajectory.mark_last_done(self.UNEXPECTED_USER_EXCEPTION)
                self.result_queue.put_nowait(request)

            finally:
                if request is not None and request is not self.SOFT_STOP_SIGNAL:
                    self.task_queue.task_done()

        logger.debug("Worker exiting...")
