import logging
from typing import Callable, Awaitable, Any, Optional, Union
import asyncio
from dataclasses import dataclass, field
import traceback
from copy import deepcopy

logger = logging.getLogger("tokenflow")

type TResult = Any
type TParams = Any

##### Action-related -> #####

@dataclass(frozen=True)
class ActionDone:
    pass

@dataclass(frozen=True)
class ActionContinue:
    continue_params: TParams

type TokenFlowAction = ActionDone | ActionContinue

##### <- Action-related #####

##### Request-related -> #####

@dataclass
class TokenFlowRequestTrajectoryElement:
    request_params: TParams
    result: TResult
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

##### <- Request-related #####

##### Event-related -> #####

type TEventPayload = Any

@dataclass
class TokenFlowDoneEvent:
    request_id: int
    result: TResult
    request_trajectory: TokenFlowRequestTrajectory
    payload: Optional[TEventPayload]

@dataclass
class TokenFlowContinueEvent:
    request_id: int
    continue_params: TParams
    result: TResult
    request_trajectory: TokenFlowRequestTrajectory
    payload: Optional[TEventPayload]

@dataclass
class TokenFlowCancelledEvent:
    request_id: int
    request_trajectory: TokenFlowRequestTrajectory

@dataclass
class TokenFlowUserExceptionEvent:
    request_id: int
    exception_str: str
    traceback_str: str
    request_trajectory: TokenFlowRequestTrajectory

type TokenFlowEvent = Union[
    TokenFlowDoneEvent,
    TokenFlowContinueEvent,
    TokenFlowCancelledEvent,
    TokenFlowUserExceptionEvent
]

##### <- Event-related #####

##### Result-queue related -> #####

@dataclass
class _RequestCancelled:
    pass

@dataclass
class UnexpectedUserException:
    exception_str: str
    traceback_str: str

type _TResultQueueNormal = tuple[TokenFlowRequest, tuple[TokenFlowAction, TResult, TEventPayload]]
type _TResultQueueCancelled = tuple[TokenFlowRequest, _RequestCancelled]
type _TResultQueueUserException = tuple[TokenFlowRequest, UnexpectedUserException]

##### <- Result-queue related #####

##### Signals -> #####

SOFT_STOP_SIGNAL: object = object()
RESULT_HANDLER_DONE_SIGNAL: object = object()

##### <- Signals #####

class TokenFlowTask:
    request_fn: Callable[[TokenFlowRequestTrajectory], Awaitable[tuple[TokenFlowAction, TResult, TEventPayload]]]
    request_params: list
    task_queue: asyncio.Queue[TokenFlowRequest]
    result_queue: asyncio.Queue[Union[_TResultQueueNormal, _TResultQueueCancelled, _TResultQueueUserException]]
    target_stay: bool
    workers: list[asyncio.Task]
    done_tasks: list[TokenFlowRequest]
    result_handler: Optional[asyncio.Task]
    todo_tasks: list[TokenFlowRequest]
    finish_event: asyncio.Event
    n_tasks: int

    event_callback: Optional[Callable[[TokenFlowEvent], None]]

    def __init__(
            self,
            request_fn: Callable[[TokenFlowRequestTrajectory], Awaitable[tuple[TokenFlowAction, TResult, TEventPayload]]],
            request_params: list,
            event_callback: Optional[Callable[[TokenFlowEvent], None]] = None
        ) -> None:
        self.request_fn = request_fn
        self.request_params = request_params
        self.target_stay = False
        self.done_tasks = []
        self.todo_tasks = []
        self.finish_event = asyncio.Event()

        self.event_callback = event_callback
        
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
            logger.debug(f"worker {worker.get_name()} started.")

    def _done_request(self, result: TokenFlowRequest) -> bool:
        self.done_tasks.append(result)
        
        if len(self.done_tasks) == self.n_tasks:
            return True
        
        return False

    async def wait_for_done(self) -> list[TokenFlowRequest]:
        await self.finish_event.wait()

        ordered_results = sorted(
            self.done_tasks,
            key=lambda x: x.id
        )

        return ordered_results
    
    def _notify_event_callback(self, event: TokenFlowEvent) -> None:
        if self.event_callback is None: return

        self.event_callback(event)

    async def soft_stop(self) -> None:
        self.target_stay = True

        for _ in self.workers:
            # send stop signal to workers
            self.task_queue.put_nowait(SOFT_STOP_SIGNAL)

        for worker in self.workers:
            await worker

        self.result_queue.put_nowait(RESULT_HANDLER_DONE_SIGNAL)
        await self.result_handler

        # Handle left elements in the task queue
        while not self.task_queue.empty():
            request = self.task_queue.get_nowait()
            if request is SOFT_STOP_SIGNAL: continue
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

        self.result_queue.put_nowait(RESULT_HANDLER_DONE_SIGNAL)
        await self.result_handler

        # Handle left elements in the task queue
        while not self.task_queue.empty():
            request = self.task_queue.get_nowait()
            if request is SOFT_STOP_SIGNAL: 
                # This should never happen
                assert False, "Task queue should not contain stop signal when hard stop."
                continue
            
            self.todo_tasks.append(request)

        # Clean up

        self.workers = []
        self.result_handler = None

    async def _result_handler(self) -> None:
        while True:
            element = await self.result_queue.get()
            
            if element is RESULT_HANDLER_DONE_SIGNAL:
                break

            match element:
                case (request, (ActionDone(), result, event_payload)):
                    request.request_trajectory.mark_last_done(result)

                    is_all_done = self._done_request(request)
                    self._notify_event_callback(TokenFlowDoneEvent(
                        request.id,
                        result,
                        deepcopy(request.request_trajectory),
                        event_payload
                    ))

                    if is_all_done:
                        self.target_stay = True
                        for _ in self.workers: self.task_queue.put_nowait(SOFT_STOP_SIGNAL)
                        for worker in self.workers: await worker
                        self.workers = []
                        self.finish_event.set()
                        break

                case (request, (ActionContinue(continue_params), result, event_payload)):
                    request.request_trajectory.mark_last_continue(result, continue_params)

                    request_trajectory_backup = deepcopy(request.request_trajectory)

                    request.request_trajectory.add(TokenFlowRequestTrajectoryElement(
                        request_params=continue_params,
                        result=None,
                        action=None
                    ))

                    self.task_queue.put_nowait(request)
                    self._notify_event_callback(TokenFlowContinueEvent(
                        request.id,
                        continue_params,
                        result,
                        request_trajectory_backup,
                        event_payload
                    ))

                case (request, _RequestCancelled()):
                    # This means that the request was cancelled
                    # and we need to put it back to the todo_tasks
                    self.todo_tasks.append(request)
                    self._notify_event_callback(TokenFlowCancelledEvent(
                        request.id,
                        deepcopy(request.request_trajectory)
                    ))

                case (request, UnexpectedUserException(error_str, traceback_str)):
                    request.request_trajectory.mark_last_done(UnexpectedUserException(error_str, traceback_str))

                    is_all_done = self._done_request(request)
                    self._notify_event_callback(TokenFlowUserExceptionEvent(
                        request.id,
                        error_str,
                        traceback_str,
                        deepcopy(request.request_trajectory)
                    ))

                    if is_all_done:
                        self.target_stay = True
                        for _ in self.workers: self.task_queue.put_nowait(SOFT_STOP_SIGNAL)
                        for worker in self.workers: await worker
                        self.workers = []
                        self.finish_event.set()
                        break
                case _:
                    # This should never happen
                    assert False, f"Should never happen. {element}"

    async def _worker(self) -> None:
        while True:
            if self.target_stay: break

            try:
                request = None
                request = await self.task_queue.get()

                if request is SOFT_STOP_SIGNAL:
                    break
        
                action, result, event_payload = await self.request_fn(request.request_trajectory)

                self.result_queue.put_nowait((request, (action, result, event_payload)))

            except asyncio.CancelledError:
                if request is None:
                    break

                # If hard stop, we need to retry the request next time started
                # and we can do this by directly putting the request in the result queue.
                # Since when the action of the last element of the trajectory is None
                # the result handler will put it back to the todo_tasks.
                self.result_queue.put_nowait((request, _RequestCancelled()))

                break

            except Exception as e:
                # this means that the user's function raised an exception
                # which is disallowed in TokenFlow
                logger.error(f"user function raised an exception: {e} \n {traceback.format_exc()}")

                self.result_queue.put_nowait((request, UnexpectedUserException(
                    str(e),
                    traceback.format_exc()
                )))

            finally:
                if request is not None and request is not SOFT_STOP_SIGNAL:
                    self.task_queue.task_done()
    
