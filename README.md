# TokenFlow
TokenFlow is designed for reliable large-scale concurrent network requests, especially for tasks such as requesting large language models (LLMs) for data synthesis.

The design goal of TokenFlow is to implement a task framework for applications such as large-scale language model data synthesis in a sufficiently concise manner, while being flexible and incorporating robust error handling, statistical information, pause and checkpoint resumption, progress persistence, and other functionalities. The entire module of TokenFlow is contained within a single Python file, making it easy for users to integrate into their projects.

## Usage Examples

### Simple Task

```python
from typing import Any
import asyncio
import logging

import openai
from tokenflow import TokenFlowTask, TokenFlowRequestTrajectory, ActionDone, ActionContinue, TokenFlowAction

logging.basicConfig(level=logging.DEBUG)

client = openai.AsyncOpenAI()

async def handler(request_trajectory: TokenFlowRequestTrajectory) -> tuple[TokenFlowAction, Any]:
    if request_trajectory.length() >= 3:
        print("Too much retry, stop")
        return ActionDone(), "Too much retry, stop"
    
    params = request_trajectory.last().request_params
    
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": params}],
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=4,
        )

        result = response.choices[0].message.content

        return ActionDone(), result
    
    except Exception as e:
        return ActionContinue(params), str(e)
    
async def main():
    params = [f"say {i}" for i in range(50)]

    tf_task = TokenFlowTask(
        request_fn=handler,
        request_params=params
    )

    tf_task.run(n_workers=4)

    result = await tf_task.wait_for_done()

    str_result = [r.request_trajectory.last().result for r in result]
    print(str_result)

    print("All done")

if __name__ == "__main__":
    asyncio.run(main())
```

### Pause and Resume

```python
from typing import Any
import asyncio
import logging

import openai
from tokenflow import TokenFlowTask, TokenFlowRequestTrajectory, ActionDone, ActionContinue, TokenFlowAction

logging.basicConfig(level=logging.DEBUG)

client = openai.AsyncOpenAI()

async def handler(request_trajectory: TokenFlowRequestTrajectory) -> tuple[TokenFlowAction, Any]:
    if request_trajectory.length() >= 3:
        print("Too much retry, stop")
        return ActionDone(), "Too much retry, stop"
    
    params = request_trajectory.last().request_params
    
    try:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": params}],
            model="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=4,
        )

        result = response.choices[0].message.content

        return ActionDone(), result
    
    except Exception as e:
        return ActionContinue(params), str(e)
    
async def main():
    params = [f"say {i}" for i in range(50)]

    tf_task = TokenFlowTask(
        request_fn=handler,
        request_params=params
    )

    tf_task.run(n_workers=2)

    await asyncio.sleep(5)

    print("Pause task")
    await tf_task.soft_stop() # or await tf_task.hard_stop() to immediately stop the task
    print("Press enter to continue...")
    input()
    print("Resume task")
    tf_task.run(n_workers=2)

    result = await tf_task.wait_for_done()

    str_result = [r.request_trajectory.last().result for r in result]
    print(str_result)

    print("All done")

if __name__ == "__main__":
    asyncio.run(main())
```

## TODO
- [ ] Statistics
- [ ] Progress persistence (eg. save to file)
