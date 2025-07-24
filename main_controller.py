import asyncio
from typing import Dict

from tasks import Task
from agent_grok4 import AgentGrok4
from agent_gpt4o import AgentGPT4o
from agent_o3pro import AgentO3Pro

class MainController:
    """Central orchestrator that dispatches tasks to specialized agents."""

    def __init__(self):
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.agents: Dict[str, object] = {
            "Manager": AgentGrok4("AgentGrok4"),
            "CodeVerifier": AgentO3Pro("AgentO3Pro"),
            "Developer": AgentGPT4o("AgentGPT4o"),
        }

    async def submit_task(self, task: Task) -> None:
        await self.queue.put(task)

    async def run(self) -> None:
        """Continuously process tasks from the queue."""
        while True:
            task = await self.queue.get()
            agent = self._select_agent(task)
            response = await agent.handle(task)
            task.response = response
            self.queue.task_done()

    def _select_agent(self, task: Task):
        """Select agent based on desired role or fallback."""
        if task.role and task.role in self.agents:
            return self.agents[task.role]
        return self.agents.get("Developer")
