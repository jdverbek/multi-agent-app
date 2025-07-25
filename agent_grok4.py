import asyncio
import os
import requests
from agent_base import Agent
from tasks import Task

CHAT_URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"

class AgentGrok4(Agent):
    """Agent wrapper for the Grok-4 API."""

    async def process_task(self, task: Task) -> dict:
        """Process task and return result in standard format."""
        try:
            result = await self.handle(task)
            return {
                'status': 'success',
                'result': result,
                'agent': 'AgentGrok4',
                'task_type': task.type
            }
        except Exception as e:
            return {
                'status': 'error',
                'result': f"Error processing task: {str(e)}",
                'agent': 'AgentGrok4',
                'task_type': task.type
            }

    async def handle(self, task: Task) -> str:
        """Call Grok's API with the task content."""
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "grok-1",  # Example model name
            "messages": [{"role": "user", "content": task.content}],
            "temperature": 0.3,
        }
        response = await asyncio.to_thread(
            requests.post, CHAT_URL_GROQ, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
