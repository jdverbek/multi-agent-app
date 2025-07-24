import asyncio
import os
import requests
from agent_base import Agent
from tasks import Task

CHAT_URL_OPENAI = "https://api.openai.com/v1/chat/completions"

class AgentO3Pro(Agent):
    """Agent wrapper for the O3Pro API."""

    async def process_task(self, task: Task) -> dict:
        """Process task and return result in standard format."""
        try:
            result = await self.handle(task)
            return {
                'status': 'success',
                'result': result,
                'agent': 'AgentO3Pro',
                'task_type': task.type
            }
        except Exception as e:
            return {
                'status': 'error',
                'result': f"Error processing task: {str(e)}",
                'agent': 'AgentO3Pro',
                'task_type': task.type
            }

    async def handle(self, task: Task) -> str:
        """Call OpenAI's GPT-3.5 (or O3Pro) model with the task content."""
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-3.5-turbo",  # Example model id for O3Pro
            "messages": [{"role": "user", "content": task.content}],
            "temperature": 0.3,
        }
        response = await asyncio.to_thread(
            requests.post, CHAT_URL_OPENAI, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
