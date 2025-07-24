import asyncio
import os
import requests
from agent_base import Agent
from tasks import Task

CHAT_URL_OPENAI = "https://api.openai.com/v1/chat/completions"

class AgentGPT4o(Agent):
    """Agent wrapper for the GPT-4o API."""

    async def handle(self, task: Task) -> str:
        """Call OpenAI's GPT-4o model with the task content."""
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": task.content}],
            "temperature": 0.3,
        }
        response = await asyncio.to_thread(
            requests.post, CHAT_URL_OPENAI, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
