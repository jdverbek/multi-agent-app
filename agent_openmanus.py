import asyncio
import os
import requests
import json
from agent_base import Agent
from tasks import Task

class AgentOpenManus(Agent):
    """Agent wrapper for the OpenManus API integration."""
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name)
        self.config = config or {}
        self.base_url = self.config.get('base_url', 'http://localhost:8000')
        self.api_key = self.config.get('api_key', os.getenv('MANUS_API_KEY', ''))
        
    async def handle(self, task: Task) -> str:
        """Process a task using OpenManus capabilities."""
        try:
            # Prepare the request payload for OpenManus
            payload = {
                "message": task.content,
                "task_type": task.type,
                "role": task.role or "general",
                "config": {
                    "model": self.config.get('model', 'gpt-4o'),
                    "temperature": self.config.get('temperature', 0.3),
                    "max_tokens": self.config.get('max_tokens', 4096)
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request to OpenManus API endpoint
            response = await asyncio.to_thread(
                requests.post, 
                f"{self.base_url}/api/process", 
                headers=headers, 
                json=payload, 
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response from OpenManus')
            else:
                return f"OpenManus API error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"OpenManus connection error: {str(e)}"
        except Exception as e:
            return f"OpenManus processing error: {str(e)}"
    
    def configure(self, config: dict):
        """Update agent configuration."""
        self.config.update(config)
        self.base_url = self.config.get('base_url', self.base_url)
        self.api_key = self.config.get('api_key', self.api_key)

