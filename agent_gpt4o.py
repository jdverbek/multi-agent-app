import asyncio
import os
import requests
import time
import json
from agent_base import Agent
from tasks import Task

CHAT_URL_OPENAI = "https://api.openai.com/v1/chat/completions"

class AgentGPT4o(Agent):
    """Agent wrapper for the GPT-4o API."""

    async def process_task(self, task: Task) -> dict:
        """Process task and return result in standard format."""
        print(f"[AGENT_GPT4O] ========== STARTING TASK PROCESSING ==========")
        print(f"[AGENT_GPT4O] Task type: {task.type}")
        print(f"[AGENT_GPT4O] Task role: {task.role}")
        print(f"[AGENT_GPT4O] Task content length: {len(task.content)} characters")
        print(f"[AGENT_GPT4O] Task content preview: {task.content[:200]}...")
        
        start_time = time.time()
        try:
            print(f"[AGENT_GPT4O] Calling handle() method...")
            result = await self.handle(task)
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"[AGENT_GPT4O] ✅ Task completed successfully in {duration:.2f} seconds")
            print(f"[AGENT_GPT4O] Result length: {len(result)} characters")
            print(f"[AGENT_GPT4O] Result preview: {result[:200]}...")
            
            return {
                'status': 'success',
                'result': result,
                'agent': 'AgentGPT4o',
                'task_type': task.type,
                'duration': duration
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[AGENT_GPT4O] ❌ Task failed after {duration:.2f} seconds")
            print(f"[AGENT_GPT4O] Error type: {type(e).__name__}")
            print(f"[AGENT_GPT4O] Error message: {str(e)}")
            
            return {
                'status': 'error',
                'result': f"Error processing task: {str(e)}",
                'agent': 'AgentGPT4o',
                'task_type': task.type,
                'duration': duration,
                'error_type': type(e).__name__
            }

    async def handle(self, task: Task) -> str:
        """Call OpenAI's GPT-4o model with the task content."""
        print(f"[AGENT_GPT4O] ========== STARTING API CALL ==========")
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            print(f"[AGENT_GPT4O] ❌ ERROR: No OPENAI_API_KEY found in environment")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"[AGENT_GPT4O] ✅ API key found (length: {len(api_key)} chars)")
        print(f"[AGENT_GPT4O] API key prefix: {api_key[:10]}...")
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"[AGENT_GPT4O] Headers prepared")
        
        # Prepare payload
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": task.content}],
            "temperature": 0.3,
        }
        print(f"[AGENT_GPT4O] Payload prepared:")
        print(f"[AGENT_GPT4O]   Model: {payload['model']}")
        print(f"[AGENT_GPT4O]   Temperature: {payload['temperature']}")
        print(f"[AGENT_GPT4O]   Message content length: {len(payload['messages'][0]['content'])} chars")
        
        # Make API call
        print(f"[AGENT_GPT4O] 🚀 Making API request to: {CHAT_URL_OPENAI}")
        print(f"[AGENT_GPT4O] Request timeout: 60 seconds")
        
        request_start = time.time()
        try:
            print(f"[AGENT_GPT4O] ⏳ Sending request...")
            response = await asyncio.to_thread(
                requests.post, CHAT_URL_OPENAI, headers=headers, json=payload, timeout=60
            )
            request_end = time.time()
            request_duration = request_end - request_start
            
            print(f"[AGENT_GPT4O] ✅ Response received in {request_duration:.2f} seconds")
            print(f"[AGENT_GPT4O] Response status code: {response.status_code}")
            print(f"[AGENT_GPT4O] Response headers: {dict(response.headers)}")
            
            # Check response status
            if response.status_code != 200:
                print(f"[AGENT_GPT4O] ❌ HTTP Error: {response.status_code}")
                print(f"[AGENT_GPT4O] Response text: {response.text}")
                response.raise_for_status()
            
            # Parse response
            print(f"[AGENT_GPT4O] 📝 Parsing JSON response...")
            try:
                response_json = response.json()
                print(f"[AGENT_GPT4O] ✅ JSON parsed successfully")
                print(f"[AGENT_GPT4O] Response keys: {list(response_json.keys())}")
                
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    print(f"[AGENT_GPT4O] ✅ Content extracted successfully")
                    print(f"[AGENT_GPT4O] Content length: {len(content)} characters")
                    print(f"[AGENT_GPT4O] Content preview: {content[:200]}...")
                    
                    # Log usage info if available
                    if 'usage' in response_json:
                        usage = response_json['usage']
                        print(f"[AGENT_GPT4O] Token usage: {usage}")
                    
                    return content
                else:
                    print(f"[AGENT_GPT4O] ❌ ERROR: No choices in response")
                    print(f"[AGENT_GPT4O] Full response: {json.dumps(response_json, indent=2)}")
                    raise ValueError("No choices found in OpenAI response")
                    
            except json.JSONDecodeError as e:
                print(f"[AGENT_GPT4O] ❌ JSON Parse Error: {str(e)}")
                print(f"[AGENT_GPT4O] Raw response: {response.text}")
                raise
                
        except requests.exceptions.Timeout:
            request_end = time.time()
            request_duration = request_end - request_start
            print(f"[AGENT_GPT4O] ⏰ Request timed out after {request_duration:.2f} seconds")
            raise
        except requests.exceptions.ConnectionError as e:
            request_end = time.time()
            request_duration = request_end - request_start
            print(f"[AGENT_GPT4O] 🔌 Connection error after {request_duration:.2f} seconds: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            request_end = time.time()
            request_duration = request_end - request_start
            print(f"[AGENT_GPT4O] 🚫 Request error after {request_duration:.2f} seconds: {str(e)}")
            raise
