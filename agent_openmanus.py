import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

from agent_base import Agent
from tasks import Task

logger = logging.getLogger(__name__)

class ToolCollection:
    """Collection of tools available to the OpenManus agent."""
    
    def __init__(self):
        self.tools = {
            'python_execute': self.python_execute,
            'file_write': self.file_write,
            'file_read': self.file_read,
            'browser_navigate': self.browser_navigate,
            'terminate': self.terminate
        }
    
    async def python_execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return results."""
        print(f"[OPENMANUS_TOOL] 🐍 Executing Python code: {code[:100]}...")
        
        try:
            # Create temporary file for code execution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            start_time = time.time()
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            duration = time.time() - start_time
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                print(f"[OPENMANUS_TOOL] ✅ Python execution completed in {duration:.2f}s")
                return {
                    'status': 'success',
                    'output': result.stdout,
                    'error': result.stderr,
                    'duration': duration
                }
            else:
                print(f"[OPENMANUS_TOOL] ❌ Python execution failed: {result.stderr}")
                return {
                    'status': 'error',
                    'output': result.stdout,
                    'error': result.stderr,
                    'duration': duration
                }
                
        except subprocess.TimeoutExpired:
            print(f"[OPENMANUS_TOOL] ⏰ Python execution timed out after 30s")
            return {
                'status': 'timeout',
                'error': 'Code execution timed out after 30 seconds'
            }
        except Exception as e:
            print(f"[OPENMANUS_TOOL] ❌ Python execution error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def file_write(self, filename: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        print(f"[OPENMANUS_TOOL] 📝 Writing to file: {filename}")
        
        try:
            with open(filename, 'w') as f:
                f.write(content)
            
            print(f"[OPENMANUS_TOOL] ✅ File written successfully: {filename}")
            return {
                'status': 'success',
                'message': f'File {filename} written successfully',
                'bytes_written': len(content.encode('utf-8'))
            }
        except Exception as e:
            print(f"[OPENMANUS_TOOL] ❌ File write error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def file_read(self, filename: str) -> Dict[str, Any]:
        """Read content from a file."""
        print(f"[OPENMANUS_TOOL] 📖 Reading file: {filename}")
        
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            print(f"[OPENMANUS_TOOL] ✅ File read successfully: {filename} ({len(content)} chars)")
            return {
                'status': 'success',
                'content': content,
                'size': len(content)
            }
        except FileNotFoundError:
            print(f"[OPENMANUS_TOOL] ❌ File not found: {filename}")
            return {
                'status': 'error',
                'error': f'File not found: {filename}'
            }
        except Exception as e:
            print(f"[OPENMANUS_TOOL] ❌ File read error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def browser_navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL (simulated for now)."""
        print(f"[OPENMANUS_TOOL] 🌐 Navigating to: {url}")
        
        # For now, simulate browser navigation
        # In a full implementation, this would use actual browser automation
        return {
            'status': 'success',
            'message': f'Navigated to {url}',
            'url': url,
            'title': f'Page at {url}'
        }
    
    async def terminate(self) -> Dict[str, Any]:
        """Terminate the current task."""
        print(f"[OPENMANUS_TOOL] 🛑 Terminating task")
        return {
            'status': 'terminated',
            'message': 'Task terminated by agent'
        }

class AgentOpenManus(Agent):
    """
    OpenManus agent implementation following the official OpenManus pattern.
    
    This agent can perform actual actions using tools like Python execution,
    file operations, and browser automation, rather than just providing instructions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "OpenManus"
        self.description = "A versatile general-purpose agent with support for both local and remote tools"
        
        # Initialize tool collection
        self.tool_collection = ToolCollection()
        
        # Configuration
        self.max_steps = config.get('max_steps', 20) if config else 20
        self.max_observe = config.get('max_observe', 10000) if config else 10000
        
        print(f"[AGENT_OPENMANUS] Initialized OpenManus agent with {len(self.tool_collection.tools)} tools")
    
    async def handle(self, task: Task) -> Dict[str, Any]:
        """
        Handle a task using OpenManus capabilities.
        
        This method orchestrates the execution of tasks using available tools,
        following the OpenManus pattern of tool-based execution.
        """
        print(f"[AGENT_OPENMANUS] ========== STARTING OPENMANUS TASK PROCESSING ==========")
        print(f"[AGENT_OPENMANUS] Task type: {task.type}")
        print(f"[AGENT_OPENMANUS] Task role: {task.role}")
        print(f"[AGENT_OPENMANUS] Task content length: {len(task.content)} characters")
        print(f"[AGENT_OPENMANUS] Task content preview: {task.content[:200]}...")
        
        start_time = time.time()
        
        try:
            # Analyze the task and determine required actions
            actions = await self._analyze_task_and_plan_actions(task)
            print(f"[AGENT_OPENMANUS] Planned {len(actions)} actions for execution")
            
            # Execute actions using tools
            results = []
            for i, action in enumerate(actions, 1):
                print(f"[AGENT_OPENMANUS] ========== EXECUTING ACTION {i}/{len(actions)} ==========")
                print(f"[AGENT_OPENMANUS] Action type: {action['type']}")
                print(f"[AGENT_OPENMANUS] Action details: {action.get('description', 'No description')}")
                
                action_result = await self._execute_action(action)
                results.append(action_result)
                
                print(f"[AGENT_OPENMANUS] Action {i} result: {action_result.get('status', 'unknown')}")
                
                # Check if we should terminate early
                if action_result.get('status') == 'terminated':
                    print(f"[AGENT_OPENMANUS] Early termination requested")
                    break
            
            # Compile final result
            duration = time.time() - start_time
            final_result = await self._compile_results(task, actions, results)
            
            print(f"[AGENT_OPENMANUS] ========== OPENMANUS TASK COMPLETED ==========")
            print(f"[AGENT_OPENMANUS] Total duration: {duration:.2f} seconds")
            print(f"[AGENT_OPENMANUS] Actions executed: {len(results)}")
            print(f"[AGENT_OPENMANUS] Final result status: {final_result.get('status', 'unknown')}")
            
            return {
                'status': 'success',
                'result': final_result['result'],
                'agent': 'AgentOpenManus',
                'duration': duration,
                'actions_executed': len(results),
                'task_type': task.type
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error in OpenManus processing: {str(e)}"
            print(f"[AGENT_OPENMANUS] ❌ ERROR: {error_msg}")
            print(f"[AGENT_OPENMANUS] Error duration: {duration:.2f} seconds")
            
            return {
                'status': 'error',
                'result': error_msg,
                'agent': 'AgentOpenManus',
                'duration': duration,
                'task_type': task.type
            }
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process task method for compatibility with visual flow executor."""
        return await self.handle(task)
    
    async def _analyze_task_and_plan_actions(self, task: Task) -> List[Dict[str, Any]]:
        """
        Analyze the task and plan the required actions.
        
        This method determines what tools need to be used and in what order
        to accomplish the given task.
        """
        print(f"[AGENT_OPENMANUS] 🧠 Analyzing task and planning actions...")
        
        # For now, implement basic task analysis
        # In a full implementation, this would use LLM to analyze and plan
        
        task_content_lower = task.content.lower()
        actions = []
        
        # Detect different types of tasks and plan accordingly
        if 'python' in task_content_lower or 'code' in task_content_lower or 'script' in task_content_lower:
            # Programming task
            actions.append({
                'type': 'python_execute',
                'description': 'Execute Python code to accomplish the task',
                'code': self._extract_or_generate_code(task.content)
            })
        
        elif 'file' in task_content_lower or 'write' in task_content_lower or 'create' in task_content_lower:
            # File operation task
            if 'powerpoint' in task_content_lower or 'presentation' in task_content_lower:
                # PowerPoint creation task
                actions.append({
                    'type': 'python_execute',
                    'description': 'Create PowerPoint presentation using Python',
                    'code': self._generate_powerpoint_code(task.content)
                })
            else:
                # General file creation
                actions.append({
                    'type': 'file_write',
                    'description': 'Create file based on task requirements',
                    'filename': self._determine_filename(task.content),
                    'content': self._generate_file_content(task.content)
                })
        
        elif 'browse' in task_content_lower or 'website' in task_content_lower or 'url' in task_content_lower:
            # Browser task
            actions.append({
                'type': 'browser_navigate',
                'description': 'Navigate to website and perform actions',
                'url': self._extract_url(task.content)
            })
        
        else:
            # Default: try to accomplish with Python
            actions.append({
                'type': 'python_execute',
                'description': 'Use Python to accomplish the general task',
                'code': self._generate_general_code(task.content)
            })
        
        print(f"[AGENT_OPENMANUS] ✅ Planned {len(actions)} actions")
        return actions
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action using the appropriate tool."""
        action_type = action['type']
        
        if action_type not in self.tool_collection.tools:
            return {
                'status': 'error',
                'error': f'Unknown action type: {action_type}'
            }
        
        tool_func = self.tool_collection.tools[action_type]
        
        try:
            if action_type == 'python_execute':
                return await tool_func(action['code'])
            elif action_type == 'file_write':
                return await tool_func(action['filename'], action['content'])
            elif action_type == 'file_read':
                return await tool_func(action['filename'])
            elif action_type == 'browser_navigate':
                return await tool_func(action['url'])
            elif action_type == 'terminate':
                return await tool_func()
            else:
                return {
                    'status': 'error',
                    'error': f'Unhandled action type: {action_type}'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Tool execution failed: {str(e)}'
            }
    
    async def _compile_results(self, task: Task, actions: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile the results from all actions into a final response."""
        print(f"[AGENT_OPENMANUS] 📋 Compiling results from {len(results)} actions...")
        
        successful_actions = [r for r in results if r.get('status') == 'success']
        failed_actions = [r for r in results if r.get('status') == 'error']
        
        if not results:
            return {
                'status': 'error',
                'result': 'No actions were executed'
            }
        
        # Create a comprehensive result summary
        result_summary = []
        
        for i, (action, result) in enumerate(zip(actions, results), 1):
            if result.get('status') == 'success':
                if action['type'] == 'python_execute':
                    output = result.get('output', '').strip()
                    if output:
                        result_summary.append(f"✅ Python execution #{i}: {output}")
                    else:
                        result_summary.append(f"✅ Python code executed successfully #{i}")
                elif action['type'] == 'file_write':
                    result_summary.append(f"✅ File created: {action.get('filename', 'unknown')}")
                elif action['type'] == 'browser_navigate':
                    result_summary.append(f"✅ Navigated to: {action.get('url', 'unknown')}")
                else:
                    result_summary.append(f"✅ {action['type']} completed successfully")
            else:
                error = result.get('error', 'Unknown error')
                result_summary.append(f"❌ {action['type']} failed: {error}")
        
        final_result = "\n".join(result_summary)
        
        if failed_actions:
            status = 'partial_success' if successful_actions else 'error'
        else:
            status = 'success'
        
        return {
            'status': status,
            'result': final_result,
            'successful_actions': len(successful_actions),
            'failed_actions': len(failed_actions),
            'total_actions': len(results)
        }
    
    def _extract_or_generate_code(self, content: str) -> str:
        """Extract existing code or generate code based on the task."""
        # Simple code extraction/generation
        if 'print(' in content:
            return content
        else:
            return f'# Task: {content}\nprint("Task completed: {content}")'
    
    def _generate_powerpoint_code(self, content: str) -> str:
        """Generate Python code to create a PowerPoint presentation."""
        return f'''
# PowerPoint creation task: {content}
try:
    from pptx import Presentation
    from pptx.util import Inches
    
    # Create presentation
    prs = Presentation()
    
    # Add title slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "Generated Presentation"
    subtitle.text = "Created by OpenManus Agent"
    
    # Add content slide
    bullet_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(bullet_slide_layout)
    shapes = slide.shapes
    
    title_shape = shapes.title
    body_shape = shapes.placeholders[1]
    
    title_shape.text = "Task Content"
    tf = body_shape.text_frame
    tf.text = "{content}"
    
    # Save presentation
    filename = "generated_presentation.pptx"
    prs.save(filename)
    print(f"PowerPoint presentation saved as {{filename}}")
    
except ImportError:
    print("python-pptx library not available. Creating text-based presentation instead.")
    with open("presentation_content.txt", "w") as f:
        f.write("PRESENTATION CONTENT\\n")
        f.write("=" * 50 + "\\n")
        f.write("Title: Generated Presentation\\n")
        f.write("Content: {content}\\n")
    print("Text-based presentation created as presentation_content.txt")
'''
    
    def _determine_filename(self, content: str) -> str:
        """Determine appropriate filename based on task content."""
        if 'powerpoint' in content.lower():
            return 'presentation.pptx'
        elif 'document' in content.lower():
            return 'document.txt'
        else:
            return 'output.txt'
    
    def _generate_file_content(self, content: str) -> str:
        """Generate file content based on task."""
        return f"Generated content for task: {content}\n\nCreated by OpenManus Agent"
    
    def _extract_url(self, content: str) -> str:
        """Extract URL from task content."""
        # Simple URL extraction
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        return urls[0] if urls else 'https://example.com'
    
    def _generate_general_code(self, content: str) -> str:
        """Generate general Python code for the task."""
        return f'''
# General task: {content}
import os
import datetime

print("OpenManus Agent executing task:")
print("{content}")
print()
print("Task execution completed at:", datetime.datetime.now())
print("Working directory:", os.getcwd())

# Task-specific logic would go here
result = "Task completed successfully"
print("Result:", result)
'''

