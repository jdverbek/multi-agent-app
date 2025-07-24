#!/usr/bin/env python3
"""
Background task system for handling long-running visual flow executions
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from threading import Thread, Lock
from tasks import Task
from visual_flow_executor import VisualBlock, VisualConnection

class BackgroundTaskManager:
    """Manages background execution of long-running tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        self.cleanup_interval = 3600  # Clean up completed tasks after 1 hour
        
    def start_task(self, visual_blocks, visual_connections, task: Task, controller) -> str:
        """Start a background task and return task ID"""
        task_id = str(uuid.uuid4())
        
        with self.lock:
            self.tasks[task_id] = {
                'id': task_id,
                'status': 'running',
                'started_at': datetime.now(),
                'progress': 0,
                'result': None,
                'error': None,
                'estimated_duration': 1500,  # 25 minutes in seconds
                'current_step': 'Initializing...'
            }
        
        # Start the task in a separate thread
        thread = Thread(
            target=self._execute_task,
            args=(task_id, visual_blocks, visual_connections, task, controller)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _execute_task(self, task_id: str, visual_blocks, visual_connections, task: Task, controller):
        """Execute the task in background"""
        try:
            # Update status
            self._update_task(task_id, status='running', current_step='Starting execution...')
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Execute the visual flow with progress tracking
            result = loop.run_until_complete(
                self._execute_with_progress(task_id, visual_blocks, visual_connections, task, controller)
            )
            
            # Task completed successfully
            self._update_task(
                task_id, 
                status='completed', 
                progress=100, 
                result=result,
                current_step='Completed successfully'
            )
            
        except Exception as e:
            # Task failed
            self._update_task(
                task_id, 
                status='failed', 
                error=str(e),
                current_step=f'Failed: {str(e)}'
            )
        finally:
            loop.close()
    
    async def _execute_with_progress(self, task_id: str, visual_blocks, visual_connections, task: Task, controller):
        """Execute visual flow with progress updates"""
        
        # Step 1: Validation (5%)
        self._update_task(task_id, progress=5, current_step='Validating flow structure...')
        await asyncio.sleep(0.1)  # Small delay to allow progress update
        
        # Step 2: Preparation (10%)
        self._update_task(task_id, progress=10, current_step='Preparing execution environment...')
        await asyncio.sleep(0.1)
        
        # Step 3: Execute with progress tracking (10% - 95%)
        class ProgressTracker:
            def __init__(self, task_manager, task_id):
                self.task_manager = task_manager
                self.task_id = task_id
                self.current_progress = 10
            
            def update_progress(self, step_name: str, progress_increment: int = 5):
                self.current_progress = min(95, self.current_progress + progress_increment)
                self.task_manager._update_task(
                    self.task_id, 
                    progress=self.current_progress, 
                    current_step=step_name
                )
        
        tracker = ProgressTracker(self, task_id)
        
        # Monkey patch the visual executor to report progress
        original_execute = controller.visual_executor.execute_visual_flow
        
        async def execute_with_tracking(blocks, connections, task_obj):
            tracker.update_progress("Analyzing workflow structure...", 5)
            await asyncio.sleep(0.1)
            
            tracker.update_progress("Initializing agents...", 5)
            await asyncio.sleep(0.1)
            
            tracker.update_progress("Starting workflow execution...", 5)
            
            # Execute the actual flow
            result = await original_execute(blocks, connections, task_obj)
            
            tracker.update_progress("Processing results...", 10)
            await asyncio.sleep(0.1)
            
            return result
        
        # Execute with tracking
        result = await execute_with_tracking(visual_blocks, visual_connections, task)
        
        # Step 4: Finalization (95% - 100%)
        self._update_task(task_id, progress=98, current_step='Finalizing results...')
        await asyncio.sleep(0.1)
        
        return result
    
    def _update_task(self, task_id: str, **updates):
        """Update task status"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(updates)
                self.tasks[task_id]['updated_at'] = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        with self.lock:
            task_info = self.tasks.get(task_id)
            if task_info:
                # Calculate elapsed time and ETA
                elapsed = (datetime.now() - task_info['started_at']).total_seconds()
                
                if task_info['status'] == 'running' and task_info['progress'] > 0:
                    estimated_total = (elapsed / task_info['progress']) * 100
                    eta_seconds = max(0, estimated_total - elapsed)
                    task_info['elapsed_seconds'] = int(elapsed)
                    task_info['eta_seconds'] = int(eta_seconds)
                    task_info['eta_formatted'] = self._format_duration(eta_seconds)
                
                task_info['elapsed_formatted'] = self._format_duration(elapsed)
                
            return task_info
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all tasks"""
        with self.lock:
            return dict(self.tasks)
    
    def cleanup_old_tasks(self):
        """Clean up old completed tasks"""
        cutoff_time = datetime.now() - timedelta(seconds=self.cleanup_interval)
        
        with self.lock:
            to_remove = []
            for task_id, task_info in self.tasks.items():
                if (task_info['status'] in ['completed', 'failed'] and 
                    task_info.get('updated_at', task_info['started_at']) < cutoff_time):
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        with self.lock:
            if task_id in self.tasks and self.tasks[task_id]['status'] == 'running':
                self.tasks[task_id]['status'] = 'cancelled'
                self.tasks[task_id]['current_step'] = 'Cancelled by user'
                self.tasks[task_id]['updated_at'] = datetime.now()
                return True
            return False

# Global task manager instance
task_manager = BackgroundTaskManager()

