import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agent_chain import AgentChain, ChainBlock, FeedbackLoop, BlockType, FeedbackType
from tasks import Task

@dataclass
class VisualBlock:
    """Represents a visual block from the flow designer."""
    id: str
    type: str
    x: float
    y: float
    config: Dict[str, Any]

@dataclass
class VisualConnection:
    """Represents a connection between visual blocks."""
    id: str
    from_block: str
    to_block: str
    type: str  # 'forward' or 'feedback'

@dataclass
class QualityGate:
    """Represents a quality control gate in the workflow."""
    manager_block: str
    worker_block: str
    max_iterations: int
    quality_criteria: str
    current_iteration: int = 0
    
class ManagerWorkerLoop:
    """Handles manager-worker feedback loops with quality control."""
    
    def __init__(self, manager_agent, worker_agent, max_iterations: int = 3):
        self.manager_agent = manager_agent
        self.worker_agent = worker_agent
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.feedback_history = []
    
    async def execute_with_quality_control(self, task: Task) -> dict:
        """Execute task with manager-worker quality control loop."""
        import time
        
        print(f"[MANAGER_WORKER_LOOP] ========== STARTING MANAGER-WORKER EXECUTION ==========")
        print(f"[MANAGER_WORKER_LOOP] Task: {task.type} - {task.content[:100]}...")
        
        # Manager delegates the task
        print(f"[MANAGER_WORKER_LOOP] 📋 MANAGER DELEGATION PHASE...")
        delegation_task = Task(
            type="delegation",
            content=f"Analyze this request and provide clear, detailed instructions for a worker to complete it:\n\n{task.content}",
            role="Manager"
        )
        
        print(f"[MANAGER_WORKER_LOOP] Delegation task content: {delegation_task.content}")
        print(f"[MANAGER_WORKER_LOOP] 🔄 CALLING MANAGER FOR DELEGATION...")
        
        delegation_start = time.time()
        delegation_result = await self.manager_agent.process_task(delegation_task)
        delegation_end = time.time()
        delegation_duration = delegation_end - delegation_start
        
        print(f"[MANAGER_WORKER_LOOP] ✅ Manager delegation completed in {delegation_duration:.2f} seconds")
        print(f"[MANAGER_WORKER_LOOP] Delegation status: {delegation_result.get('status', 'unknown')}")
        print(f"[MANAGER_WORKER_LOOP] Delegation result length: {len(str(delegation_result.get('result', '')))} chars")
        print(f"[MANAGER_WORKER_LOOP] Delegation instructions: {str(delegation_result.get('result', ''))[:300]}...")
        
        if delegation_result.get('status') == 'error':
            print(f"[MANAGER_WORKER_LOOP] ❌ Manager delegation failed: {delegation_result.get('result', '')}")
            return {
                'final_result': delegation_result,
                'quality_approved': False,
                'iterations': 0,
                'max_iterations': self.max_iterations,
                'feedback_history': [],
                'delegation_instructions': delegation_result,
                'status': 'delegation_failed'
            }
        
        # Initialize work result
        work_result = None
        quality_approved = False
        
        # Quality control loop with detailed API call logging
        import time
        start_time = time.time()
        
        print(f"[MANAGER_WORKER_LOOP] ========== STARTING QUALITY CONTROL LOOP ==========")
        print(f"[MANAGER_WORKER_LOOP] Manager agent: {type(self.manager_agent).__name__}")
        print(f"[MANAGER_WORKER_LOOP] Worker agent: {type(self.worker_agent).__name__}")
        print(f"[MANAGER_WORKER_LOOP] Max iterations: {self.max_iterations}")
        print(f"[MANAGER_WORKER_LOOP] Task type: {task.type}")
        print(f"[MANAGER_WORKER_LOOP] Task content: {task.content[:200]}...")
        
        while not quality_approved and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            iteration_start = time.time()
            print(f"[MANAGER_WORKER_LOOP] ========== ITERATION {self.current_iteration}/{self.max_iterations} ==========")
            
            # Worker executes the task
            if self.current_iteration == 1:
                # First iteration - use original task with manager's instructions
                worker_task = Task(
                    type=task.type,
                    content=f"Manager Instructions: {delegation_result.get('result', '')}\n\nOriginal Request: {task.content}",
                    role="Worker"
                )
                print(f"[MANAGER_WORKER_LOOP] First iteration - using manager instructions")
            else:
                # Subsequent iterations - include feedback
                last_feedback = self.feedback_history[-1] if self.feedback_history else ""
                worker_task = Task(
                    type=task.type,
                    content=f"Manager Instructions: {delegation_result.get('result', '')}\n\nOriginal Request: {task.content}\n\nPrevious Work: {work_result.get('result', '')}\n\nManager Feedback: {last_feedback}",
                    role="Worker"
                )
                print(f"[MANAGER_WORKER_LOOP] Iteration {self.current_iteration} - including feedback")
                print(f"[MANAGER_WORKER_LOOP] Previous feedback: {last_feedback[:100]}...")
            
            print(f"[MANAGER_WORKER_LOOP] Worker task content length: {len(worker_task.content)} chars")
            print(f"[MANAGER_WORKER_LOOP] 🔄 CALLING WORKER AGENT...")
            
            worker_start = time.time()
            work_result = await self.worker_agent.process_task(worker_task)
            worker_end = time.time()
            worker_duration = worker_end - worker_start
            
            print(f"[MANAGER_WORKER_LOOP] ✅ Worker completed in {worker_duration:.2f} seconds")
            print(f"[MANAGER_WORKER_LOOP] Worker result status: {work_result.get('status', 'unknown')}")
            print(f"[MANAGER_WORKER_LOOP] Worker result length: {len(str(work_result.get('result', '')))} chars")
            print(f"[MANAGER_WORKER_LOOP] Worker result preview: {str(work_result.get('result', ''))[:200]}...")
            
            if work_result.get('status') == 'error':
                print(f"[MANAGER_WORKER_LOOP] ❌ Worker failed: {work_result.get('result', '')}")
                break
            
            # Manager reviews the work
            review_task = Task(
                type="quality_review",
                content=f"Review this work and determine if it meets quality standards. Respond with either 'APPROVED' or 'NEEDS IMPROVEMENT: [specific feedback]':\n\nOriginal Request: {task.content}\n\nWorker's Output: {work_result.get('result', '')}\n\nIteration: {self.current_iteration}/{self.max_iterations}",
                role="Manager"
            )
            
            print(f"[MANAGER_WORKER_LOOP] Review task content length: {len(review_task.content)} chars")
            print(f"[MANAGER_WORKER_LOOP] 🔍 CALLING MANAGER FOR REVIEW...")
            
            manager_start = time.time()
            review_result = await self.manager_agent.process_task(review_task)
            manager_end = time.time()
            manager_duration = manager_end - manager_start
            
            print(f"[MANAGER_WORKER_LOOP] ✅ Manager review completed in {manager_duration:.2f} seconds")
            print(f"[MANAGER_WORKER_LOOP] Manager review status: {review_result.get('status', 'unknown')}")
            print(f"[MANAGER_WORKER_LOOP] Manager review length: {len(str(review_result.get('result', '')))} chars")
            print(f"[MANAGER_WORKER_LOOP] Manager review: {str(review_result.get('result', ''))}")
            
            if review_result.get('status') == 'error':
                print(f"[MANAGER_WORKER_LOOP] ❌ Manager review failed: {review_result.get('result', '')}")
                break
            
            # Parse manager's decision
            review_text = review_result.get('result', '').lower()
            print(f"[MANAGER_WORKER_LOOP] 📋 ANALYZING MANAGER DECISION...")
            print(f"[MANAGER_WORKER_LOOP] Review text (lowercase): {review_text}")
            
            # Check for approval keywords
            approval_keywords = ['approved', 'accept', 'good', 'satisfactory', 'complete', 'done', 'finished']
            rejection_keywords = ['reject', 'needs improvement', 'revise', 'redo', 'not good', 'insufficient']
            
            found_approval = [kw for kw in approval_keywords if kw in review_text]
            found_rejection = [kw for kw in rejection_keywords if kw in review_text]
            
            print(f"[MANAGER_WORKER_LOOP] Found approval keywords: {found_approval}")
            print(f"[MANAGER_WORKER_LOOP] Found rejection keywords: {found_rejection}")
            
            if found_approval:
                quality_approved = True
                print(f"[MANAGER_WORKER_LOOP] ✅ WORK APPROVED on iteration {self.current_iteration}")
            elif found_rejection or 'feedback:' in review_text:
                # Extract feedback for next iteration
                feedback = review_result.get('result', '')
                self.feedback_history.append(feedback)
                quality_approved = False
                print(f"[MANAGER_WORKER_LOOP] ❌ WORK REJECTED, feedback provided")
                print(f"[MANAGER_WORKER_LOOP] Feedback: {feedback}")
            else:
                # If unclear, treat as approved to avoid infinite loops
                quality_approved = True
                print(f"[MANAGER_WORKER_LOOP] ⚠️ UNCLEAR RESPONSE, treating as APPROVED")
            
            iteration_end = time.time()
            iteration_duration = iteration_end - iteration_start
            total_duration = iteration_end - start_time
            
            print(f"[MANAGER_WORKER_LOOP] Iteration {self.current_iteration} completed in {iteration_duration:.2f} seconds")
            print(f"[MANAGER_WORKER_LOOP] Total elapsed time: {total_duration:.2f} seconds")
            print(f"[MANAGER_WORKER_LOOP] Quality approved: {quality_approved}")
            
            if quality_approved:
                print(f"[MANAGER_WORKER_LOOP] 🎉 QUALITY CONTROL LOOP COMPLETED SUCCESSFULLY")
                break
            elif self.current_iteration >= self.max_iterations:
                print(f"[MANAGER_WORKER_LOOP] ⏰ MAX ITERATIONS REACHED")
                break
            else:
                print(f"[MANAGER_WORKER_LOOP] 🔄 CONTINUING TO NEXT ITERATION...")
        
        final_duration = time.time() - start_time
        print(f"[MANAGER_WORKER_LOOP] ========== QUALITY CONTROL LOOP FINISHED ==========")
        print(f"[MANAGER_WORKER_LOOP] Total duration: {final_duration:.2f} seconds")
        print(f"[MANAGER_WORKER_LOOP] Final iterations: {self.current_iteration}")
        print(f"[MANAGER_WORKER_LOOP] Final quality approved: {quality_approved}")
        print(f"[MANAGER_WORKER_LOOP] Feedback history length: {len(self.feedback_history)}")
        
        return {
            'final_result': work_result,
            'quality_approved': quality_approved,
            'iterations': self.current_iteration,
            'max_iterations': self.max_iterations,
            'feedback_history': self.feedback_history,
            'delegation_instructions': delegation_result,
            'status': 'completed' if quality_approved else 'max_iterations_reached'
        }

class VisualFlowExecutor:
    """Converts visual flows to executable agent chains."""
    
    def __init__(self, agents: Dict[str, Any]):
        print(f"[VISUAL_EXECUTOR] Initializing VisualFlowExecutor with agents: {list(agents.keys())}")
        self.agents = agents
        self.chain_system = AgentChain(agents)
        self.quality_gates = []
        print(f"[VISUAL_EXECUTOR] VisualFlowExecutor initialized successfully")
    
    def detect_manager_worker_patterns(self, visual_blocks: List[VisualBlock], 
                                     visual_connections: List[VisualConnection]) -> List[QualityGate]:
        """Detect manager-worker feedback loop patterns."""
        print(f"[VISUAL_EXECUTOR] Starting manager-worker pattern detection")
        print(f"[VISUAL_EXECUTOR] Input: {len(visual_blocks)} blocks, {len(visual_connections)} connections")
        
        quality_gates = []
        
        # Find feedback connections
        feedback_connections = [conn for conn in visual_connections if conn.type == 'feedback']
        print(f"[VISUAL_EXECUTOR] Found {len(feedback_connections)} feedback connections")
        
        for fb_conn in feedback_connections:
            print(f"[VISUAL_EXECUTOR] Processing feedback connection: {fb_conn.from_block} -> {fb_conn.to_block}")
            
            # Find the blocks involved
            from_block = next((b for b in visual_blocks if b.id == fb_conn.from_block), None)
            to_block = next((b for b in visual_blocks if b.id == fb_conn.to_block), None)
            
            if not from_block or not to_block:
                print(f"[VISUAL_EXECUTOR] ERROR: Missing blocks for connection {fb_conn.id}")
                continue
                
            print(f"[VISUAL_EXECUTOR] From block: {from_block.id} ({from_block.type})")
            print(f"[VISUAL_EXECUTOR] To block: {to_block.id} ({to_block.type})")
            
            # Check if this looks like a manager-worker pattern
            if from_block.type == 'agent' and to_block.type == 'agent':
                from_role = from_block.config.get('role', '').lower()
                to_role = to_block.config.get('role', '').lower()
                
                print(f"[VISUAL_EXECUTOR] Checking roles: from='{from_role}', to='{to_role}'")
                
                # Manager keywords
                manager_keywords = ['manager', 'supervisor', 'lead', 'reviewer', 'coordinator']
                worker_keywords = ['worker', 'developer', 'assistant', 'creator', 'executor']
                
                is_manager_worker = (
                    any(keyword in from_role for keyword in manager_keywords) and
                    any(keyword in to_role for keyword in worker_keywords)
                )
                
                print(f"[VISUAL_EXECUTOR] Is manager-worker pattern: {is_manager_worker}")
                
                if is_manager_worker:
                    quality_gate = QualityGate(
                        manager_block=from_block.id,
                        worker_block=to_block.id,
                        max_iterations=3,
                        quality_criteria=f"Quality review by {from_role} for {to_role} work"
                    )
                    quality_gates.append(quality_gate)
                    print(f"[VISUAL_EXECUTOR] Created quality gate: {quality_gate.manager_block} -> {quality_gate.worker_block}")
        
        print(f"[VISUAL_EXECUTOR] Detected {len(quality_gates)} quality gates")
        self.quality_gates = quality_gates
        return quality_gates
        
        for fb_conn in feedback_connections:
            # Get the blocks involved
            from_block = next((b for b in visual_blocks if b.id == fb_conn.from_block), None)
            to_block = next((b for b in visual_blocks if b.id == fb_conn.to_block), None)
            
            if not from_block or not to_block:
                continue
                
            # Check if this looks like a manager-worker pattern
            from_role = from_block.config.get('role', '').lower()
            to_role = to_block.config.get('role', '').lower()
            
            # Detect manager-worker patterns
            manager_keywords = ['manager', 'supervisor', 'reviewer', 'lead', 'senior']
            worker_keywords = ['worker', 'developer', 'analyst', 'assistant', 'junior']
            
            is_manager_to_worker = (
                any(keyword in from_role for keyword in manager_keywords) and
                any(keyword in to_role for keyword in worker_keywords)
            )
            
            if is_manager_to_worker:
                # Extract quality criteria from manager block
                manager_task = from_block.config.get('task', '')
                criteria = f"Quality standards based on: {manager_task}"
                
                quality_gate = QualityGate(
                    manager_block=fb_conn.from_block,
                    worker_block=fb_conn.to_block,
                    max_iterations=3,  # Default to 3 iterations
                    quality_criteria=criteria
                )
                
                quality_gates.append(quality_gate)
        
        return quality_gates
    
    def convert_visual_flow_to_chain(self, visual_blocks: List[VisualBlock], 
                                   visual_connections: List[VisualConnection]) -> str:
        """Convert a visual flow to an executable agent chain."""
        
        # Detect quality gates first
        self.quality_gates = self.detect_manager_worker_patterns(visual_blocks, visual_connections)
        
        # Generate unique chain ID
        chain_id = f"visual_flow_{len(visual_blocks)}_{hash(str(visual_blocks))}"
        
        # Convert visual blocks to chain blocks
        chain_blocks = []
        feedback_loops = []
        
        # Create mapping of visual blocks
        block_map = {block.id: block for block in visual_blocks}
        
        # Convert each visual block to chain block
        for visual_block in visual_blocks:
            if visual_block.type in ['start', 'end']:
                continue  # Skip start/end blocks for chain execution
                
            chain_block = self._convert_visual_block_to_chain_block(
                visual_block, visual_connections, block_map
            )
            if chain_block:
                chain_blocks.append(chain_block)
        
        # Create enhanced feedback loops for manager-worker patterns
        for quality_gate in self.quality_gates:
            feedback_loop = self._create_enhanced_feedback_loop(quality_gate, block_map)
            if feedback_loop:
                feedback_loops.append(feedback_loop)
        
        # Process remaining feedback connections (non-manager-worker)
        feedback_connections = [conn for conn in visual_connections if conn.type == 'feedback']
        for fb_conn in feedback_connections:
            # Skip if already handled by quality gates
            if any(qg.manager_block == fb_conn.from_block and qg.worker_block == fb_conn.to_block 
                   for qg in self.quality_gates):
                continue
                
            feedback_loop = self._create_feedback_loop_from_connection(fb_conn, block_map)
            if feedback_loop:
                feedback_loops.append(feedback_loop)
        
        # Create the chain
        self.chain_system.create_chain(chain_id, chain_blocks, feedback_loops)
        
        return chain_id
    
    def _create_enhanced_feedback_loop(self, quality_gate: QualityGate, 
                                     block_map: Dict[str, VisualBlock]) -> Optional[FeedbackLoop]:
        """Create an enhanced feedback loop for manager-worker patterns."""
        
        manager_block = block_map.get(quality_gate.manager_block)
        worker_block = block_map.get(quality_gate.worker_block)
        
        if not manager_block or not worker_block:
            return None
        
        # Get manager agent role
        manager_agent = manager_block.config.get('agent', 'Manager')
        
        return FeedbackLoop(
            id=f"quality_gate_{quality_gate.manager_block}_{quality_gate.worker_block}",
            type=FeedbackType.REVIEW,
            reviewer_role=manager_agent,
            target_block=quality_gate.worker_block,
            criteria=quality_gate.quality_criteria,
            max_iterations=quality_gate.max_iterations,
            improvement_prompt="Based on manager feedback, improve the work to meet quality standards"
        )
    
    def _convert_visual_block_to_chain_block(self, visual_block: VisualBlock, 
                                           connections: List[VisualConnection],
                                           block_map: Dict[str, VisualBlock]) -> Optional[ChainBlock]:
        """Convert a visual block to a chain block."""
        
        # Find next blocks (forward connections)
        next_blocks = []
        for conn in connections:
            if conn.from_block == visual_block.id and conn.type == 'forward':
                # Skip start/end blocks in next_blocks
                target_block = block_map.get(conn.to_block)
                if target_block and target_block.type not in ['start', 'end']:
                    next_blocks.append(conn.to_block)
        
        # Convert block type
        block_type = self._get_chain_block_type(visual_block.type)
        if not block_type:
            return None
        
        # Get agent role for agent blocks
        agent_role = None
        if visual_block.type == 'agent':
            agent_role = visual_block.config.get('agent', 'Developer')
        
        return ChainBlock(
            id=visual_block.id,
            type=block_type,
            agent_role=agent_role,
            config=visual_block.config,
            next_blocks=next_blocks,
            condition=visual_block.config.get('condition') if visual_block.type == 'condition' else None
        )
    
    def _get_chain_block_type(self, visual_type: str) -> Optional[BlockType]:
        """Convert visual block type to chain block type."""
        type_mapping = {
            'agent': BlockType.AGENT,
            'condition': BlockType.CONDITION,
            'merge': BlockType.MERGE,
            'split': BlockType.SPLIT
        }
        return type_mapping.get(visual_type)
    
    def _create_feedback_loop_from_connection(self, connection: VisualConnection,
                                            block_map: Dict[str, VisualBlock]) -> Optional[FeedbackLoop]:
        """Create a feedback loop from a feedback connection."""
        
        from_block = block_map.get(connection.from_block)
        to_block = block_map.get(connection.to_block)
        
        if not from_block or not to_block or from_block.type != 'agent':
            return None
        
        # Use the agent from the source block as reviewer
        reviewer_role = from_block.config.get('agent', 'Manager')
        
        # Generate feedback criteria based on the target block
        criteria = self._generate_feedback_criteria(to_block)
        
        return FeedbackLoop(
            id=f"feedback_{connection.id}",
            type=FeedbackType.REVIEW,
            reviewer_role=reviewer_role,
            target_block=connection.to_block,
            criteria=criteria,
            max_iterations=2,
            improvement_prompt=f"Improve based on {reviewer_role} feedback"
        )
    
    def _generate_feedback_criteria(self, target_block: VisualBlock) -> str:
        """Generate feedback criteria based on target block configuration."""
        
        if target_block.type == 'agent':
            task = target_block.config.get('task', 'Process the input')
            role = target_block.config.get('role', 'Assistant')
            return f"Evaluate the quality and accuracy of the {role}'s work on: {task}"
        
        return "Evaluate the quality and accuracy of the output"
    
    async def execute_visual_flow(self, visual_blocks: List[VisualBlock], 
                                visual_connections: List[VisualConnection],
                                initial_task: Task) -> Dict[str, Any]:
        """Execute a visual flow with enhanced manager-worker support."""
        print(f"[VISUAL_EXECUTOR] Starting execute_visual_flow")
        print(f"[VISUAL_EXECUTOR] Blocks: {[b.id for b in visual_blocks]}")
        print(f"[VISUAL_EXECUTOR] Connections: {[(c.from_block, c.to_block, c.type) for c in visual_connections]}")
        print(f"[VISUAL_EXECUTOR] Task: {initial_task.type} - {initial_task.content[:100]}...")
        
        try:
            # Detect manager-worker patterns
            print(f"[VISUAL_EXECUTOR] Step 1: Detecting manager-worker patterns")
            quality_gates = self.detect_manager_worker_patterns(visual_blocks, visual_connections)
            
            # If we have manager-worker patterns, use specialized execution
            if quality_gates:
                print(f"[VISUAL_EXECUTOR] Step 2: Using manager-worker execution path")
                result = await self._execute_manager_worker_flow(
                    visual_blocks, visual_connections, initial_task, quality_gates
                )
                print(f"[VISUAL_EXECUTOR] Manager-worker execution completed")
                return result
            
            # Otherwise, use standard chain execution
            print(f"[VISUAL_EXECUTOR] Step 2: Using standard chain execution path")
            chain_id = self.convert_visual_flow_to_chain(visual_blocks, visual_connections)
            print(f"[VISUAL_EXECUTOR] Created chain with ID: {chain_id}")
            
            result = await self.chain_system.execute_chain(chain_id, initial_task)
            print(f"[VISUAL_EXECUTOR] Chain execution completed")
            
            # Add visual flow metadata
            result['visual_flow'] = {
                'blocks': len(visual_blocks),
                'connections': len(visual_connections),
                'chain_id': chain_id,
                'execution_type': 'standard'
            }
            
            print(f"[VISUAL_EXECUTOR] Final result prepared with metadata")
            return result
            
        except Exception as e:
            print(f"[VISUAL_EXECUTOR] ERROR in execute_visual_flow: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _execute_manager_worker_flow(self, visual_blocks: List[VisualBlock],
                                         visual_connections: List[VisualConnection],
                                         initial_task: Task,
                                         quality_gates: List[QualityGate]) -> Dict[str, Any]:
        """Execute a flow with manager-worker quality control."""
        
        results = {}
        execution_history = []
        
        # Find the flow entry point (start block or first agent)
        start_blocks = [b for b in visual_blocks if b.type == 'start']
        if start_blocks:
            current_block_id = self._find_next_agent_block(start_blocks[0].id, visual_connections, visual_blocks)
        else:
            # Find first agent block
            agent_blocks = [b for b in visual_blocks if b.type == 'agent']
            current_block_id = agent_blocks[0].id if agent_blocks else None
        
        if not current_block_id:
            return {'error': 'No agent blocks found in flow'}
        
        # Execute the flow
        current_task = initial_task
        processed_blocks = set()
        
        while current_block_id and current_block_id not in processed_blocks:
            current_block = next((b for b in visual_blocks if b.id == current_block_id), None)
            if not current_block or current_block.type != 'agent':
                break
            
            processed_blocks.add(current_block_id)
            
            # Check if this block is part of a manager-worker pattern
            quality_gate = next((qg for qg in quality_gates if qg.worker_block == current_block_id), None)
            
            if quality_gate:
                # Execute with manager-worker quality control
                manager_block = next((b for b in visual_blocks if b.id == quality_gate.manager_block), None)
                
                if manager_block:
                    print(f"[VISUAL_EXECUTOR] Manager block config: {manager_block.config}")
                    print(f"[VISUAL_EXECUTOR] Worker block config: {current_block.config}")
                    
                    # Get agent based on model selection, not agent type
                    manager_model = manager_block.config.get('model', 'gpt-4o')
                    worker_model = current_block.config.get('model', 'gpt-4o')
                    
                    print(f"[VISUAL_EXECUTOR] Manager model: {manager_model}")
                    print(f"[VISUAL_EXECUTOR] Worker model: {worker_model}")
                    
                    # Map models to agents
                    manager_agent = self._get_agent_by_model(manager_model)
                    worker_agent = self._get_agent_by_model(worker_model)
                    
                    print(f"[VISUAL_EXECUTOR] Manager agent: {type(manager_agent).__name__ if manager_agent else 'None'}")
                    print(f"[VISUAL_EXECUTOR] Worker agent: {type(worker_agent).__name__ if worker_agent else 'None'}")
                    
                    if manager_agent and worker_agent:
                        loop_executor = ManagerWorkerLoop(
                            manager_agent, worker_agent, quality_gate.max_iterations
                        )
                        
                        loop_result = await loop_executor.execute_with_quality_control(current_task)
                        results[current_block_id] = loop_result
                        
                        execution_history.append({
                            'block_id': current_block_id,
                            'type': 'manager_worker_loop',
                            'iterations': loop_result['iterations'],
                            'quality_approved': loop_result['quality_approved'],
                            'feedback_count': len(loop_result['feedback_history'])
                        })
                        
                        # Update task for next block
                        if loop_result['final_result']:
                            current_task = Task(
                                type=current_task.type,
                                content=loop_result['final_result'].get('result', ''),
                                role=current_task.role
                            )
            else:
                # Standard agent execution
                print(f"[VISUAL_EXECUTOR] Standard execution for block: {current_block.id}")
                print(f"[VISUAL_EXECUTOR] Block config: {current_block.config}")
                
                # Get agent based on model selection, not agent type
                model = current_block.config.get('model', 'gpt-4o')
                print(f"[VISUAL_EXECUTOR] Selected model: {model}")
                
                agent = self._get_agent_by_model(model)
                print(f"[VISUAL_EXECUTOR] Selected agent: {type(agent).__name__ if agent else 'None'}")
                
                if agent:
                    block_task = Task(
                        type=current_task.type,
                        content=current_task.content,
                        role=current_block.config.get('role', 'Assistant')
                    )
                    
                    result = await agent.process_task(block_task)
                    results[current_block_id] = result
                    
                    execution_history.append({
                        'block_id': current_block_id,
                        'type': 'standard_agent',
                        'result': result
                    })
                    
                    # Update task for next block
                    current_task = Task(
                        type=current_task.type,
                        content=result.get('result', ''),
                        role=current_task.role
                    )
            
            # Find next block
            current_block_id = self._find_next_agent_block(current_block_id, visual_connections, visual_blocks)
        
        return {
            'final_result': results,
            'execution_history': execution_history,
            'quality_gates': len(quality_gates),
            'visual_flow': {
                'blocks': len(visual_blocks),
                'connections': len(visual_connections),
                'execution_type': 'manager_worker_enhanced'
            }
        }
    
    def _get_agent_by_model(self, model: str):
        """Get the appropriate agent based on the selected model."""
        print(f"[VISUAL_EXECUTOR] Mapping model '{model}' to agent")
        
        # Map models to agents
        model_to_agent = {
            'gpt-4o': self.agents.get('Developer'),  # AgentGPT4o
            'claude-3': self.agents.get('Developer'),  # Use GPT4o as fallback for Claude
            'grok': self.agents.get('Manager'),  # AgentGrok4
            'o3-mini': self.agents.get('CodeVerifier'),  # AgentO3Pro
            'openmanus': self.agents.get('OpenManus'),  # AgentOpenManus
        }
        
        # Try exact match first
        agent = model_to_agent.get(model)
        if agent:
            print(f"[VISUAL_EXECUTOR] Found exact match: {model} -> {type(agent).__name__}")
            return agent
        
        # Try partial matches
        model_lower = model.lower()
        if 'gpt' in model_lower or '4o' in model_lower:
            agent = self.agents.get('Developer')
            print(f"[VISUAL_EXECUTOR] GPT model detected: {model} -> {type(agent).__name__ if agent else 'None'}")
            return agent
        elif 'claude' in model_lower:
            agent = self.agents.get('Developer')  # Use GPT4o as fallback
            print(f"[VISUAL_EXECUTOR] Claude model detected: {model} -> {type(agent).__name__ if agent else 'None'}")
            return agent
        elif 'grok' in model_lower:
            agent = self.agents.get('Manager')
            print(f"[VISUAL_EXECUTOR] Grok model detected: {model} -> {type(agent).__name__ if agent else 'None'}")
            return agent
        elif 'o3' in model_lower:
            agent = self.agents.get('CodeVerifier')
            print(f"[VISUAL_EXECUTOR] O3 model detected: {model} -> {type(agent).__name__ if agent else 'None'}")
            return agent
        elif 'manus' in model_lower:
            agent = self.agents.get('OpenManus')
            print(f"[VISUAL_EXECUTOR] OpenManus model detected: {model} -> {type(agent).__name__ if agent else 'None'}")
            return agent
        
        # Default fallback to GPT4o
        agent = self.agents.get('Developer')
        print(f"[VISUAL_EXECUTOR] Using default fallback: {model} -> {type(agent).__name__ if agent else 'None'}")
        return agent
    
    def _find_next_agent_block(self, current_block_id: str, 
                             connections: List[VisualConnection],
                             blocks: List[VisualBlock]) -> Optional[str]:
        """Find the next agent block in the flow."""
        
        # Find forward connections from current block
        next_connections = [c for c in connections if c.from_block == current_block_id and c.type == 'forward']
        
        for conn in next_connections:
            next_block = next((b for b in blocks if b.id == conn.to_block), None)
            if next_block:
                if next_block.type == 'agent':
                    return next_block.id
                elif next_block.type == 'end':
                    return None
                else:
                    # Continue searching from this block
                    return self._find_next_agent_block(next_block.id, connections, blocks)
        
        return None
    
    def validate_visual_flow(self, visual_blocks: List[VisualBlock], 
                           visual_connections: List[VisualConnection]) -> Dict[str, Any]:
        """Validate a visual flow for common issues."""
        
        errors = []
        warnings = []
        
        # Check for start and end blocks
        start_blocks = [b for b in visual_blocks if b.type == 'start']
        end_blocks = [b for b in visual_blocks if b.type == 'end']
        
        if len(start_blocks) == 0:
            warnings.append("No start block found. Flow will start from the first available block.")
        elif len(start_blocks) > 1:
            warnings.append("Multiple start blocks found. Only one will be used.")
        
        if len(end_blocks) == 0:
            warnings.append("No end block found. Flow will end when no more blocks to process.")
        
        # Check for disconnected blocks
        connected_blocks = set()
        for conn in visual_connections:
            connected_blocks.add(conn.from_block)
            connected_blocks.add(conn.to_block)
        
        disconnected = [b.id for b in visual_blocks if b.id not in connected_blocks and b.type not in ['start', 'end']]
        if disconnected:
            warnings.append(f"Disconnected blocks found: {', '.join(disconnected)}")
        
        # Check agent blocks configuration
        for block in visual_blocks:
            if block.type == 'agent':
                if not block.config.get('agent'):
                    errors.append(f"Agent block '{block.id}' missing agent configuration")
                if not block.config.get('task'):
                    warnings.append(f"Agent block '{block.id}' missing task description")
        
        # Validate manager-worker patterns
        quality_gates = self.detect_manager_worker_patterns(visual_blocks, visual_connections)
        for qg in quality_gates:
            manager_block = next((b for b in visual_blocks if b.id == qg.manager_block), None)
            worker_block = next((b for b in visual_blocks if b.id == qg.worker_block), None)
            
            if manager_block and not manager_block.config.get('task'):
                warnings.append(f"Manager block '{qg.manager_block}' should have clear delegation instructions")
            
            if worker_block and not worker_block.config.get('task'):
                warnings.append(f"Worker block '{qg.worker_block}' should have clear task description")
        
        # Check for circular dependencies (basic check)
        if self._has_circular_dependency(visual_blocks, visual_connections):
            errors.append("Circular dependency detected in flow")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'quality_gates_detected': len(quality_gates),
            'manager_worker_patterns': [
                {
                    'manager': qg.manager_block,
                    'worker': qg.worker_block,
                    'max_iterations': qg.max_iterations
                }
                for qg in quality_gates
            ]
        }
    
    def _has_circular_dependency(self, blocks: List[VisualBlock], 
                               connections: List[VisualConnection]) -> bool:
        """Check for circular dependencies in the flow."""
        
        # Build adjacency list (only forward connections)
        graph = {}
        for block in blocks:
            graph[block.id] = []
        
        for conn in connections:
            if conn.type == 'forward':
                if conn.from_block in graph:
                    graph[conn.from_block].append(conn.to_block)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for block_id in graph:
            if block_id not in visited:
                if has_cycle(block_id):
                    return True
        
        return False
    
    def get_flow_statistics(self, visual_blocks: List[VisualBlock], 
                          visual_connections: List[VisualConnection]) -> Dict[str, Any]:
        """Get statistics about a visual flow."""
        
        block_types = {}
        for block in visual_blocks:
            block_types[block.type] = block_types.get(block.type, 0) + 1
        
        connection_types = {}
        for conn in visual_connections:
            connection_types[conn.type] = connection_types.get(conn.type, 0) + 1
        
        # Detect quality gates
        quality_gates = self.detect_manager_worker_patterns(visual_blocks, visual_connections)
        
        # Calculate complexity score
        complexity = len(visual_blocks) + len(visual_connections) * 0.5
        
        return {
            'total_blocks': len(visual_blocks),
            'total_connections': len(visual_connections),
            'block_types': block_types,
            'connection_types': connection_types,
            'complexity_score': complexity,
            'has_feedback_loops': connection_types.get('feedback', 0) > 0,
            'quality_gates': len(quality_gates),
            'manager_worker_patterns': len(quality_gates),
            'estimated_max_iterations': sum(qg.max_iterations for qg in quality_gates)
        }

