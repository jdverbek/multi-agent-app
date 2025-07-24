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
        
    async def execute_with_quality_control(self, task: Task) -> Dict[str, Any]:
        """Execute task with manager-worker quality control loop."""
        
        # Manager analyzes and delegates the task
        delegation_task = Task(
            type="delegation",
            content=f"Analyze this request and provide clear instructions for a worker: {task.content}",
            role="Manager"
        )
        
        delegation_result = await self.manager_agent.process_task(delegation_task)
        
        # Initialize work result
        work_result = None
        quality_approved = False
        
        # Quality control loop
        while not quality_approved and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            # Worker executes the task
            if self.current_iteration == 1:
                # First iteration - use original task with manager's instructions
                worker_task = Task(
                    type=task.type,
                    content=f"Manager Instructions: {delegation_result.get('result', '')}\n\nOriginal Request: {task.content}",
                    role="Worker"
                )
            else:
                # Subsequent iterations - include feedback
                last_feedback = self.feedback_history[-1] if self.feedback_history else ""
                worker_task = Task(
                    type=task.type,
                    content=f"Manager Instructions: {delegation_result.get('result', '')}\n\nOriginal Request: {task.content}\n\nPrevious Work: {work_result.get('result', '')}\n\nManager Feedback: {last_feedback}",
                    role="Worker"
                )
            
            work_result = await self.worker_agent.process_task(worker_task)
            
            # Manager reviews the work
            review_task = Task(
                type="quality_review",
                content=f"Review this work and determine if it meets quality standards:\n\nOriginal Request: {task.content}\n\nWorker's Output: {work_result.get('result', '')}\n\nIteration: {self.current_iteration}/{self.max_iterations}",
                role="Manager"
            )
            
            review_result = await self.manager_agent.process_task(review_task)
            
            # Parse manager's decision
            review_text = review_result.get('result', '').lower()
            
            # Check for approval keywords
            approval_keywords = ['approved', 'accept', 'good', 'satisfactory', 'complete', 'done', 'finished']
            rejection_keywords = ['reject', 'needs improvement', 'revise', 'redo', 'not good', 'insufficient']
            
            if any(keyword in review_text for keyword in approval_keywords):
                quality_approved = True
            elif any(keyword in review_text for keyword in rejection_keywords) or 'feedback:' in review_text:
                # Extract feedback for next iteration
                feedback = review_result.get('result', '')
                self.feedback_history.append(feedback)
                quality_approved = False
            else:
                # If unclear, ask manager to be more explicit
                clarification_task = Task(
                    type="clarification",
                    content=f"Please provide a clear APPROVED or NEEDS IMPROVEMENT decision for this work:\n{work_result.get('result', '')}\n\nIf needs improvement, provide specific feedback.",
                    role="Manager"
                )
                
                clarification_result = await self.manager_agent.process_task(clarification_task)
                clarification_text = clarification_result.get('result', '').lower()
                
                if 'approved' in clarification_text:
                    quality_approved = True
                else:
                    feedback = clarification_result.get('result', '')
                    self.feedback_history.append(feedback)
                    quality_approved = False
        
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
        self.agents = agents
        self.chain_system = AgentChain(agents)
        self.quality_gates = []
    
    def detect_manager_worker_patterns(self, visual_blocks: List[VisualBlock], 
                                     visual_connections: List[VisualConnection]) -> List[QualityGate]:
        """Detect manager-worker feedback loop patterns."""
        
        quality_gates = []
        
        # Find feedback connections
        feedback_connections = [conn for conn in visual_connections if conn.type == 'feedback']
        
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
        
        # Detect manager-worker patterns
        quality_gates = self.detect_manager_worker_patterns(visual_blocks, visual_connections)
        
        # If we have manager-worker patterns, use specialized execution
        if quality_gates:
            return await self._execute_manager_worker_flow(
                visual_blocks, visual_connections, initial_task, quality_gates
            )
        
        # Otherwise, use standard chain execution
        chain_id = self.convert_visual_flow_to_chain(visual_blocks, visual_connections)
        result = await self.chain_system.execute_chain(chain_id, initial_task)
        
        # Add visual flow metadata
        result['visual_flow'] = {
            'blocks': len(visual_blocks),
            'connections': len(visual_connections),
            'chain_id': chain_id,
            'execution_type': 'standard'
        }
        
        return result
    
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
                    manager_agent = self.agents.get(manager_block.config.get('agent', 'Manager'))
                    worker_agent = self.agents.get(current_block.config.get('agent', 'Developer'))
                    
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
                agent = self.agents.get(current_block.config.get('agent', 'Developer'))
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

