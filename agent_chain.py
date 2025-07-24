import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from tasks import Task

class BlockType(Enum):
    """Types of blocks in the agent chain."""
    AGENT = "agent"
    CONDITION = "condition"
    FEEDBACK = "feedback"
    MERGE = "merge"
    SPLIT = "split"

class FeedbackType(Enum):
    """Types of feedback loops."""
    REVIEW = "review"
    REFINE = "refine"
    VALIDATE = "validate"
    ITERATE = "iterate"

@dataclass
class ChainBlock:
    """Represents a block in the agent chain."""
    id: str
    type: BlockType
    agent_role: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    next_blocks: List[str] = field(default_factory=list)
    feedback_blocks: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    max_iterations: int = 3
    
@dataclass
class FeedbackLoop:
    """Represents a feedback loop configuration."""
    id: str
    type: FeedbackType
    reviewer_role: str
    target_block: str
    criteria: str
    max_iterations: int = 3
    improvement_prompt: str = ""

@dataclass
class ChainExecution:
    """Tracks the execution state of a chain."""
    chain_id: str
    current_block: str
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    feedback_iterations: Dict[str, int] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False

class AgentChain:
    """Manages agent chains with customizable blocks and feedback loops."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.chains: Dict[str, List[ChainBlock]] = {}
        self.feedback_loops: Dict[str, List[FeedbackLoop]] = {}
        
    def create_chain(self, chain_id: str, blocks: List[ChainBlock], feedback_loops: List[FeedbackLoop] = None):
        """Create a new agent chain."""
        self.chains[chain_id] = blocks
        self.feedback_loops[chain_id] = feedback_loops or []
        
    def add_block(self, chain_id: str, block: ChainBlock):
        """Add a block to an existing chain."""
        if chain_id not in self.chains:
            self.chains[chain_id] = []
        self.chains[chain_id].append(block)
        
    def add_feedback_loop(self, chain_id: str, feedback_loop: FeedbackLoop):
        """Add a feedback loop to a chain."""
        if chain_id not in self.feedback_loops:
            self.feedback_loops[chain_id] = []
        self.feedback_loops[chain_id].append(feedback_loop)
        
    async def execute_chain(self, chain_id: str, initial_task: Task) -> Dict[str, Any]:
        """Execute an agent chain with feedback loops."""
        if chain_id not in self.chains:
            raise ValueError(f"Chain {chain_id} not found")
            
        execution = ChainExecution(chain_id=chain_id, current_block="start")
        blocks = {block.id: block for block in self.chains[chain_id]}
        feedback_loops = {loop.id: loop for loop in self.feedback_loops[chain_id]}
        
        # Find the starting block
        start_blocks = [block for block in self.chains[chain_id] if not any(
            block.id in other.next_blocks for other in self.chains[chain_id]
        )]
        
        if not start_blocks:
            raise ValueError("No starting block found in chain")
            
        current_task = initial_task
        current_blocks = [start_blocks[0].id]
        
        while current_blocks and not execution.completed:
            next_blocks = []
            
            for block_id in current_blocks:
                if block_id not in blocks:
                    continue
                    
                block = blocks[block_id]
                execution.current_block = block_id
                
                # Execute the block
                result = await self._execute_block(block, current_task, execution)
                execution.results[block_id] = result
                
                # Check for feedback loops
                feedback_result = await self._process_feedback_loops(
                    block, result, current_task, feedback_loops, execution
                )
                
                if feedback_result:
                    execution.results[block_id] = feedback_result
                    result = feedback_result
                
                # Update task for next blocks
                if result and isinstance(result, str):
                    current_task = Task(
                        type=current_task.type,
                        content=result,
                        role=current_task.role
                    )
                
                # Add next blocks to execution queue
                next_blocks.extend(block.next_blocks)
                
                # Log execution
                execution.execution_history.append({
                    "block_id": block_id,
                    "block_type": block.type.value,
                    "agent_role": block.agent_role,
                    "result": result,
                    "timestamp": asyncio.get_event_loop().time()
                })
            
            current_blocks = next_blocks
            
            # Check if we've reached the end
            if not current_blocks:
                execution.completed = True
        
        return {
            "chain_id": chain_id,
            "execution": execution,
            "final_result": execution.results,
            "history": execution.execution_history
        }
    
    async def _execute_block(self, block: ChainBlock, task: Task, execution: ChainExecution) -> Any:
        """Execute a single block in the chain."""
        if block.type == BlockType.AGENT:
            if block.agent_role and block.agent_role in self.agents:
                agent = self.agents[block.agent_role]
                
                # Create task with specific role
                agent_task = Task(
                    type=task.type,
                    content=task.content,
                    role=block.agent_role
                )
                
                return await agent.handle(agent_task)
            else:
                return f"Agent role '{block.agent_role}' not found"
                
        elif block.type == BlockType.CONDITION:
            # Simple condition evaluation (can be extended)
            condition = block.condition or "true"
            if "length" in condition:
                return len(task.content) > 100
            return True
            
        elif block.type == BlockType.MERGE:
            # Merge results from previous blocks
            previous_results = []
            for prev_block_id in execution.results:
                if prev_block_id in execution.results:
                    previous_results.append(execution.results[prev_block_id])
            return " ".join(str(r) for r in previous_results)
            
        elif block.type == BlockType.SPLIT:
            # Split task into multiple parts (simplified)
            parts = task.content.split('\n')
            return parts
            
        return "Block type not implemented"
    
    async def _process_feedback_loops(self, block: ChainBlock, result: Any, 
                                    original_task: Task, feedback_loops: Dict[str, FeedbackLoop], 
                                    execution: ChainExecution) -> Optional[Any]:
        """Process feedback loops for a block."""
        improved_result = result
        
        for loop_id, feedback_loop in feedback_loops.items():
            if feedback_loop.target_block != block.id:
                continue
                
            # Check iteration limit
            current_iterations = execution.feedback_iterations.get(loop_id, 0)
            if current_iterations >= feedback_loop.max_iterations:
                continue
                
            # Get reviewer agent
            if feedback_loop.reviewer_role not in self.agents:
                continue
                
            reviewer = self.agents[feedback_loop.reviewer_role]
            
            # Create feedback task
            feedback_prompt = f"""
            Please review the following result and provide feedback based on these criteria: {feedback_loop.criteria}
            
            Original task: {original_task.content}
            Current result: {improved_result}
            
            Feedback type: {feedback_loop.type.value}
            {feedback_loop.improvement_prompt}
            
            If the result needs improvement, provide specific suggestions. 
            If it's satisfactory, respond with "APPROVED".
            """
            
            feedback_task = Task(
                type="review",
                content=feedback_prompt,
                role=feedback_loop.reviewer_role
            )
            
            feedback_response = await reviewer.handle(feedback_task)
            
            # Process feedback
            if "APPROVED" not in feedback_response.upper():
                # Need improvement - create refinement task
                refinement_prompt = f"""
                Please improve the following result based on this feedback:
                
                Original task: {original_task.content}
                Current result: {improved_result}
                Feedback: {feedback_response}
                
                Provide an improved version:
                """
                
                refinement_task = Task(
                    type=original_task.type,
                    content=refinement_prompt,
                    role=block.agent_role
                )
                
                if block.agent_role in self.agents:
                    improved_result = await self.agents[block.agent_role].handle(refinement_task)
                    
                # Update iteration count
                execution.feedback_iterations[loop_id] = current_iterations + 1
                
                # Log feedback iteration
                execution.execution_history.append({
                    "block_id": f"{block.id}_feedback_{loop_id}",
                    "block_type": "feedback",
                    "agent_role": feedback_loop.reviewer_role,
                    "result": feedback_response,
                    "iteration": current_iterations + 1,
                    "timestamp": asyncio.get_event_loop().time()
                })
        
        return improved_result if improved_result != result else None
    
    def get_chain_definition(self, chain_id: str) -> Dict[str, Any]:
        """Get the definition of a chain."""
        if chain_id not in self.chains:
            return {}
            
        return {
            "chain_id": chain_id,
            "blocks": [
                {
                    "id": block.id,
                    "type": block.type.value,
                    "agent_role": block.agent_role,
                    "config": block.config,
                    "next_blocks": block.next_blocks,
                    "feedback_blocks": block.feedback_blocks,
                    "condition": block.condition,
                    "max_iterations": block.max_iterations
                }
                for block in self.chains[chain_id]
            ],
            "feedback_loops": [
                {
                    "id": loop.id,
                    "type": loop.type.value,
                    "reviewer_role": loop.reviewer_role,
                    "target_block": loop.target_block,
                    "criteria": loop.criteria,
                    "max_iterations": loop.max_iterations,
                    "improvement_prompt": loop.improvement_prompt
                }
                for loop in self.feedback_loops[chain_id]
            ]
        }
    
    def list_chains(self) -> List[str]:
        """List all available chains."""
        return list(self.chains.keys())

