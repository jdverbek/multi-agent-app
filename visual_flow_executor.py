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

class VisualFlowExecutor:
    """Converts visual flows to executable agent chains."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.chain_system = AgentChain(agents)
    
    def convert_visual_flow_to_chain(self, visual_blocks: List[VisualBlock], 
                                   visual_connections: List[VisualConnection]) -> str:
        """Convert a visual flow to an executable agent chain."""
        
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
        
        # Process feedback connections
        feedback_connections = [conn for conn in visual_connections if conn.type == 'feedback']
        for fb_conn in feedback_connections:
            feedback_loop = self._create_feedback_loop_from_connection(
                fb_conn, block_map
            )
            if feedback_loop:
                feedback_loops.append(feedback_loop)
        
        # Create the chain
        self.chain_system.create_chain(chain_id, chain_blocks, feedback_loops)
        
        return chain_id
    
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
        """Execute a visual flow."""
        
        # Convert to chain
        chain_id = self.convert_visual_flow_to_chain(visual_blocks, visual_connections)
        
        # Execute the chain
        result = await self.chain_system.execute_chain(chain_id, initial_task)
        
        # Add visual flow metadata
        result['visual_flow'] = {
            'blocks': len(visual_blocks),
            'connections': len(visual_connections),
            'chain_id': chain_id
        }
        
        return result
    
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
        
        # Check for circular dependencies (basic check)
        if self._has_circular_dependency(visual_blocks, visual_connections):
            errors.append("Circular dependency detected in flow")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
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
        
        # Calculate complexity score
        complexity = len(visual_blocks) + len(visual_connections) * 0.5
        
        return {
            'total_blocks': len(visual_blocks),
            'total_connections': len(visual_connections),
            'block_types': block_types,
            'connection_types': connection_types,
            'complexity_score': complexity,
            'has_feedback_loops': connection_types.get('feedback', 0) > 0
        }

