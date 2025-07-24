import asyncio
from typing import Dict, List

from tasks import Task
from agent_grok4 import AgentGrok4
from agent_gpt4o import AgentGPT4o
from agent_o3pro import AgentO3Pro
from agent_openmanus import AgentOpenManus
from agent_chain import AgentChain, ChainBlock, FeedbackLoop, BlockType, FeedbackType

class MainController:
    """Central orchestrator that dispatches tasks to specialized agents and manages chains."""

    def __init__(self):
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.agents: Dict[str, object] = {
            "Manager": AgentGrok4("AgentGrok4"),
            "CodeVerifier": AgentO3Pro("AgentO3Pro"),
            "Developer": AgentGPT4o("AgentGPT4o"),
            "OpenManus": AgentOpenManus("AgentOpenManus", {
                "base_url": "http://localhost:8000",
                "model": "gpt-4o",
                "temperature": 0.3
            })
        }
        
        # Initialize agent chain system
        self.chain_system = AgentChain(self.agents)
        self._setup_default_chains()

    def _setup_default_chains(self):
        """Set up some default agent chains."""
        
        # Code Development Chain with Feedback
        code_blocks = [
            ChainBlock(
                id="generate_code",
                type=BlockType.AGENT,
                agent_role="Developer",
                next_blocks=["review_code"]
            ),
            ChainBlock(
                id="review_code",
                type=BlockType.AGENT,
                agent_role="CodeVerifier",
                next_blocks=["final_approval"]
            ),
            ChainBlock(
                id="final_approval",
                type=BlockType.AGENT,
                agent_role="Manager",
                next_blocks=[]
            )
        ]
        
        code_feedback_loops = [
            FeedbackLoop(
                id="code_review_loop",
                type=FeedbackType.REVIEW,
                reviewer_role="CodeVerifier",
                target_block="generate_code",
                criteria="Code quality, best practices, security, and functionality",
                max_iterations=2,
                improvement_prompt="Focus on code quality, security, and best practices."
            ),
            FeedbackLoop(
                id="manager_approval_loop",
                type=FeedbackType.VALIDATE,
                reviewer_role="Manager",
                target_block="review_code",
                criteria="Overall solution quality and completeness",
                max_iterations=1,
                improvement_prompt="Ensure the solution meets all requirements."
            )
        ]
        
        self.chain_system.create_chain("code_development", code_blocks, code_feedback_loops)
        
        # Analysis Chain with OpenManus
        analysis_blocks = [
            ChainBlock(
                id="initial_analysis",
                type=BlockType.AGENT,
                agent_role="OpenManus",
                next_blocks=["deep_review"]
            ),
            ChainBlock(
                id="deep_review",
                type=BlockType.AGENT,
                agent_role="Manager",
                next_blocks=["verification"]
            ),
            ChainBlock(
                id="verification",
                type=BlockType.AGENT,
                agent_role="CodeVerifier",
                next_blocks=[]
            )
        ]
        
        analysis_feedback_loops = [
            FeedbackLoop(
                id="analysis_refinement",
                type=FeedbackType.REFINE,
                reviewer_role="Manager",
                target_block="initial_analysis",
                criteria="Depth of analysis, accuracy, and completeness",
                max_iterations=2,
                improvement_prompt="Provide more detailed analysis with specific examples."
            )
        ]
        
        self.chain_system.create_chain("analysis_workflow", analysis_blocks, analysis_feedback_loops)
        
        # Simple Sequential Chain
        simple_blocks = [
            ChainBlock(
                id="process_input",
                type=BlockType.AGENT,
                agent_role="Developer",
                next_blocks=["validate_output"]
            ),
            ChainBlock(
                id="validate_output",
                type=BlockType.AGENT,
                agent_role="CodeVerifier",
                next_blocks=[]
            )
        ]
        
        self.chain_system.create_chain("simple_workflow", simple_blocks)

    async def submit_task(self, task: Task) -> None:
        await self.queue.put(task)

    async def execute_chain(self, chain_id: str, task: Task) -> Dict:
        """Execute a specific agent chain."""
        return await self.chain_system.execute_chain(chain_id, task)

    async def run(self) -> None:
        """Continuously process tasks from the queue."""
        while True:
            task = await self.queue.get()
            
            # Check if this is a chain execution request
            if hasattr(task, 'chain_id') and task.chain_id:
                result = await self.execute_chain(task.chain_id, task)
                task.response = result
            else:
                # Regular single agent processing
                agent = self._select_agent(task)
                response = await agent.handle(task)
                task.response = response
                
            self.queue.task_done()

    def _select_agent(self, task: Task):
        """Select agent based on desired role or fallback."""
        if task.role and task.role in self.agents:
            return self.agents[task.role]
        return self.agents.get("Developer")
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return list(self.agents.keys())
    
    def get_available_chains(self) -> List[str]:
        """Get list of available chains."""
        return self.chain_system.list_chains()
    
    def get_chain_definition(self, chain_id: str) -> Dict:
        """Get the definition of a specific chain."""
        return self.chain_system.get_chain_definition(chain_id)
    
    def create_custom_chain(self, chain_id: str, blocks_config: List[Dict], 
                          feedback_config: List[Dict] = None) -> bool:
        """Create a custom chain from configuration."""
        try:
            blocks = []
            for block_config in blocks_config:
                block = ChainBlock(
                    id=block_config['id'],
                    type=BlockType(block_config['type']),
                    agent_role=block_config.get('agent_role'),
                    config=block_config.get('config', {}),
                    next_blocks=block_config.get('next_blocks', []),
                    feedback_blocks=block_config.get('feedback_blocks', []),
                    condition=block_config.get('condition'),
                    max_iterations=block_config.get('max_iterations', 3)
                )
                blocks.append(block)
            
            feedback_loops = []
            if feedback_config:
                for fb_config in feedback_config:
                    feedback_loop = FeedbackLoop(
                        id=fb_config['id'],
                        type=FeedbackType(fb_config['type']),
                        reviewer_role=fb_config['reviewer_role'],
                        target_block=fb_config['target_block'],
                        criteria=fb_config['criteria'],
                        max_iterations=fb_config.get('max_iterations', 3),
                        improvement_prompt=fb_config.get('improvement_prompt', '')
                    )
                    feedback_loops.append(feedback_loop)
            
            self.chain_system.create_chain(chain_id, blocks, feedback_loops)
            return True
            
        except Exception as e:
            print(f"Error creating custom chain: {e}")
            return False
