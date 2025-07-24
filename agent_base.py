import abc
from tasks import Task

class Agent(abc.ABC):
    """Base class for agents calling external LLM APIs."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    async def handle(self, task: Task) -> str:
        """Process a Task and return a response."""
        raise NotImplementedError
