from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class Task:
    """Represents a unit of work for agents."""
    type: str
    content: str
    role: Optional[str] = None
    response: Optional[str] = None
    chain_id: Optional[str] = None  # For chain execution requests
    metadata: Optional[dict] = None  # Additional metadata