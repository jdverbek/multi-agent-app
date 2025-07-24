from dataclasses import dataclass
from typing import Optional

@dataclass
class Task:
    type: str
    content: str
    role: Optional[str] = None
    response: Optional[str] = None
