import operator
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    research_data: Optional[str]
    analysis_data: Optional[str]
    step_count: Annotated[int, operator.add]
    # HITL: bandera para pausar si la tarea requiere costo o impacto alto
    is_critical: bool
    approved: Optional[bool]