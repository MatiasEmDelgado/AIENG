import operator
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Acumulador de mensajes estándar de LangGraph
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Próximo nodo a ejecutar: "researcher", "analyst", o "FINISH"
    next_step: str

    # Contexto específico estructurado para evitar contaminación
    research_data: Optional[str]
    analysis_data: Optional[str]

    # Contador de iteraciones para evitar el "Supervisor Infinito"
    step_count: Annotated[int, operator.add]