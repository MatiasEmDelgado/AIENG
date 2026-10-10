from typing import Dict, Any
from langgraph.types import interrupt
from app.state import AgentState


async def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Nodo Human-in-the-Loop: Pausa la ejecución si la tarea fue marcada como crítica."""
    if state.get("is_critical", False) and state.get("approved") is None:
        # Pausa la ejecución del grafo hasta que un POST /tasks/{id}/approve reanude con valor booleano
        decision = interrupt({
            "action": "APPROVAL_REQUIRED",
            "message": "Operación crítica detectada (análisis financiero/cómputo). Requiere validación humana.",
            "research_data": state.get("research_data"),
        })
        return {"approved": bool(decision)}
    return {"approved": True}