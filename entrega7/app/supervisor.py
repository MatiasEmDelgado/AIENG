from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.state import AgentState

OPCIONES = Literal["researcher", "analyst", "FINISH"]


class SupervisorDecision(BaseModel):
    next_step: OPCIONES = Field(description="Siguiente agente o FINISH.")
    razon: str = Field(description="Motivo del ruteo.")
    is_critical: bool = Field(
        default=False,
        description="True si la tarea involucra cálculos financieros de alto impacto o presupuesto corporativo.",
    )
    sintesis_final: str = Field(default="", description="Síntesis final estructurada si next_step es FINISH.")


def get_supervisor_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)
    return llm.with_structured_output(SupervisorDecision)


async def supervisor_node(state: AgentState):
    if state.get("step_count", 0) >= 6:
        return {
            "next_step": "FINISH",
            "messages": [HumanMessage(content="[Supervisor]: Límite de pasos alcanzado. Cerrando tarea.")],
        }

    system_prompt = (
        "Eres el Supervisor de un equipo multi-agente:\n"
        "1. 'researcher': Recolecta métricas y datos crudos.\n"
        "2. 'analyst': Calcula TCO y análisis cuantitativos.\n"
        "Marca is_critical=True si la tarea procesa proyecciones financieras o de infraestructura a gran escala.\n"
        "Si ya tienes datos y cálculos completos, responde FINISH con la síntesis final."
    )

    consulta = state["messages"][0].content
    research_actual = state.get("research_data") or "Pendiente"
    analysis_actual = state.get("analysis_data") or "Pendiente"

    contexto = (
        f"Consigna: {consulta}\n"
        f"Datos Investigación: {research_actual}\n"
        f"Análisis Cómputo: {analysis_actual}\n"
        f"Aprobación Humana: {state.get('approved')}"
    )

    llm = get_supervisor_llm()
    decision: SupervisorDecision = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=contexto),
    ])

    updates = {
        "next_step": decision.next_step,
        "is_critical": decision.is_critical,
    }

    if decision.next_step == "FINISH" and decision.sintesis_final:
        updates["messages"] = [HumanMessage(content=f"### Síntesis Final\n\n{decision.sintesis_final}")]

    return updates