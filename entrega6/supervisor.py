from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

OPCIONES = Literal["researcher", "analyst", "FINISH"]


class SupervisorDecision(BaseModel):
    next_step: OPCIONES = Field(
        description="El siguiente agente a intervenir ('researcher', 'analyst') o 'FINISH' si ya se cuenta con la investigación y el análisis completos."
    )
    razon: str = Field(description="Explicación breve de la decisión tomada.")
    sintesis_final: str = Field(
        default="",
        description="Si next_step es FINISH, redacta la síntesis ejecutiva integrando investigación y análisis."
    )


def get_supervisor_model():
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.0,
        max_retries=3,
    )
    return llm.with_structured_output(SupervisorDecision)

async def supervisor_node(state: AgentState):
    """Nodo supervisor que actúa como router y sintetizador final."""
    if state.get("step_count", 0) >= 6:
        return {
            "next_step": "FINISH",
            "messages": [HumanMessage(content="[Supervisor]: Límite de pasos alcanzado. Finalizando proceso.")],
        }

    system_prompt = (
        "Eres el Supervisor de un equipo multi-agente con dos especialistas:\n"
        "1. 'researcher': Investiga datos, benchmarks y métricas técnicas de infraestructura.\n"
        "2. 'analyst': Procesa cálculos matemáticos, TCO y proyecciones numéricas sobre los datos recolectados.\n\n"
        "Reglas de Supervisión estricta:\n"
        "- Si falta recopilar datos de benchmarks/métricas, delega a 'researcher'.\n"
        "- Si ya tienes los datos de investigación pero aún no se calculó el análisis numérico/TCO, delega a 'analyst'.\n"
        "- Si ya tienes tanto los datos de investigación como el análisis numérico, elige 'FINISH' y genera la síntesis final detallada."
    )

    consulta_usuario = state["messages"][0].content
    research_actual = state.get("research_data") or "Pendiente (aún no ejecutado)"
    analysis_actual = state.get("analysis_data") or "Pendiente (aún no ejecutado)"

    contexto_turno = (
        f"Consigna del usuario: {consulta_usuario}\n\n"
        f"Estado actual de la tarea:\n"
        f"- Datos de investigación: {research_actual}\n"
        f"- Análisis numérico: {analysis_actual}\n\n"
        f"¿Cuál es el siguiente paso?"
    )

    structured_supervisor = get_supervisor_model()
    # Enviamos siempre SystemMessage + HumanMessage para cumplir con la API de Gemini
    decision: SupervisorDecision = await structured_supervisor.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=contexto_turno),
    ])

    updates = {"next_step": decision.next_step}
    if decision.next_step == "FINISH" and decision.sintesis_final:
        updates["messages"] = [HumanMessage(content=f"### Síntesis Ejecutiva Final\n\n{decision.sintesis_final}")]

    return updates