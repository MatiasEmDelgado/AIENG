from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from state import AgentState


@tool
def calcular_tco_anual(costo_mensual: float, requests_millon_mes: float, latencia_ms: float) -> Dict[str, Any]:
    """Calcula el costo total anual de propiedad (TCO) y califica la eficiencia por latencia.

    Args:
        costo_mensual: Costo base mensual en USD.
        requests_millon_mes: Volumen mensual en millones de peticiones.
        latencia_ms: Latencia promedio en milisegundos.
    """
    costo_base_anual = costo_mensual * 12
    costo_volumen = requests_millon_mes * 0.15 * 12
    total_anual = costo_base_anual + costo_volumen
    eficiencia = "ALTA" if latencia_ms < 30 else ("MEDIA" if latencia_ms < 50 else "BAJA")

    return {
        "tco_anual_usd": round(total_anual, 2),
        "eficiencia_latencia": eficiencia,
        "desglose": f"Base: ${costo_base_anual:.2f}/año + Volumen: ${costo_volumen:.2f}/año",
    }


tools_analyst = [calcular_tco_anual]


def get_analyst_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.0,
        max_retries=3,
    )
    return create_react_agent(llm, tools=tools_analyst)


def extraer_texto(contenido) -> str:
    if isinstance(contenido, list):
        textos = [b.get("text", "") for b in contenido if isinstance(b, dict) and "text" in b]
        return "\n".join(textos) if textos else str(contenido)
    return str(contenido)


async def analyst_node(state: AgentState) -> Dict[str, Any]:
    """Nodo especialista de cómputo y análisis numérico."""
    datos_previos = state.get("research_data", "Sin datos de investigación.")
    system_prompt = (
        "Eres un Agente Especialista en Análisis Numérico y Cómputo. Debes usar tu herramienta "
        "para calcular el TCO anual de los proveedores comparados tomando en cuenta el volumen de requests indicado."
    )
    prompt_usuario = (
        f"{system_prompt}\n\nDatos de investigación:\n{datos_previos}\n\n"
        f"Consigna original: {state['messages'][0].content}"
    )

    agent = get_analyst_agent()
    resultado = await agent.ainvoke({"messages": [HumanMessage(content=prompt_usuario)]})
    texto = extraer_texto(resultado["messages"][-1].content)

    return {
        "messages": [HumanMessage(content=f"[Analista]: {texto}")],
        "analysis_data": texto,
        "step_count": 1,
    }