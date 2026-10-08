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
def consultar_benchmarks_cloud(proveedor: str) -> Dict[str, Any]:
    """Consulta métricas de latencia, costo mensual base y soporte de un proveedor de base vectorial.

    Args:
        proveedor: Nombre del proveedor ('Pinecone', 'ChromaDB', o 'Qdrant').
    """
    db_metricas = {
        "pinecone": {"latencia_ms": 42, "costo_mensual_usd": 70, "uptime": 99.99, "soporte_serverless": True},
        "chromadb": {"latencia_ms": 15, "costo_mensual_usd": 25, "uptime": 99.50, "soporte_serverless": False},
        "qdrant": {"latencia_ms": 35, "costo_mensual_usd": 55, "uptime": 99.90, "soporte_serverless": True},
    }
    key = proveedor.lower().strip()
    return db_metricas.get(key, {"error": f"No hay datos para {proveedor}. Disponibles: Pinecone, ChromaDB, Qdrant"})


tools_researcher = [consultar_benchmarks_cloud]


def get_research_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.0,
        max_retries=3,
    )
    return create_react_agent(llm, tools=tools_researcher)


def extraer_texto(contenido) -> str:
    if isinstance(contenido, list):
        textos = [b.get("text", "") for b in contenido if isinstance(b, dict) and "text" in b]
        return "\n".join(textos) if textos else str(contenido)
    return str(contenido)


async def research_node(state: AgentState) -> Dict[str, Any]:
    """Nodo especialista en investigación técnica."""
    system_prompt = (
        "Eres un Agente Especialista en Investigación Técnica. Tu único objetivo es consultar "
        "las métricas y benchmarks de los proveedores solicitados usando tu herramienta y devolver solo los datos encontrados."
    )
    consulta = state["messages"][0].content
    agent = get_research_agent()

    resultado = await agent.ainvoke({
        "messages": [HumanMessage(content=f"{system_prompt}\nConsulta: {consulta}")]
    })

    texto = extraer_texto(resultado["messages"][-1].content)

    return {
        "messages": [HumanMessage(content=f"[Investigador]: {texto}")],
        "research_data": texto,
        "step_count": 1,
    }