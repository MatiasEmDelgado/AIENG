from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from app.state import AgentState
from app.tools import consultar_benchmarks_cloud, calcular_tco_anual
from app.supervisor import supervisor_node
from app.hitl import human_approval_node

# Configuración de especialistas
research_agent = create_react_agent(
    ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0),
    tools=[consultar_benchmarks_cloud],
)

analyst_agent = create_react_agent(
    ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0),
    tools=[calcular_tco_anual],
)


async def research_node(state: AgentState):
    prompt = f"Consulta: {state['messages'][0].content}"
    res = await research_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    texto = res["messages"][-1].content
    if isinstance(texto, list):
        texto = " ".join([b.get("text", "") for b in texto if isinstance(b, dict)])
    return {
        "messages": [HumanMessage(content=f"[Investigador]: {texto}")],
        "research_data": texto,
        "step_count": 1,
    }


async def analyst_node(state: AgentState):
    # Si la tarea era crítica y no fue aprobada por el humano, no se calcula
    if state.get("is_critical") and state.get("approved") is False:
        return {
            "messages": [HumanMessage(content="[Analista]: Operación rechazada por supervisión humana.")],
            "analysis_data": "Rechazado por humano.",
            "step_count": 1,
        }

    prompt = f"Datos investigación: {state.get('research_data')}\nConsigna: {state['messages'][0].content}"
    res = await analyst_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    texto = res["messages"][-1].content
    if isinstance(texto, list):
        texto = " ".join([b.get("text", "") for b in texto if isinstance(b, dict)])
    return {
        "messages": [HumanMessage(content=f"[Analista]: {texto}")],
        "analysis_data": texto,
        "step_count": 1,
    }


def route_supervisor(state: AgentState):
    next_step = state["next_step"]
    # Si va al analista y es crítico, pasa primero por la compuerta humana
    if next_step == "analyst" and state.get("is_critical") and state.get("approved") is None:
        return "human_approval"
    if next_step in ("researcher", "analyst"):
        return next_step
    return END


# Checkpointer en memoria compatible con interrupt de LangGraph
checkpointer = MemorySaver()

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", research_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("analyst", analyst_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "human_approval": "human_approval",
        "analyst": "analyst",
        END: END,
    },
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("human_approval", "analyst")
workflow.add_edge("analyst", "supervisor")

app_graph = workflow.compile(checkpointer=checkpointer)