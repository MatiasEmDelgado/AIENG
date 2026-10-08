from langgraph.graph import StateGraph, START, END
from state import AgentState
from supervisor import supervisor_node
from research_agent import research_node
from analyst_agent import analyst_node


def build_multiagent_graph():
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("analyst", analyst_node)

    # Flujo de entrada
    workflow.add_edge(START, "supervisor")

    # Aristas condicionales de ruteo
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_step"],
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "FINISH": END,
        },
    )

    # Retorno al supervisor para validación y síntesis
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")

    return workflow.compile()