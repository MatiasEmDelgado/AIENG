import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tools import ALL_TOOLS

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DB_PATH = BASE_DIR / "checkpoints.db"


def build_model():
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.1)
    return llm.bind_tools(ALL_TOOLS)


async def call_model_node(state: MessagesState):
    llm_with_tools = build_model()
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def get_agent_graph(checkpointer: AsyncSqliteSaver):
    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(MessagesState)

    # Nodos
    workflow.add_node("agent", call_model_node)
    workflow.add_node("tools", tool_node)

    # Aristas
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # Compilar con el checkpointer de persistencia
    return workflow.compile(checkpointer=checkpointer)