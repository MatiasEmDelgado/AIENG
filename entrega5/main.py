import asyncio
import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent import get_agent_graph, DB_PATH


async def cli():
    session_id = f"session-{uuid.uuid4().hex[:6]}"
    print(f"💬 Agente ReAct Interactivo - Sesión ID: {session_id}")
    print("   (Escribe 'salir' para cerrar la sesión)\n")

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app = await get_agent_graph(checkpointer)
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 10
        }

        while True:
            try:
                user_prompt = input("\n🧑 Vos: ").strip()
                if user_prompt.lower() in ("salir", "exit", "quit", ""):
                    print("👋 Sesión finalizada.")
                    break

                async for event in app.astream(
                    {"messages": [HumanMessage(content=user_prompt)]},
                    config,
                    stream_mode="values"
                ):
                    msg = event["messages"][-1]
                    if msg.__class__.__name__ == "AIMessage" and msg.content and not getattr(msg, "tool_calls", None):
                        print(f"\n🤖 Agente: {msg.content}")

            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    asyncio.run(cli())