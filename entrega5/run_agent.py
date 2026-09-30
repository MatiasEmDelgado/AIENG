import asyncio
import json
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import get_agent_graph, DB_PATH

TRAZA_FILE = Path(__file__).resolve().parent / "traza_ejecucion.json"


async def ejecutar_test():
    print("🚀 Iniciando prueba del Agente con persistencia Sqlite...")

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app = await get_agent_graph(checkpointer)
        config = {
            "configurable": {"thread_id": "sesion-techcorp-001"},
            "recursion_limit": 10  # Techo de recursión exigido para evitar bucles
        }

        traza_completa = []

        # Paso 1: Pregunta que exige invocar al menos 2 herramientas (buscar pedidos y consultar detalle)
        query_1 = "¿Cuántos pedidos tuvo el cliente 102, cuál fue el total acumulado y qué productos compró en su último pedido?"
        print(f"\n🧑 Input 1: {query_1}")

        eventos_p1 = []
        async for event in app.astream({"messages": [HumanMessage(content=query_1)]}, config, stream_mode="values"):
            ultimo_msg = event["messages"][-1]
            tipo = ultimo_msg.__class__.__name__
            contenido = ultimo_msg.content
            tool_calls = getattr(ultimo_msg, "tool_calls", None)

            eventos_p1.append({
                "tipo": tipo,
                "contenido": str(contenido),
                "tool_calls": tool_calls
            })

            if tool_calls:
                print(f"  ⚙️ [Agente decide llamar herramienta]: {tool_calls}")
            elif tipo == "ToolMessage":
                print(f"  📥 [Resultado de herramienta]: {contenido}")
            elif tipo == "AIMessage" and contenido:
                print(f"\n🤖 [Respuesta Agente Paso 1]:\n{contenido}")

        traza_completa.append({"paso": 1, "input": query_1, "eventos": eventos_p1})

        # Paso 2: Resiliencia de estado (memoria en el mismo thread_id sin repetir datos)
        query_2 = "¿Y cómo abonó ese último pedido?"
        print(f"\n🧑 Input 2 (Memoria / mismo thread_id): {query_2}")

        eventos_p2 = []
        async for event in app.astream({"messages": [HumanMessage(content=query_2)]}, config, stream_mode="values"):
            ultimo_msg = event["messages"][-1]
            tipo = ultimo_msg.__class__.__name__
            contenido = ultimo_msg.content
            tool_calls = getattr(ultimo_msg, "tool_calls", None)

            eventos_p2.append({
                "tipo": tipo,
                "contenido": str(contenido),
                "tool_calls": tool_calls
            })

            if tool_calls:
                print(f"  ⚙️ [Agente decide llamar herramienta]: {tool_calls}")
            elif tipo == "ToolMessage":
                print(f"  📥 [Resultado de herramienta]: {contenido}")
            elif tipo == "AIMessage" and contenido:
                print(f"\n🤖 [Respuesta Agente Paso 2]:\n{contenido}")

        traza_completa.append({"paso": 2, "input": query_2, "eventos": eventos_p2})

        # Guardar la traza ReAct en formato JSON dentro del repo
        with open(TRAZA_FILE, "w", encoding="utf-8") as f:
            json.dump(traza_completa, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Traza ReAct exportada exitosamente a: {TRAZA_FILE.name}")


if __name__ == "__main__":
    asyncio.run(ejecutar_test())