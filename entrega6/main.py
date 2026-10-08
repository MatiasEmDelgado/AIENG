from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno ANTES de importar los nodos/agentes
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

import asyncio
from langchain_core.messages import HumanMessage
from graph import build_multiagent_graph


async def main():
    print("🚀 Compilando el Orquestador Multi-Agente...")
    app = build_multiagent_graph()

    mermaid_graph = app.get_graph().draw_mermaid()
    mermaid_file = BASE_DIR / "graph_diagram.md"
    mermaid_file.write_text(f"```mermaid\n{mermaid_graph}\n```", encoding="utf-8")
    print(f"📊 Diagrama Mermaid guardado en {mermaid_file.name}")

    query = (
        "Necesito evaluar Pinecone frente a ChromaDB: "
        "obtené sus benchmarks de latencia y costo mensual, "
        "y luego calculá el TCO anual proyectado asumiendo un volumen de 5 millones de requests al mes."
    )

    print(f"\n🧑 Solicitud del Usuario:\n{query}\n")
    print("=" * 80)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_step": "",
        "research_data": None,
        "analysis_data": None,
        "step_count": 0,
    }

    async for chunk in app.astream(initial_state, stream_mode="updates"):
        for node_name, updates in chunk.items():
            print(f"📍 [NODO EJECUTADO: {node_name.upper()}]")
            if "next_step" in updates:
                print(f"   Decisión: {updates['next_step']}")
            if "messages" in updates:
                ultimo_msg = updates["messages"][-1]
                print(f"   Mensaje emitido:\n{ultimo_msg.content}\n")
            print("-" * 60)

    print("=" * 80)
    print("✅ Flujo multi-agente finalizado con éxito.")


if __name__ == "__main__":
    asyncio.run(main())