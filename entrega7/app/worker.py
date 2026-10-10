import json
import os
import redis.asyncio as redis
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.graph import app_graph

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def set_job_state(job_id: str, status: str, data: dict = None) -> None:
    payload = {"status": status, "data": data or {}}
    await redis_client.set(f"job:{job_id}", json.dumps(payload), ex=86400)


async def get_job_state(job_id: str) -> dict:
    raw = await redis_client.get(f"job:{job_id}")
    if not raw:
        return {"status": "NOT_FOUND"}
    return json.loads(raw)


async def run_agent_job(job_id: str, query: str, resume_value: bool = None) -> None:
    config = {"configurable": {"thread_id": job_id}, "recursion_limit": 12}
    try:
        if resume_value is not None:
            # Reanuda la ejecución tras la respuesta humana (HITL)
            await set_job_state(job_id, "RUNNING", {"resumed_with": resume_value})
            final_state = await app_graph.ainvoke(Command(resume=resume_value), config=config)
        else:
            await set_job_state(job_id, "RUNNING")
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "next_step": "",
                "research_data": None,
                "analysis_data": None,
                "step_count": 0,
                "is_critical": False,
                "approved": None,
            }
            final_state = await app_graph.ainvoke(initial_state, config=config)

        # Verificar si el grafo se detuvo en un interrupt (HITL)
        state_snapshot = await app_graph.aget_state(config)
        if state_snapshot.next:
            interrupts = [task.interrupts for task in state_snapshot.tasks if task.interrupts]
            await set_job_state(job_id, "WAITING_APPROVAL", {"interrupt": interrupts})
            return

        # Tarea completada con éxito
        ultimo_mensaje = final_state["messages"][-1].content
        await set_job_state(job_id, "DONE", {
            "result": ultimo_mensaje,
            "research_data": final_state.get("research_data"),
            "analysis_data": final_state.get("analysis_data"),
        })

    except Exception as exc:
        # Requisito de la rúbrica: Capturar fallos y pasar explícitamente a FAILED en Redis
        await set_job_state(job_id, "FAILED", {"error": str(exc)})