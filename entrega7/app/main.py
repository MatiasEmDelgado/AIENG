import uuid
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.observability import setup_observability
from app.worker import run_agent_job, get_job_state

setup_observability()

app = FastAPI(
    title="Orquestador Multi-Agente API",
    version="1.0.0",
    description="API REST asíncrona con persistencia de estado en Redis, observabilidad y HITL.",
)


class TaskRequest(BaseModel):
    query: str


class ApproveRequest(BaseModel):
    approved: bool


@app.post("/tasks", status_code=202)
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks):
    """Encola una tarea de razonamiento multi-agente y retorna de inmediato el job_id."""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_agent_job, job_id, req.query)
    return {"job_id": job_id, "status": "PENDING"}


@app.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    """Consulta el estado del job en Redis (PENDING, RUNNING, WAITING_APPROVAL, DONE, FAILED)."""
    state = await get_job_state(job_id)
    if state.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return {"job_id": job_id, **state}


@app.post("/tasks/{job_id}/approve", status_code=202)
async def approve_task(job_id: str, req: ApproveRequest, background_tasks: BackgroundTasks):
    """Reanuda la ejecución del grafo pausado en el nodo Human-in-the-Loop."""
    state = await get_job_state(job_id)
    if state.get("status") != "WAITING_APPROVAL":
        raise HTTPException(status_code=400, detail="La tarea no está esperando aprobación humana.")

    background_tasks.add_task(run_agent_job, job_id, "", resume_value=req.approved)
    return {"job_id": job_id, "status": "RESUMING", "decision": req.approved}