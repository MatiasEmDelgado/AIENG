import asyncio
import time
import httpx
import numpy as np

API_URL = "http://localhost:8000"

QUERIES = [
    "Evaluá Pinecone vs ChromaDB con 5 millones de peticiones al mes.",
    "Compará Qdrant y Pinecone obteniendo benchmarks y calculá el TCO anual con 2M req/mes.",
    "Analizá los costos de ChromaDB vs Qdrant con 10M peticiones al mes.",
    "Calculá la eficiencia y TCO de Pinecone asumiendo 1 millón de peticiones al mes.",
    "Compará Pinecone, ChromaDB y Qdrant: obtené métricas y calculá TCO anual proyectado.",
]


async def dispatch_and_wait(client: httpx.AsyncClient, query: str, index: int):
    start = time.perf_counter()
    res = await client.post(f"{API_URL}/tasks", json={"query": query})
    job_id = res.json()["job_id"]
    print(f"[{index+1}] Job encolado: {job_id}")

    # Polling hasta que finalice o requiera aprobación
    while True:
        await asyncio.sleep(2)
        status_res = await client.get(f"{API_URL}/tasks/{job_id}")
        data = status_res.json()
        status = data.get("status")

        if status == "WAITING_APPROVAL":
            # Aprueba automáticamente en el test de carga
            await client.post(f"{API_URL}/tasks/{job_id}/approve", json={"approved": True})
        elif status in ("DONE", "FAILED"):
            duration = time.perf_counter() - start
            print(f"[{index+1}] Job {job_id} terminado en {duration:.2f}s (Estado: {status})")
            return duration


async def run_load_test():
    print("🚀 Lanzando 5 peticiones concurrentes contra la API...")
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [dispatch_and_wait(client, q, i) for i, q in enumerate(QUERIES)]
        latencias = await asyncio.gather(*tasks)

    p95 = np.percentile(latencias, 95)
    print("\n" + "=" * 50)
    print(f" Latencia media: {np.mean(latencias):.2f}s")
    print(f" Latencia p95:   {p95:.2f}s")
    print("=" * 50)
    print("👉 Abre tu dashboard de LangSmith/Phoenix para tomar las capturas de trazas y costo.")


if __name__ == "__main__":
    asyncio.run(run_load_test())