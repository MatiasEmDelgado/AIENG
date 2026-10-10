from typing import Dict, Any
from langchain_core.tools import tool


@tool
def consultar_benchmarks_cloud(proveedor: str) -> Dict[str, Any]:
    """Consulta métricas de latencia, costo mensual base y soporte de un proveedor vectorial."""
    db = {
        "pinecone": {"latencia_ms": 42, "costo_mensual_usd": 70, "uptime": 99.99, "soporte_serverless": True},
        "chromadb": {"latencia_ms": 15, "costo_mensual_usd": 25, "uptime": 99.50, "soporte_serverless": False},
        "qdrant": {"latencia_ms": 35, "costo_mensual_usd": 55, "uptime": 99.90, "soporte_serverless": True},
    }
    return db.get(proveedor.lower().strip(), {"error": f"No hay datos para {proveedor}."})


@tool
def calcular_tco_anual(costo_mensual: float, requests_millon_mes: float, latencia_ms: float) -> Dict[str, Any]:
    """Calcula el costo total anual proyectado (TCO) y califica la eficiencia por latencia."""
    costo_base_anual = costo_mensual * 12
    costo_volumen = requests_millon_mes * 0.15 * 12
    total_anual = costo_base_anual + costo_volumen
    eficiencia = "ALTA" if latencia_ms < 30 else ("MEDIA" if latencia_ms < 50 else "BAJA")
    return {
        "tco_anual_usd": round(total_anual, 2),
        "eficiencia_latencia": eficiencia,
        "desglose": f"Base: ${costo_base_anual:.2f}/año + Volumen: ${costo_volumen:.2f}/año",
    }