import json
from pathlib import Path
from typing import List, Dict
from hybrid_retriever import build_rag_system, RAGSystem

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = BASE_DIR / "golden_dataset.json"


def evaluar(rag_system: RAGSystem, golden_set: List[Dict]) -> Dict:
    resultados_por_pregunta = []

    for caso in golden_set:
        top_k = rag_system.obtener_top_k(caso["pregunta"])
        fuentes_recuperadas = [r["fuente"] for r in top_k]

        recall = 1.0 if caso["documento_id_esperado"] in fuentes_recuperadas else 0.0
        coincidencias = sum(1 for f in fuentes_recuperadas if f == caso["documento_id_esperado"])
        precision = coincidencias / len(top_k) if top_k else 0.0

        resultados_por_pregunta.append({
            "pregunta": caso["pregunta"],
            "esperado": caso["documento_id_esperado"],
            "recuperados": fuentes_recuperadas,
            "recall@5": recall,
            "precision@5": precision,
        })

    recall_promedio = sum(r["recall@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)
    precision_promedio = sum(r["precision@5"] for r in resultados_por_pregunta) / len(resultados_por_pregunta)

    return {
        "detalle": resultados_por_pregunta,
        "recall@5_promedio": recall_promedio,
        "precision@5_promedio": precision_promedio,
    }


def main():
    if not GOLDEN_SET_PATH.exists():
        print(f"❌ No se encontró el archivo {GOLDEN_SET_PATH}")
        return

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    print("🚀 Inicializando RAGSystem híbrido...")
    rag = build_rag_system(k=5)

    print("📊 Ejecutando evaluación con Golden Dataset...\n")
    reporte = evaluar(rag, golden_set)

    print("=" * 80)
    for r in reporte["detalle"]:
        estado = "✅" if r["recall@5"] == 1.0 else "❌"
        print(f"{estado} Pregunta: {r['pregunta']}")
        print(f"   Esperado:    {r['esperado']}")
        print(f"   Recuperados: {r['recuperados']}")
        print(f"   Recall@5:    {r['recall@5']:.0%} | Precision@5: {r['precision@5']:.0%}\n")

    print("=" * 80)
    print(f"📈 RECALL@5 PROMEDIO:    {reporte['recall@5_promedio']:.1%}")
    print(f"📈 PRECISION@5 PROMEDIO: {reporte['precision@5_promedio']:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()