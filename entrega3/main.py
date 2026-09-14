import asyncio
from getpass import getpass
import os

from ingest import indexar_documentos
from rag_chain import get_rag_response


def asegurar_credenciales():
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass("🔑 GOOGLE_API_KEY: ").strip()


async def main():
    asegurar_credenciales()
    indexar_documentos()

    print("\n" + "=" * 60)
    print("--- 1. PRUEBA DE PREGUNTA CON CONTEXTO (Vacaciones) ---")
    print("=" * 60)
    pregunta_valida = "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?"
    print(f"Pregunta: {pregunta_valida}")
    resp_ok = await get_rag_response(pregunta_valida)
    print(f"\nRespuesta: {resp_ok.respuesta}")
    print(f"Fuentes: {resp_ok.fuentes}")
    print(f"Fragmentos recuperados: {resp_ok.fragmentos_recuperados}")

    print("\n" + "=" * 60)
    print("--- 2. PRUEBA DE PREGUNTA TRAMPA (Anti-alucinación) ---")
    print("=" * 60)
    pregunta_trampa = "¿Cuál es la política de bonos por rendimiento anual en TechCorp?"
    print(f"Pregunta: {pregunta_trampa}")
    resp_trampa = await get_rag_response(pregunta_trampa)
    print(f"\nRespuesta: {resp_trampa.respuesta}")
    print(f"Fuentes: {resp_trampa.fuentes}")
    print(f"Fragmentos recuperados: {resp_trampa.fragmentos_recuperados}")

    print("\n" + "=" * 60)
    print("--- 3. MODO INTERACTIVO (Escribe 'salir' para finalizar) ---")
    print("=" * 60)
    while True:
        try:
            pregunta = input("\n🧑 Vos: ").strip()
            if pregunta.lower() in ("salir", "exit", "quit", ""):
                print("👋 Finalizando sesión.")
                break

            resultado = await get_rag_response(pregunta)
            print(f"\n🤖 Asistente: {resultado.respuesta}")
            print(f"📎 Fuentes: {', '.join(resultado.fuentes)}")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    asyncio.run(main())