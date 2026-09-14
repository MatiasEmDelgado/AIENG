import os
import asyncio
import logging
from pathlib import Path
from getpass import getpass
from dotenv import load_dotenv

from chain import process_text

# Carga de variables de entorno buscando en la raíz o en entrega1/entrega2
ruta_actual = Path(__file__).resolve().parent
load_dotenv(ruta_actual / ".env")
load_dotenv(ruta_actual.parent / ".env")
load_dotenv(ruta_actual.parent / "entrega1" / ".env")

# Fallback solo si realmente falta la key
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("🔑 GOOGLE_API_KEY: ").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def main():
    texto_ejemplo = """
    Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis
    parece saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan
    porque el pool está mal dimensionado. Esto está afectando a usuarios en producción.
    """

    print("\n" + "=" * 50)
    print("--- 1. PRUEBA CON TEXTO TÉCNICO COMPLETO (GEMINI) ---")
    print("=" * 50)

    try:
        resultado = await process_text(texto_ejemplo, provider="gemini")
        print("\nSalida validada (GEMINI):")
        print(resultado.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nGEMINI falló: {e}")

    print("\n" + "=" * 50)
    print("--- 2. PRUEBA DE ESTRÉS (Texto ambiguo) ---")
    print("=" * 50)

    texto_ambiguo = "El sistema anda medio raro últimamente, no sé bien qué está pasando."
    try:
        resultado_ambiguo = await process_text(texto_ambiguo, provider="gemini")
        print("\nSalida para texto ambiguo:")
        print(resultado_ambiguo.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nError capturado correctamente ante datos insuficientes: {e}")


if __name__ == "__main__":
    asyncio.run(main())