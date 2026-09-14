import os
import asyncio
from getpass import getpass
from dotenv import load_dotenv

from schemas import ChatMessage, LLMConfig, Provider
from clients import AsyncLLMManager

# 1. Carga o solicitud segura de credenciales
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("🔑 OPENAI_API_KEY: ").strip()

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass("🔑 ANTHROPIC_API_KEY: ").strip()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("🔑 Ingresá tu GOOGLE_API_KEY (gratis en aistudio.google.com/apikey): ").strip()


async def main():
    mensajes = [
        ChatMessage(role="user", content="¿Qué es la entropía en un párrafo breve?")
    ]

    # Configuramos el proveedor a evaluar (puedes cambiar a Provider.ANTHROPIC o Provider.GEMINI)
    config = LLMConfig(
        provider=Provider.GEMINI,
        model="gemini-3.6-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.7,
        max_tokens=500,
    )

    manager = AsyncLLMManager(config)

    print("--- 1. LLAMADA REGULAR (No Streaming) ---")
    respuesta = await manager.generate(mensajes)
    if respuesta.error:
        print(f"Error: {respuesta.error}")
    else:
        print(f"[{respuesta.provider.value} - {respuesta.model}]:\n{respuesta.content}\n")

    print("--- 2. LLAMADA EN MODO STREAMING ---")
    print(f"[{config.provider.value} streaming]: ", end="", flush=True)
    async for token in manager.generate_stream(mensajes):
        print(token, end="", flush=True)
    print("\n\n--- Fin de la prueba ---")


if __name__ == "__main__":
    asyncio.run(main())