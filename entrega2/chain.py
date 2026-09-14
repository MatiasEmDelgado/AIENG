import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from schemas import EntidadesTecnicas

logger = logging.getLogger("pipeline_extraccion")

# Prompt Template modular sin f-strings hardcodeadas
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un analista técnico. Extraé información estructurada del texto que te pasa el usuario. "
        "Identificá tecnologías mencionadas, evaluá el nivel de criticidad del problema o arquitectura "
        "descripta, y generá un resumen técnico breve.",
    ),
    ("human", "{texto}"),
])


def get_model(provider: str):
    """Instancia el modelo de chat tomando las credenciales de entorno y los modelos vigentes."""
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model="claude-3-5-sonnet-latest",
            temperature=0,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    elif provider == "gemini":
        # Google API usa GOOGLE_API_KEY o GEMINI_API_KEY
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0,
            google_api_key=api_key,
        )
    raise ValueError(f"Proveedor no soportado: {provider}")


def build_chain(provider: str = "gemini"):
    """Construye la cadena LCEL con salida estructurada y resiliencia de reintentos."""
    model = get_model(provider)
    structured_model = model.with_structured_output(EntidadesTecnicas)

    chain = (prompt | structured_model).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
    return chain


async def process_text(text: str, provider: str = "gemini") -> EntidadesTecnicas:
    """Ejecuta la extracción estructurada de manera asíncrona usando .ainvoke()."""
    chain = build_chain(provider)
    logger.info(f"[{provider}] Procesando texto ({len(text)} caracteres)...")

    try:
        resultado: EntidadesTecnicas = await chain.ainvoke({"texto": text})
        logger.info(f"[{provider}] ✅ Extracción validada con éxito")
        return resultado
    except Exception as e:
        logger.error(f"[{provider}] ❌ Falló la extracción tras reintentos: {e}")
        raise