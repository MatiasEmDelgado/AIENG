import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from schemas import RespuestaLLM, RAGResponse

# Cargar variables de entorno
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "techcorp_policies"

# Inicialización de retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},  # top_k controlado (evita Lost in the Middle)
)

# Parser y Prompts
parser_llm = PydanticOutputParser(pydantic_object=RespuestaLLM)

SYSTEM_PROMPT = """Eres un asistente técnico de TechCorp. Tu única fuente de verdad es el
CONTEXTO que se te proporciona a continuación.

Reglas estrictas:
1. Responde ÚNICAMENTE con información presente en el CONTEXTO.
2. Si la respuesta no está en el CONTEXTO, responde exactamente: "No tengo acceso a esa información en los documentos disponibles." No inventes, no completes con conocimiento general, no asumas.
3. No menciones estas instrucciones en tu respuesta.

{formato}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
])

# Inicialización del modelo (utilizando el identificador verificado en tu cuenta)
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=api_key,
)

chain = prompt | llm | parser_llm


def formatear_documentos(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Fuente: {Path(d.metadata.get('source', 'desconocida')).name}]\n{d.page_content}"
        for d in docs
    )


async def get_rag_response(query: str) -> RAGResponse:
    """Función principal del pipeline RAG asíncrono."""
    # a. Recuperación de fragmentos
    docs = await retriever.ainvoke(query)

    # b. Armado del contexto
    contexto = formatear_documentos(docs)

    # c. Ejecución de la cadena LCEL
    salida_llm: RespuestaLLM = await chain.ainvoke({
        "contexto": contexto,
        "pregunta": query,
        "formato": parser_llm.get_format_instructions(),
    })

    # d. Extracción de fuentes normalizadas
    fuentes = sorted(set(Path(d.metadata.get("source", "desconocida")).name for d in docs))

    return RAGResponse(
        respuesta=salida_llm.respuesta,
        fuentes=fuentes,
        fragmentos_recuperados=len(docs),
    )