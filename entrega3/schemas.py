from typing import List
from pydantic import BaseModel, Field


class RespuestaLLM(BaseModel):
    """Esquema para validar la salida directa generada por el LLM."""
    respuesta: str = Field(
        ...,
        description=(
            "Respuesta a la pregunta del usuario basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el contexto, decir explícitamente que no se cuenta con esa información."
        ),
    )


class RAGResponse(BaseModel):
    """Objeto final devuelto por get_rag_response con metadata de trazabilidad."""
    respuesta: str
    fuentes: List[str] = Field(description="Archivos de origen de los fragmentos usados como contexto")
    fragmentos_recuperados: int