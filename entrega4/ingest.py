import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Rutas y configuración de entorno
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

DATA_DIR = BASE_DIR / "data"
INDEX_NAME = os.environ.get("INDEX_NAME", "techcorp-rag-hibrido")
NAMESPACE = "politicas-internas"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DOCUMENTOS = {
    "politica_vacaciones.txt": """
Política de Vacaciones - TechCorp

Todos los empleados en relación de dependencia directa de TechCorp tienen derecho a
días de descanso anual pago, calculados según su antigüedad en la empresa.

Los empleados con menos de 5 años de antigüedad acceden a 14 días corridos de vacaciones
por año calendario. Los empleados con 5 años o más, y hasta 10 años, acceden a 21 días
corridos. Los empleados con más de 10 años de antigüedad acceden a 28 días corridos.

Las vacaciones deben solicitarse con un mínimo de 30 días de anticipación a través del
sistema interno de Recursos Humanos, y quedan sujetas a la aprobación del líder de equipo
directo.
""",
    "politica_teletrabajo.txt": """
Política de Teletrabajo - TechCorp

TechCorp adopta un esquema de trabajo híbrido para todas las áreas cuyas funciones no
requieran presencia física obligatoria. El esquema estándar es de 3 días de trabajo
remoto y 2 días de trabajo presencial por semana, coordinados con el líder de equipo.

Los empleados deben contar con una conexión a internet estable de al menos 20 Mbps.
TechCorp provee un subsidio mensual para conectividad y equipamiento de home office.

Las áreas de Soporte Técnico Nivel 1 y Recepción mantienen un esquema 100% presencial
por la naturaleza de sus funciones.
""",
    "politica_seguridad_informatica.txt": """
Política de Seguridad Informática - TechCorp

Todo empleado con acceso a sistemas internos de TechCorp debe utilizar autenticación
de dos factores (2FA) en las plataformas corporativas de correo, repositorios de código
y sistemas de gestión interna, sin excepción.

Las contraseñas deben tener un mínimo de 12 caracteres, combinando mayúsculas,
minúsculas, números y símbolos, y deben renovarse cada 90 días.

Ante la sospecha de un incidente de seguridad, el empleado debe notificar de inmediato
al área de Seguridad de la Información dentro de las primeras 2 horas de detectado el evento.
""",
    "onboarding_nuevos_empleados.txt": """
Proceso de Onboarding - TechCorp

El proceso de incorporación de nuevos empleados en TechCorp dura 4 semanas y está
compuesto por tres etapas: inducción general, capacitación específica del rol, y
acompañamiento con un mentor asignado (buddy).

Durante la primera semana, el nuevo empleado recibe sus credenciales de acceso, el
equipamiento de trabajo y participa de una inducción general sobre la cultura de TechCorp.

En la semana 4 se realiza una reunión de cierre de onboarding entre el nuevo empleado,
su líder y Recursos Humanos.
""",
}


def asegurar_archivos():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for nombre_archivo, contenido in DOCUMENTOS.items():
        archivo = DATA_DIR / nombre_archivo
        if not archivo.exists():
            archivo.write_text(contenido.strip(), encoding="utf-8")
    print(f"✅ Archivos de datos listos en {DATA_DIR}")


def obtener_chunks():
    asegurar_archivos()
    loader = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documentos_crudos = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documentos_crudos)

    for i, chunk in enumerate(chunks):
        nombre_archivo = Path(chunk.metadata["source"]).name
        chunk.metadata["source"] = nombre_archivo
        chunk.metadata["categoria"] = nombre_archivo.replace(".txt", "").replace("_", " ")
        chunk.metadata["chunk_id"] = i

    return chunks


def setup_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        from getpass import getpass
        api_key = getpass("🔑 PINECONE_API_KEY: ").strip()
        os.environ["PINECONE_API_KEY"] = api_key

    pc = Pinecone(api_key=api_key)
    indices_existentes = [i["name"] for i in pc.list_indexes()]

    if INDEX_NAME not in indices_existentes:
        print(f"🆕 Creando índice '{INDEX_NAME}' (dimensión {EMBEDDING_DIM})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"♻️ El índice '{INDEX_NAME}' ya existe en Pinecone.")

    return pc.Index(INDEX_NAME)


def indexar():
    indice = setup_pinecone()
    chunks = obtener_chunks()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    stats = indice.describe_index_stats()
    vectores_en_namespace = stats.get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", 0)

    if vectores_en_namespace > 0:
        print(f"♻️ Se encontraron {vectores_en_namespace} vectores existentes en el namespace '{NAMESPACE}'. Cargando sin reindexar.")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=NAMESPACE,
        )
    else:
        print(f"🆕 Indexando {len(chunks)} fragmentos en Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )
        print("✅ Documentos subidos con éxito a Pinecone.")

    return vectorstore, chunks


if __name__ == "__main__":
    indexar()