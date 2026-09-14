from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "techcorp_policies"

DOCUMENTOS = {
    "politica_vacaciones.txt": """Política de Vacaciones - TechCorp

Todos los empleados en relación de dependencia directa de TechCorp tienen derecho a
días de descanso anual pago, calculados según su antigüedad en la empresa.

Los empleados con menos de 5 años de antigüedad acceden a 14 días corridos de vacaciones
por año calendario. Los empleados con 5 años o más, y hasta 10 años, acceden a 21 días
corridos. Los empleados con más de 10 años de antigüedad acceden a 28 días corridos.

Las vacaciones deben solicitarse con un mínimo de 30 días de anticipación a través del
sistema interno de Recursos Humanos, y quedan sujetas a la aprobación del líder de equipo
directo. No se acumulan más de 5 días de un período al siguiente, salvo autorización
expresa de Recursos Humanos por motivos operativos.

Durante el mes de diciembre y la primera quincena de enero, TechCorp aplica un régimen
especial de guardias mínimas para garantizar la continuidad operativa, por lo que las
solicitudes de vacaciones en ese período están sujetas a un cupo máximo por equipo.
""",
    "politica_teletrabajo.txt": """Política de Teletrabajo - TechCorp

TechCorp adopta un esquema de trabajo híbrido para todas las áreas cuyas funciones no
requieran presencia física obligatoria. El esquema estándar es de 3 días de trabajo
remoto y 2 días de trabajo presencial por semana, coordinados con el líder de equipo.

Los empleados deben contar con una conexión a internet estable de al menos 20 Mbps y
un espacio de trabajo adecuado. TechCorp provee un subsidio mensual para conectividad
y equipamiento de home office, sujeto a la presentación de comprobantes.

Durante los días de trabajo remoto, se espera que el empleado esté disponible en el
horario laboral habitual (9 a 18 hs) y responda a las comunicaciones internas dentro
de un margen razonable. El incumplimiento reiterado de disponibilidad puede derivar en
la revisión del esquema de teletrabajo asignado.

Las áreas de Soporte Técnico Nivel 1 y Recepción mantienen un esquema 100% presencial
por la naturaleza de sus funciones.
""",
    "politica_seguridad_informatica.txt": """Política de Seguridad Informática - TechCorp

Todo empleado con acceso a sistemas internos de TechCorp debe utilizar autenticación
de dos factores (2FA) en las plataformas corporativas de correo, repositorios de código
y sistemas de gestión interna, sin excepción.

Las contraseñas deben tener un mínimo de 12 caracteres, combinando mayúsculas,
minúsculas, números y símbolos, y deben renovarse cada 90 días. Está prohibido
reutilizar contraseñas de servicios personales en sistemas corporativos.

Ante la sospecha de un incidente de seguridad (phishing, acceso no autorizado, pérdida
de un dispositivo corporativo), el empleado debe notificar de inmediato al área de
Seguridad de la Información a través del canal interno de incidentes, dentro de las
primeras 2 horas de detectado el evento.

El uso de dispositivos personales para acceder a sistemas corporativos (BYOD) requiere
la instalación previa del perfil de gestión de dispositivos móviles (MDM) provisto por
el área de IT.
""",
    "onboarding_nuevos_empleados.txt": """Proceso de Onboarding - TechCorp

El proceso de incorporación de nuevos empleados en TechCorp dura 4 semanas y está
compuesto por tres etapas: inducción general, capacitación específica del rol, y
acompañamiento con un mentor asignado (buddy).

Durante la primera semana, el nuevo empleado recibe sus credenciales de acceso, el
equipamiento de trabajo (notebook, accesorios) y participa de una inducción general
sobre la cultura, valores y estructura organizacional de TechCorp.

Durante las semanas 2 y 3, el empleado recibe capacitación específica de su área junto
a su líder directo, y comienza a participar de reuniones de equipo como observador.

En la semana 4 se realiza una reunión de cierre de onboarding entre el nuevo empleado,
su líder y Recursos Humanos, donde se revisan los objetivos del primer trimestre y se
resuelven dudas pendientes sobre procesos internos.
""",
}


def asegurar_archivos():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for nombre, contenido in DOCUMENTOS.items():
        archivo = DATA_DIR / nombre
        if not archivo.exists():
            archivo.write_text(contenido.strip(), encoding="utf-8")


def indexar_documentos():
    asegurar_archivos()

    # Mismo modelo de embeddings tanto para ingestión como para consulta
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    ya_existe_indice = PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())

    if ya_existe_indice:
        print("♻️ Índice persistente detectado en vectorstore/ — cargando base existente.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )

    print("🆕 Indexando documentos en ChromaDB por primera vez...")
    loader = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documentos_crudos = loader.load()

    # Chunking estratégico: 500 tokens mínimo con overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=70,
    )
    chunks = splitter.split_documents(documentos_crudos)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"✅ Ingesta finalizada. Total de fragmentos indexados: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    indexar_documentos()