# AI Engineer Core: Unified Async LLM Client & Best Practices 🚀

Repositorio orientado al aprendizaje y puesta en práctica de patrones de diseño, arquitectura y buenas prácticas requeridas en el rol de **AI Engineer**.

Este proyecto implementa un cliente unificado, tipado, asíncrono y desacoplado para interactuar con los principales proveedores de modelos de lenguaje grande (LLMs): **OpenAI**, **Anthropic** y **Google GenAI**.


🛠️ Instalación y Entorno
1. Clonar el repositorio y crear entorno virtual
Se recomienda utilizar Python 3.12+:


```
bash
# Crear el entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1
# En Linux/macOS:
source .venv/bin/activate
```

### 2. Instalar dependencias

> ⚠️ **Importante**: Si ejecutas los comandos en tu terminal local (PowerShell, CMD o Bash), **no utilices el prefijo `!`** (dicho prefijo solo se emplea dentro de Google Colab o Jupyter Notebooks).

Instalación directa:
```bash
pip install -q openai anthropic google-genai pydantic python-dotenv
```

O utilizando el archivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🔑 Configuración de Credenciales

El proyecto soporta dos mecanismos complementarios para gestionar las claves de API:

### Opción A: Archivo `.env` (Recomendado para desarrollo)
Copia el archivo `.env.example` y renómbralo a `.env`:

```env
OPENAI_API_KEY=tu_api_key_de_openai
ANTHROPIC_API_KEY=tu_api_key_de_anthropic
GOOGLE_API_KEY=tu_api_key_de_google_ai_studio
```

> **Nota:** Puedes obtener tu clave gratuita de Google en [Google AI Studio](https://aistudio.google.com/apikey).

### Opción B: Fallback Interactivo en Consola
Si ejecutas el script sin variables configuradas en el archivo `.env`, el sistema solicitará interactivamente las credenciales en tiempo de ejecución sin exponerlas en texto claro:

```python
import os
from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("🔑 OPENAI_API_KEY: ").strip()

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass("🔑 ANTHROPIC_API_KEY: ").strip()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("🔑 Ingresá tu GOOGLE_API_KEY (gratis en aistudio.google.com/apikey): ").strip()
```

---

## 🚀 Ejecución y Validación

Para poner a prueba la intercambiabilidad de los clientes y el soporte de streaming, ejecuta el script principal:

```bash
python main.py
```

---

# Módulo 2: Pipeline de Extracción de Entidades Técnicas (LCEL & Resiliencia) 🛠️

Pipeline asíncrono para procesar texto desestructurado (logs o descripciones de arquitectura) y convertirlo en un objeto validado mediante Pydantic y LangChain.

### Dependencias adicionales
```bash
pip install -q langchain langchain-core langchain-openai langchain-anthropic langchain-google-genai
```

---

# Módulo 3: Sistema RAG Local con ChromaDB 📚

Flujo End-to-End de RAG para responder consultas sobre políticas internas utilizando ChromaDB local y LangChain (LCEL) con salida estructurada anti-alucinación.

### Dependencias adicionales

```bash
pip install -q chromadb langchain-chroma langchain-huggingface sentence-transformers langchain-community tiktoken
```

## Ejecución

```Bash
cd entrega3
python ingest.py
python main.py
```

---

# Módulo 4: Sistema RAG Escalable en la Nube con Pinecone y Búsqueda Híbrida 🌲

Pipeline de recuperación híbrida en la nube combinando búsqueda vectorial densa con Pinecone Serverless y búsqueda léxica dispersa con BM25 (`EnsembleRetriever`), evaluado mediante métricas objetivas de recuperación.

### Dependencias adicionales
```bash
pip install -q pinecone langchain-pinecone rank-bm25

```

### Configuración de Variables
Asegúrate de configurar en tu archivo `.env`:

```env
PINECONE_API_KEY=tu_api_key_de_pinecone
INDEX_NAME=techcorp-rag-hibrido
```

### Ejecución y Pruebas

```bash
# 1. Indexar los datos en Pinecone Serverless
cd entrega4
python ingest.py

# 2. Ejecutar la evaluación del benchmark
python evaluate.py

# 3. Probar el modo interactivo CLI
python main.py
```