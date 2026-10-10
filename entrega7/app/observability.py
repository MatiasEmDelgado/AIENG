import os


def setup_observability() -> None:
    """Valida la configuración de observabilidad para LangSmith."""
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT", "entrega7-multiagent-api")

    if tracing and api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project
        print(f"📡 Observabilidad LangSmith activa en el proyecto: '{project}'")
    else:
        print("⚠️ Observabilidad LangSmith deshabilitada o LANGSMITH_API_KEY ausente.")