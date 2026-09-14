from hybrid_retriever import build_rag_system


def main():
    print("💬 Modo interactivo — consultá el sistema RAG híbrido (BM25 + Pinecone)")
    print("   (escribí 'salir' para terminar)\n")

    rag_system = build_rag_system(k=5)

    while True:
        try:
            pregunta_usuario = input("🧑 Vos: ").strip()
            if pregunta_usuario.lower() in ("salir", "exit", "quit", ""):
                print("\n👋 Sesión finalizada.")
                break

            resultados = rag_system.obtener_top_k(pregunta_usuario)

            if not resultados:
                print("⚠️ No se recuperó ningún fragmento.\n")
                continue

            print(f"\n📎 Top {len(resultados)} fragmentos recuperados:")
            for i, r in enumerate(resultados, 1):
                print(f"\n  {i}. [{r['fuente']} | {r['categoria']}]")
                print(f"     {r['contenido'][:200]}{'...' if len(r['contenido']) > 200 else ''}")
            print("-" * 80)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()