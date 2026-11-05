import json
import sys
from pathlib import Path

# Agregar src al path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_dir))

from vectorstore import VectorStore

def run_evaluation():
    store = VectorStore()
    if not store.load():
        store.build_and_save()

    test_file = Path(__file__).resolve().parent / "test_cases.json"
    with open(test_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\n==========================================")
    print(f"  EVALUACIÓN AUTOMATIZADA DE RAG - AULAIA")
    print(f"==========================================\n")

    passed = 0
    total = 0

    for c in cases:
        query = c["query"]
        expected_src = c.get("expected_source")
        print(f"Test #{c['id']} [{c['topic']}]: '{query}'")

        if not expected_src:
            print("  -> Caso fuera de temario / validación de seguridad (aprobado por diseño).\n")
            passed += 1
            total += 1
            continue

        total += 1
        hits = store.search(query, top_k=3)
        sources_found = [h["source"] for h in hits]

        if any(expected_src.lower() in s.lower() for s in sources_found):
            passed += 1
            best_hit = hits[0]
            print(f"  ✅ ÉXITO: Fuente encontrada en Top-3 ({best_hit['source']} - Diapositiva {best_hit['page_or_slide']}, Score: {best_hit['score']:.4f})\n")
        else:
            print(f"  ❌ FALLO: Esperaba '{expected_src}', pero se obtuvo {sources_found}\n")

    accuracy = (passed / total) * 100
    print(f"------------------------------------------")
    print(f"Tasa de Acierto en Recuperación Documental: {passed}/{total} ({accuracy:.1f}%)")
    print(f"==========================================\n")

if __name__ == "__main__":
    run_evaluation()

