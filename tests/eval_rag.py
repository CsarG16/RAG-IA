"""
Evaluación automatizada del pipeline RAG de AulaIA.

Uso:
    python tests/eval_rag.py --tag antes            # solo recuperación (rápido, casi sin cuota)
    python tests/eval_rag.py --tag despues --e2e    # + respuestas completas de Gemini (latencia, alucinación)

Métricas de recuperación (por método: bm25 / dense / hybrid):
    Hit@1 archivo   -> la 1ª diapositiva recuperada es de la clase correcta
    Hit@3 archivo   -> alguna de las 3 primeras es de la clase correcta
    Hit@3 diapo     -> alguna de las 3 primeras es exactamente una diapositiva esperada
    MRR diapo       -> 1/posición de la primera diapositiva correcta (top-6); 1.0 = siempre primera

Métricas end-to-end (--e2e):
    Rechazo fuera de dominio, respuestas con cita de fuente, cobertura de conceptos clave,
    latencia hasta el primer token (TTFT) y latencia total.

Genera tests/reports/eval_<tag>.md y .json como evidencia.
"""
import argparse
import json
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import logging
logging.disable(logging.INFO)

from rag_engine import RAGEngine, STREAM_RESET  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "eval_cases.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODES = ["bm25", "dense", "hybrid"]
REFUSAL_MARK = "no se encuentra registrado"
TOP_K = 6


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


# Equivalencias: el LLM suele escribir los símbolos en LaTeX ($\delta$, $q_0$) en vez de Unicode (δ, q0)
CONCEPT_ALIASES = {
    "σ": [r"\sigma", "sigma", "alfabeto"],
    "δ": [r"\delta", "delta", "funcion de transicion"],
    "ε": [r"\varepsilon", r"\epsilon", "epsilon", "ɛ"],
    "γ": [r"\gamma", "gamma", "alfabeto de la pila", "alfabeto de cinta"],
    "λ": [r"\lambda", "lambda", "palabra vacia"],
    "q0": ["q_0", "q_{0}", "estado inicial"],
    "z0": ["z_0", "z_{0}", "zo"],
    "δ*": [r"\hat{\delta}", r"\delta^*", "δ^*", "extendida"],
    "l1": ["l_1", "l_{1}"],
    "l2": ["l_2", "l_{2}"],
}


def concept_present(concept: str, text_norm: str) -> bool:
    c = norm(concept)
    if c in text_norm:
        return True
    return any(norm(a) in text_norm for a in CONCEPT_ALIASES.get(c, []))


def retrieval_query(engine, case):
    """Usa la misma consulta que usa el motor real (incluye el contexto si el motor lo soporta)."""
    hist = case.get("historial") or []
    if hasattr(engine, "build_retrieval_query"):
        return engine.build_retrieval_query(case["pregunta"], hist)
    return case["pregunta"]


def eval_retrieval(engine, cases):
    rows = []
    for case in cases:
        if not case.get("fuentes"):
            continue
        q = retrieval_query(engine, case)
        row = {"id": case["id"], "tipo": case["tipo"], "pregunta": case["pregunta"], "modos": {}}
        for mode in MODES:
            hits = engine.vector_store.search(q, top_k=TOP_K, mode=mode)
            src_ok = [any(h["source"].startswith(f) for f in case["fuentes"]) for h in hits]
            slide_ok = [ok and h["page_or_slide"] in case["diapositivas"] for ok, h in zip(src_ok, hits)]
            first = next((i + 1 for i, ok in enumerate(slide_ok) if ok), None)
            row["modos"][mode] = {
                "hit1_archivo": bool(src_ok[:1] and src_ok[0]),
                "hit3_archivo": any(src_ok[:3]),
                "hit3_diapo": any(slide_ok[:3]),
                "rr_diapo": (1.0 / first) if first else 0.0,
                "top3": [f"{h['source'][:4].strip()}#{h['page_or_slide']}" for h in hits[:3]],
            }
        rows.append(row)
    return rows


def summarize_retrieval(rows, tipo=None):
    sel = [r for r in rows if tipo is None or r["tipo"] == tipo]
    out = {}
    for mode in MODES:
        ms = [r["modos"][mode] for r in sel]
        n = len(ms) or 1
        out[mode] = {
            "n": len(ms),
            "hit1_archivo": sum(m["hit1_archivo"] for m in ms) / n,
            "hit3_archivo": sum(m["hit3_archivo"] for m in ms) / n,
            "hit3_diapo": sum(m["hit3_diapo"] for m in ms) / n,
            "mrr_diapo": sum(m["rr_diapo"] for m in ms) / n,
        }
    return out


def eval_e2e(engine, cases, pause):
    rows = []
    for i, case in enumerate(cases):
        if i:
            time.sleep(pause)  # respetar el límite de peticiones por minuto del plan gratuito
        t0 = time.perf_counter()
        ttft = None
        text = ""
        error = None
        try:
            gen, _sources = engine.ask_stream(case["pregunta"], history=case.get("historial") or [], level="formal")
            for tok in gen:
                if tok == STREAM_RESET:
                    text = ""
                    continue
                if tok and ttft is None:
                    ttft = time.perf_counter() - t0
                text += tok or ""
        except Exception as e:  # noqa: BLE001
            error = str(e)
        total = time.perf_counter() - t0
        nt = norm(text)
        refused = norm(REFUSAL_MARK) in nt
        quota_fail = "se agoto la cuota" in nt
        conceptos = case.get("conceptos") or []
        cov = (sum(concept_present(c, nt) for c in conceptos) / len(conceptos)) if conceptos else None
        rows.append({
            "id": case["id"], "tipo": case["tipo"], "pregunta": case["pregunta"],
            "ttft_s": round(ttft, 2) if ttft else None, "total_s": round(total, 2),
            "rechazo": refused, "cita_fuente": "fuente" in nt, "cobertura_conceptos": cov,
            "error": error or ("cuota/red" if quota_fail else None),
            "respuesta_inicio": " ".join(text.split())[:160],
            "respuesta_completa": text,
        })
        print(f"  e2e #{case['id']:>2} {case['tipo']:<17} ttft={rows[-1]['ttft_s']}s total={rows[-1]['total_s']}s "
              f"rechazo={refused} cita={rows[-1]['cita_fuente']} cobertura={cov}")
    return rows


def summarize_e2e(rows):
    ok = [r for r in rows if not r["error"]]
    ood = [r for r in ok if r["tipo"] == "fuera_de_dominio"]
    ind = [r for r in ok if r["tipo"] != "fuera_de_dominio"]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"]]
    totals = [r["total_s"] for r in ok]
    covs = [r["cobertura_conceptos"] for r in ind if r["cobertura_conceptos"] is not None]

    def pct(xs, q):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]

    return {
        "casos_ok": len(ok), "casos_con_error": len(rows) - len(ok),
        "rechazo_fuera_de_dominio": (sum(r["rechazo"] for r in ood) / len(ood)) if ood else None,
        "falsos_rechazos_en_dominio": (sum(r["rechazo"] for r in ind) / len(ind)) if ind else None,
        "respuestas_con_cita": (sum(r["cita_fuente"] for r in ind) / len(ind)) if ind else None,
        "cobertura_conceptos_media": statistics.mean(covs) if covs else None,
        "ttft_mediana_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p90_s": pct(ttfts, 0.9),
        "total_mediana_s": statistics.median(totals) if totals else None,
        "total_p90_s": pct(totals, 0.9),
    }


def fmt_pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def fmt_s(x):
    return "—" if x is None else f"{x:.2f} s"


def write_report(tag, engine, ret_rows, e2e_rows):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    vs = engine.vector_store
    emb = getattr(vs, "embedding_model_name", None) or ("gemini-embedding-001" if vs.use_gemini_embeddings else "vectorizador local (hash de palabras)")
    data = {
        "tag": tag, "fecha": datetime.now().isoformat(timespec="seconds"),
        "embeddings": emb, "diapositivas_indexadas": len(vs.metadata), "modelos_llm": engine.models,
        "recuperacion": {
            "global": summarize_retrieval(ret_rows),
            "directa": summarize_retrieval(ret_rows, "directa"),
            "parafraseada": summarize_retrieval(ret_rows, "parafraseada"),
            "seguimiento": summarize_retrieval(ret_rows, "seguimiento"),
        },
        "detalle_recuperacion": ret_rows,
    }
    if e2e_rows is not None:
        data["e2e"] = summarize_e2e(e2e_rows)
        data["detalle_e2e"] = e2e_rows

    (REPORTS_DIR / f"eval_{tag}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"# Evaluación RAG — AulaIA (`{tag}`)", "",
         f"- Fecha: {data['fecha']}",
         f"- Embeddings: **{emb}**",
         f"- Diapositivas indexadas: {data['diapositivas_indexadas']}",
         f"- LLM: {', '.join(engine.models)}", "",
         "## 1. Recuperación (top-6)", ""]
    for grupo, titulo in [("global", "Todas las preguntas"), ("directa", "Preguntas directas"),
                          ("parafraseada", "Preguntas parafraseadas (sinónimos, sin términos técnicos)"),
                          ("seguimiento", "Preguntas de seguimiento (dependen del turno anterior)")]:
        s = data["recuperacion"][grupo]
        L += [f"### {titulo} (n={s['hybrid']['n']})", "",
              "| Método | Hit@1 archivo | Hit@3 archivo | Hit@3 diapositiva | MRR diapositiva |",
              "|---|---|---|---|---|"]
        for m in MODES:
            x = s[m]
            L.append(f"| {m} | {fmt_pct(x['hit1_archivo'])} | {fmt_pct(x['hit3_archivo'])} | "
                     f"{fmt_pct(x['hit3_diapo'])} | {x['mrr_diapo']:.2f} |")
        L.append("")
    L += ["### Detalle por pregunta (híbrido)", "", "| # | Tipo | Pregunta | Top-3 | Diapositiva correcta en top-3 |", "|---|---|---|---|---|"]
    for r in ret_rows:
        h = r["modos"]["hybrid"]
        L.append(f"| {r['id']} | {r['tipo']} | {r['pregunta']} | {', '.join(h['top3'])} | {'✅' if h['hit3_diapo'] else '❌'} |")
    L.append("")

    if e2e_rows is not None:
        e = data["e2e"]
        L += ["## 2. Respuestas completas (end-to-end con Gemini)", "",
              "| Métrica | Valor |", "|---|---|",
              f"| Casos ejecutados sin error | {e['casos_ok']} / {e['casos_ok'] + e['casos_con_error']} |",
              f"| Rechazo correcto de preguntas fuera del temario | {fmt_pct(e['rechazo_fuera_de_dominio'])} |",
              f"| Falsos rechazos (preguntas del curso rechazadas) | {fmt_pct(e['falsos_rechazos_en_dominio'])} |",
              f"| Respuestas del curso con cita de fuente | {fmt_pct(e['respuestas_con_cita'])} |",
              f"| Cobertura media de conceptos clave | {fmt_pct(e['cobertura_conceptos_media'])} |",
              f"| Latencia hasta el primer token (mediana / p90) | {fmt_s(e['ttft_mediana_s'])} / {fmt_s(e['ttft_p90_s'])} |",
              f"| Latencia total (mediana / p90) | {fmt_s(e['total_mediana_s'])} / {fmt_s(e['total_p90_s'])} |", "",
              "| # | Tipo | TTFT | Total | Rechazo | Cita | Conceptos | Inicio de la respuesta |",
              "|---|---|---|---|---|---|---|---|"]
        for r in e2e_rows:
            cov = "—" if r["cobertura_conceptos"] is None else f"{r['cobertura_conceptos'] * 100:.0f}%"
            ini = (r["error"] and f"⚠️ {r['error']}") or r["respuesta_inicio"].replace("|", "/")
            L.append(f"| {r['id']} | {r['tipo']} | {fmt_s(r['ttft_s'])} | {fmt_s(r['total_s'])} | "
                     f"{'sí' if r['rechazo'] else 'no'} | {'sí' if r['cita_fuente'] else 'no'} | {cov} | {ini} |")
        L.append("")

    (REPORTS_DIR / f"eval_{tag}.md").write_text("\n".join(L), encoding="utf-8")
    return data


def main():
    ap = argparse.ArgumentParser(description="Evaluación del RAG de AulaIA")
    ap.add_argument("--tag", default="actual", help="nombre del reporte (ej. antes / despues)")
    ap.add_argument("--e2e", action="store_true", help="evaluar también las respuestas completas de Gemini")
    ap.add_argument("--pausa", type=float, default=4.0, help="segundos entre llamadas e2e (límite de cuota)")
    args = ap.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    engine = RAGEngine()
    if engine.vector_store.index is None:
        engine.vector_store.load()

    print(f"\n=== Evaluación RAG AulaIA [{args.tag}] — {len(cases)} casos ===\n")
    ret_rows = eval_retrieval(engine, cases)
    e2e_rows = None
    if args.e2e:
        print("Ejecutando respuestas completas con Gemini...")
        e2e_rows = eval_e2e(engine, cases, args.pausa)

    data = write_report(args.tag, engine, ret_rows, e2e_rows)
    print(f"\nEmbeddings: {data['embeddings']}")
    for grupo in ["global", "directa", "parafraseada", "seguimiento"]:
        s = data["recuperacion"][grupo]
        print(f"[{grupo:<12}] " + " | ".join(
            f"{m}: Hit@3 diapo {fmt_pct(s[m]['hit3_diapo'])}, MRR {s[m]['mrr_diapo']:.2f}" for m in MODES))
    if "e2e" in data:
        print("e2e:", json.dumps(data["e2e"], ensure_ascii=False))
    print(f"\nReporte: tests/reports/eval_{args.tag}.md\n")


if __name__ == "__main__":
    main()
