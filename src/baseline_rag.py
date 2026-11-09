"""Baseline con RAG: el mismo LLM, el mismo prompt y la misma forma de responder que
src/baseline.py, pero con los k chunks recuperados del corpus delante del prompt.

Para cada pregunta:
  1. Se recuperan los k chunks más parecidos a la pregunta con uno de los buscadores
     de src/coverage.py (por defecto, el híbrido: BM25 + denso con RRF).
  2. Se construye el prompt de baseline.py y se le antepone el contexto.
  3. Como en baseline.py, se elige la opción ("1", "2"...) más probable como
     siguiente token.

Lo único que cambia respecto a baseline.py es el contexto: la muestra (--n, misma
semilla), el modelo, la cuantización y la forma de puntuar son las mismas. El
fichero de salida empieza por "baseline_", así que analyze_results.py lo trata como
un modelo más (acierto, sesgo de posición y McNemar frente al baseline sin RAG).

Uso:
  python src/baseline_rag.py --split dev
  python src/baseline_rag.py --split dev --n 200 --k 10
  python src/baseline_rag.py --split dev --retriever bm25 --corpus libretexts
Salida:
  <out>/baseline_<modelo>+rag-<retriever>-<corpus>-k<k>_<split>.jsonl  (una fila por pregunta)
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from tqdm import tqdm

from baseline import DEFAULT_MODEL, SEED, build_prompt, load_model, option_token_ids
from coverage import (CORPUS_DIR, MODELO_DENSO, corpus_disponibles, crear_buscador, leer_jsonl,
                      normalizar, presente, respuesta_correcta)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEY = ["year", "category", "qid"]


def recuperar(buscador, consultas, k):
    """Índices de los k chunks más puntuados para cada consulta, de mejor a peor."""
    tops = []
    for _, puntuaciones in buscador.puntuar(consultas):
        for fila in puntuaciones:
            top = np.argpartition(-fila, k)[:k] if len(fila) > k else np.arange(len(fila))
            tops.append(top[np.argsort(-fila[top])].tolist())
    return tops


def build_rag_prompt(row, chunks) -> str:
    """El prompt de baseline.py con los chunks recuperados delante."""
    contexto = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(chunks, 1))
    return (
        "Contexto (fragmentos recuperados de libros y fichas técnicas; puede que no todos sean útiles):\n\n"
        f"{contexto}\n\n{build_prompt(row)}"
    )


@torch.no_grad()
def predict(prompt, n_opciones, tokenizer, model, option_ids):
    """Igual que baseline.predict, pero recibe el prompt ya construido."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]  # logits del siguiente token

    probs = torch.softmax(logits[option_ids[:n_opciones]].float(), dim=0).tolist()
    pred = probs.index(max(probs)) + 1
    return pred, probs


def comparar_con_baseline(results, base_file):
    """Acierto sin RAG vs. con RAG sobre las mismas preguntas, si existe el baseline."""
    if not base_file.exists():
        print(f"\n(No está {base_file.name}: ejecuta baseline.py con el mismo --split, --n y --out para comparar)")
        return
    base = pd.read_json(base_file, lines=True)[KEY + ["correct"]]
    m = results.merge(base, on=KEY, suffixes=("_rag", "_sin_rag"))
    if len(m) < len(results):
        print(f"\n(El baseline solo comparte {len(m)} de {len(results)} preguntas: usa el mismo --n)")
    m["examen"] = m["category"].map({"biology": "BIR", "pharmacology": "FIR"})
    m["contexto"] = m["correcta_en_contexto"].map({True: "correcta en contexto", False: "correcta no en contexto"})

    print(f"\n== Sin RAG vs. con RAG ({len(m)} preguntas comunes) ==")
    cols = ["correct_sin_rag", "correct_rag"]
    for por in ["examen", "contexto"]:
        tabla = m.groupby(por)[cols].mean()
        tabla["preguntas"] = m.groupby(por).size()
        print(tabla.round(3).to_string(), "\n")
    print(f"total: sin RAG {m['correct_sin_rag'].mean():.3f} | con RAG {m['correct_rag'].mean():.3f}")

    solo_rag = int((m["correct_rag"] & ~m["correct_sin_rag"]).sum())
    solo_sin = int((~m["correct_rag"] & m["correct_sin_rag"]).sum())
    p = binomtest(solo_rag, solo_rag + solo_sin, 0.5).pvalue if solo_rag + solo_sin else 1.0
    print(f"McNemar: aciertos solo con RAG {solo_rag}, solo sin RAG {solo_sin}, p = {p:.2g}")


def main():
    disponibles = corpus_disponibles()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--n", type=int, help="usar solo una muestra aleatoria de n preguntas")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", nargs="+", choices=disponibles, default=disponibles,
                        help="corpus de data/corpus/ (por defecto, todos)")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--modelo-denso", default=MODELO_DENSO, help="embeddings para dense o hybrid")
    parser.add_argument("--k", type=int, default=5, help="chunks de contexto por pregunta")
    parser.add_argument("--aleatorio", action="store_true",
                        help="control: k chunks al azar en vez de los recuperados")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    df = pd.read_json(DATA_DIR / f"{args.split}.jsonl", lines=True)
    if args.n:
        df = df.sample(n=min(args.n, len(df)), random_state=SEED)  # misma muestra que baseline.py

    # 1) Recuperación para todas las preguntas, antes de cargar el LLM (que ocupa la GPU)
    nombre = "+".join(args.corpus)
    corpus = {c: leer_jsonl(CORPUS_DIR / f"{c}.jsonl") for c in args.corpus}
    docs, buscador = crear_buscador(args.retriever, args.modelo_denso, nombre, args.corpus, corpus)
    print("  Recuperando chunks...")
    tops = recuperar(buscador, df["qtext"].tolist(), args.k)

    # Control: k chunks al azar en vez de los recuperados
    if args.aleatorio:
        rng = np.random.default_rng(SEED)
        tops = [rng.choice(len(docs), args.k, replace=False).tolist() for _ in range(len(df))]

    # Similitud coseno (embeddings) de cada chunk usado con la pregunta
    denso = buscador.buscadores[1] if args.retriever == "hybrid" else buscador
    sims = None
    if args.retriever != "bm25":
        q = denso.codificar(df["qtext"].tolist(), "query: ")
        sims = [[float(denso.emb[i] @ q[iq]) for i in top] for iq, top in enumerate(tops)]
    del buscador, denso
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Respuesta del LLM con los chunks como contexto
    tokenizer, model = load_model(args.model)
    option_ids = option_token_ids(tokenizer)

    rows = []
    for (_, row), top in tqdm(zip(df.iterrows(), tops), total=len(df)):
        chunks = [docs[i] for i in top]
        pred, probs = predict(build_rag_prompt(row, chunks), len(row["answers"]),
                              tokenizer, model, option_ids)
        contexto = set().union(*(normalizar(f"{p['title']} {p['text']}") for p in chunks))
        rows.append({
            "year": row["year"], "category": row["category"], "qid": row["qid"],
            "ra": row["ra"], "pred": pred, "correct": pred == row["ra"], "probs": probs,
            "chunks": [p["id"] for p in chunks],
            "sims": sims[len(rows)] if sims else None,
            "correcta_en_contexto": presente(normalizar(respuesta_correcta(row)), contexto),
        })
    results = pd.DataFrame(rows)

    modelo = args.model.split("/")[-1]
    args.out.mkdir(parents=True, exist_ok=True)
    etiqueta = f"k{args.k}" + ("-aleatorio" if args.aleatorio else "")
    out_file = args.out / f"baseline_{modelo}+rag-{args.retriever}-{nombre}-{etiqueta}_{args.split}.jsonl"
    results.to_json(out_file, orient="records", lines=True)

    print(f"\nModelo: {args.model} | RAG: {args.retriever} sobre {nombre}, k={args.k}"
          f" | Split: {args.split} | Preguntas: {len(results)}")
    print(f"Precisión global: {results['correct'].mean():.3f}")
    for cat, acc in results.groupby("category")["correct"].mean().items():
        print(f"  {cat}: {acc:.3f}")
    print(f"Correcta en el contexto recuperado: {results['correcta_en_contexto'].mean():.3f}")
    comparar_con_baseline(results, args.out / f"baseline_{modelo}_{args.split}.jsonl")
    print(f"Guardado en {out_file}")


if __name__ == "__main__":
    main()
