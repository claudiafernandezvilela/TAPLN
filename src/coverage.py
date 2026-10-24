"""Mide cuánto cubren los corpus de data/corpus/ las preguntas de HEAD-QA (FIR y BIR).

Para cada corpus por separado y para todos juntos ("TODAS") indexa los pasajes con
BM25 (léxico, sin GPU) y calcula:
  correcta_en_contexto      la respuesta correcta aparece en los k pasajes
                            recuperados con la pregunta
  distractores_en_contexto  fracción de opciones incorrectas que aparecen
  solo_correcta             aparece la correcta y ningún distractor
  ir_acierto                baseline IR de HEAD-QA: se elige la opción cuya
                            consulta (pregunta + opción) obtiene el pasaje de mayor
                            puntuación. Compararlo con "azar" (1 / nº de opciones).

Uso:
  python src/coverage.py                                  # todos los corpus, sobre test
  python src/coverage.py --corpus libretexts cima
  python src/coverage.py --split dev --k 10

Entrada:  data/<split>.jsonl y data/corpus/<corpus>.jsonl  ({id, source, title, text})
Salida:   outputs/cobertura/cobertura_<split>_<corpus>.csv          (por pregunta)
          outputs/cobertura/cobertura_<split>_<corpus>_resumen.csv  (medias por examen)
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
OUT_DIR = ROOT / "outputs" / "cobertura"

METRICAS = ["correcta_en_contexto", "distractores_en_contexto", "solo_correcta", "ir_acierto", "azar"]

STOPWORDS = set("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello
aquellos aqui asi aun cada como con contra cual cuales cuando de del desde donde dos el
ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estan estas este
esto estos fue fueron ha han hasta hay la las le les lo los mas me mediante mi muy no nos
o otra otras otro otros para pero poco por porque que quien se sea segun ser si sido sin
sobre son su sus tambien tan tanto te tiene tienen todo todos tras un una unas uno unos y ya
cual siguiente siguientes correcta correcto incorrecta falsa verdadera afirmacion respecto
""".split())


def leer_jsonl(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def respuesta_correcta(q) -> str:
    return next(a["atext"] for a in q["answers"] if a["aid"] == q["ra"])


def corpus_disponibles():
    """Corpus descargados (ficheros .jsonl no vacíos de data/corpus/)."""
    return sorted(r.stem for r in CORPUS_DIR.glob("*.jsonl") if r.stat().st_size > 0)


def normalizar(texto: str):
    """Minúsculas, sin tildes, sin stopwords y con 'stemming' por prefijo (7 letras)."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t[:7] for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, textos, k1=1.2, b=0.75):
        self.vec = CountVectorizer(analyzer=normalizar)
        tf = self.vec.fit_transform(textos).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        filas = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[filas]) * idf[tf.indices]
        self.w = tf.T.tocsr()  # términos x documentos

    def puntuar(self, consultas, lote=128):
        """Genera (índice de la primera consulta, matriz densa de puntuaciones) por lotes."""
        q = self.vec.transform(consultas)
        q.data[:] = 1
        for i in range(0, q.shape[0], lote):
            yield i, (q[i:i + lote] @ self.w).toarray()


def presente(opcion_tokens, contexto_tokens, umbral=0.8):
    """La opción 'aparece' si al menos el 80% de sus términos están en el contexto."""
    if not opcion_tokens:
        return False
    return sum(t in contexto_tokens for t in opcion_tokens) / len(opcion_tokens) >= umbral


def evaluar(nombre, docs, preguntas, k):
    print(f"  Indexando {nombre} ({len(docs)} pasajes)...")
    bm25 = BM25([f"{d['title']} {d['text']}" for d in docs])
    tokens_doc = {}

    def toks(i):
        if i not in tokens_doc:
            tokens_doc[i] = set(normalizar(docs[i]["title"] + " " + docs[i]["text"]))
        return tokens_doc[i]

    filas = []
    # 1) Recuperación con la pregunta: ¿están la correcta / los distractores en el top-k?
    for ini, sc in bm25.puntuar([q["qtext"] for q in preguntas]):
        for j, fila in enumerate(sc):
            q = preguntas[ini + j]
            top = np.argpartition(-fila, k)[:k] if len(fila) > k else np.arange(len(fila))
            top = top[np.argsort(-fila[top])]
            ctx = set().union(*(toks(i) for i in top))
            correcta = presente(normalizar(respuesta_correcta(q)), ctx)
            distr = [presente(normalizar(a["atext"]), ctx) for a in q["answers"] if a["aid"] != q["ra"]]
            filas.append({"year": q["year"], "category": q["category"], "qid": q["qid"],
                          "qtext": q["qtext"], "corpus": nombre,
                          "correcta_en_contexto": correcta,
                          "distractores_en_contexto": float(np.mean(distr)) if distr else 0.0,
                          "solo_correcta": correcta and not any(distr),
                          "top1": docs[top[0]]["title"] if len(top) else "",
                          "max_score": float(fila[top[0]]) if len(top) else 0.0})

    # 2) Baseline IR de HEAD-QA: una consulta por opción, gana la de mayor puntuación
    consultas, quien = [], []
    for iq, q in enumerate(preguntas):
        for a in q["answers"]:
            consultas.append(f"{q['qtext']} {a['atext']}")
            quien.append((iq, a["aid"]))
    mejor = {}
    for ini, sc in bm25.puntuar(consultas):
        for j, m in enumerate(sc.max(axis=1)):
            iq, aid = quien[ini + j]
            if iq not in mejor or m > mejor[iq][0]:
                mejor[iq] = (m, aid)
    for iq, q in enumerate(preguntas):
        filas[iq]["ir_acierto"] = mejor[iq][1] == q["ra"]
        filas[iq]["azar"] = 1 / len(q["answers"])
    return filas


def resumir(df, orden):
    """Medias de cada métrica por corpus y examen (BIR, FIR y total)."""
    resumen = (df.assign(category=df["category"].map({"pharmacology": "FIR", "biology": "BIR"}))
                 .pivot_table(index="corpus", columns="category", values=METRICAS, aggfunc="mean"))
    total = df.groupby("corpus")[METRICAS].mean()
    total.columns = pd.MultiIndex.from_product([total.columns, ["total"]])
    resumen = pd.concat([resumen, total], axis=1).sort_index(axis=1).round(3)
    return resumen.reindex(orden)


def cobertura(nombres, split, k):
    preguntas = leer_jsonl(DATA_DIR / f"{split}.jsonl")
    corpus = {n: leer_jsonl(CORPUS_DIR / f"{n}.jsonl") for n in nombres}
    if len(corpus) > 1:
        corpus["TODAS"] = [d for ds in corpus.values() for d in ds]

    filas = []
    for nombre, docs in corpus.items():
        filas += evaluar(nombre, docs, preguntas, k)
    df = pd.DataFrame(filas)
    resumen = resumir(df, list(corpus))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"cobertura_{split}_{'+'.join(nombres)}"
    df.to_csv(f"{base}.csv", index=False)
    resumen.to_csv(f"{base}_resumen.csv")

    pd.set_option("display.width", 200)
    print(f"\nCobertura sobre {split} ({len(preguntas)} preguntas, top-{k} pasajes):\n")
    for m in METRICAS[:-1]:
        print(f"{m}:")
        print(resumen[m].to_string(), "\n")
    print(f"azar: {df['azar'].mean():.3f}")
    print(f"Guardado en {base.relative_to(ROOT)}.csv y ..._resumen.csv")


def main():
    disponibles = corpus_disponibles()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", nargs="+", choices=disponibles, default=disponibles,
                   help="corpus de data/corpus/ a evaluar (por defecto, todos)")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--k", type=int, default=5, help="pasajes recuperados por pregunta")
    args = p.parse_args()
    if not args.corpus:
        p.error(f"no hay corpus en {CORPUS_DIR.relative_to(ROOT)}; ejecuta antes src/build_corpus.py")
    cobertura(args.corpus, args.split, args.k)


if __name__ == "__main__":
    main()
