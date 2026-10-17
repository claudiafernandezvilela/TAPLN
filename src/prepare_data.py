"""Descarga HEAD-QA v2 (español), lo limpia y lo divide en dev/test.

Pasos:
  1. Descarga el subconjunto en español de HEAD-QA v2.
  2. Quita las preguntas con imagen y las anuladas (ra = 0).
  3. Divide por años: test = 2021-2022, dev = 2013-2020.
  4. Quita del test las preguntas casi idénticas a alguna de dev
     (similitud TF-IDF >= SIM_THRESHOLD) y las guarda aparte, salvo los
     falsos positivos revisados a mano (NOT_DUPLICATES).

Uso:  python src/prepare_data.py
Salida:
  data/dev.jsonl
  data/test.jsonl
  data/test_duplicados.jsonl  (preguntas de test eliminadas, con su gemela de dev)
"""
from pathlib import Path

import pandas as pd
from datasets import Image, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TEST_YEARS = [2021, 2022]
SIM_THRESHOLD = 0.9  # a partir de aquí se consideran preguntas "gemelas"

# Falsos positivos revisados a mano: superan el umbral pero son preguntas distintas
# (con distinta respuesta correcta), así que se mantienen en el test.
# Clave: (year, category, qid), porque qid solo es único dentro de cada examen.
NOT_DUPLICATES = {
    (2022, "pharmacology", 183),  # error de tipo I (dev 2016 pregunta por el tipo II)
    (2022, "chemistry", 157),     # (R,R)-1,2-dibromociclobutano (dev 2019: isómero cis, meso)
}


def question_text(row) -> str:
    """Pregunta + opciones en un único texto, para comparar preguntas."""
    options = " ".join(a["atext"] for a in row["answers"])
    return f"{row['qtext']} {options}".lower()


def remove_near_duplicates(test, dev, threshold):
    """Separa las preguntas de test que tienen una gemela en dev."""
    dev_texts = dev.apply(question_text, axis=1).tolist()
    test_texts = test.apply(question_text, axis=1).tolist()

    vec = TfidfVectorizer().fit(dev_texts + test_texts)
    sim = cosine_similarity(vec.transform(test_texts), vec.transform(dev_texts))

    test = test.copy()
    test["max_sim"] = sim.max(axis=1)
    best = sim.argmax(axis=1)
    test["dev_year"] = dev["year"].to_numpy()[best]
    test["dev_qtext"] = dev["qtext"].to_numpy()[best]

    keys = zip(test["year"], test["category"], test["qid"])
    is_exception = [k in NOT_DUPLICATES for k in keys]
    is_dup = (test["max_sim"] >= threshold) & ~pd.Series(is_exception, index=test.index)
    duplicates = test[is_dup]
    clean_test = test[~is_dup].drop(columns=["max_sim", "dev_year", "dev_qtext"])
    return clean_test, duplicates


def save(df, name):
    df.to_json(DATA_DIR / name, orient="records", lines=True, force_ascii=False)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    ds = load_dataset("alesi12/head_qa_v2", "es", split="train")
    # No decodificar las imágenes: solo necesitamos saber si existen (y así no hace falta Pillow)
    ds = ds.cast_column("image", Image(decode=False))
    df = ds.remove_columns("image").to_pandas()
    df["tiene_imagen"] = [x["image"] is not None for x in ds.select_columns(["image"])]
    df["n_opciones"] = df["answers"].apply(len)

    # Quitar preguntas con imagen y anuladas (ra = 0)
    limpio = df[(~df["tiene_imagen"]) & (df["ra"] > 0)].drop(columns=["tiene_imagen"])

    # División por años
    test = limpio[limpio["year"].isin(TEST_YEARS)]
    dev = limpio[~limpio["year"].isin(TEST_YEARS)]

    # Quitar del test las preguntas casi idénticas a alguna de dev
    test, duplicados = remove_near_duplicates(test, dev, SIM_THRESHOLD)

    save(dev, "dev.jsonl")
    save(test, "test.jsonl")
    save(duplicados, "test_duplicados.jsonl")

    print(f"Total: {len(df)} | Limpio: {len(limpio)} | Dev: {len(dev)} | Test: {len(test)}")
    print(f"Quitadas del test por tener gemela en dev (sim >= {SIM_THRESHOLD}): {len(duplicados)}")
    print(f"Guardado en {DATA_DIR}")


if __name__ == "__main__":
    main()