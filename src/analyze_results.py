"""Analiza los resultados de baseline.py: tablas y tests estadísticos.

Lee todos los archivos baseline_<modelo>_<split>.jsonl de una carpeta y genera:
  - Acierto global y por examen (FIR, BIR), con intervalo de confianza del 95 %.
  - Sesgo de posición: cuántas veces elige cada opción frente a cuántas es la correcta.
  - Test de McNemar entre cada par de modelos (solo por pantalla, si hay 2 o más).

No necesita GPU: se ejecuta en local sobre los .jsonl descargados de Drive.

Uso:  python src/analyze_results.py --results results_final --split test
Salida (en --out, por defecto outputs/analisis):
  acierto_<split>.csv, sesgo_posicion_<split>.csv
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
EXAMS = {"Global": None, "BIR": "biology", "FIR": "pharmacology"}
KEY = ["year", "category", "qid"]


def load_results(results_dir, split):
    """{nombre del modelo: DataFrame} con todos los baseline_*_<split>.jsonl."""
    results = {}
    for f in sorted(results_dir.glob(f"baseline_*_{split}.jsonl")):
        model = f.stem.removeprefix("baseline_").removesuffix(f"_{split}")
        results[model] = pd.read_json(f, lines=True)
    if not results:
        raise SystemExit(f"No hay archivos baseline_*_{split}.jsonl en {results_dir}")
    return results


def accuracy_table(results):
    """Acierto e intervalo de confianza del 95 % (aproximación normal) por modelo y examen."""
    rows = []
    for model, r in results.items():
        for exam, cat in EXAMS.items():
            s = r["correct"] if cat is None else r.loc[r["category"] == cat, "correct"]
            p, n = s.mean(), len(s)
            rows.append({"modelo": model, "examen": exam, "preguntas": n,
                         "acierto": p, "ic95": 1.96 * np.sqrt(p * (1 - p) / n)})
    return pd.DataFrame(rows)


def position_bias_table(results):
    """Por modelo y opción: veces que la elige, veces que es la correcta y acierto."""
    rows = []
    for model, r in results.items():
        for opt in sorted(set(r["ra"]) | set(r["pred"])):
            rows.append({"modelo": model, "opcion": opt,
                         "elegida": int((r["pred"] == opt).sum()),
                         "correcta": int((r["ra"] == opt).sum()),
                         "acierto_si_es_correcta": r.loc[r["ra"] == opt, "correct"].mean()})
    return pd.DataFrame(rows)


def mcnemar_table(results):
    """Test de McNemar exacto entre cada par de modelos, sobre las preguntas comunes."""
    rows = []
    for a, b in combinations(results, 2):
        m = results[a].merge(results[b], on=KEY, suffixes=("_a", "_b"))
        only_a = int((m["correct_a"] & ~m["correct_b"]).sum())
        only_b = int((~m["correct_a"] & m["correct_b"]).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append({"modelo_a": a, "modelo_b": b, "preguntas": len(m),
                     "ambos_aciertan": int((m["correct_a"] & m["correct_b"]).sum()),
                     "solo_a": only_a, "solo_b": only_b,
                     "ambos_fallan": int((~m["correct_a"] & ~m["correct_b"]).sum()),
                     "p_valor": p})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results_final")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "analisis")
    args = parser.parse_args()

    results = load_results(args.results, args.split)
    args.out.mkdir(parents=True, exist_ok=True)

    acc = accuracy_table(results)
    bias = position_bias_table(results)
    acc.to_csv(args.out / f"acierto_{args.split}.csv", index=False)
    bias.to_csv(args.out / f"sesgo_posicion_{args.split}.csv", index=False)

    pd.set_option("display.width", 120)
    print(f"== Acierto ({args.split}) ==")
    print(acc.assign(acierto=acc["acierto"].map("{:.1%}".format),
                     ic95=acc["ic95"].map("±{:.1%}".format)).to_string(index=False))
    print(f"\n== Sesgo de posición ({args.split}) ==")
    print(bias.assign(acierto_si_es_correcta=bias["acierto_si_es_correcta"].map("{:.0%}".format))
          .to_string(index=False))
    if len(results) > 1:
        mcn = mcnemar_table(results)
        print(f"\n== McNemar ({args.split}) ==")
        print(mcn.assign(p_valor=mcn["p_valor"].map("{:.2g}".format)).to_string(index=False))
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
