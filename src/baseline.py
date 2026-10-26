"""Baseline sin RAG: el LLM responde las preguntas solo con lo que "sabe".

Para cada pregunta se construye un prompt con la pregunta y sus opciones, y en
lugar de generar texto se mira la probabilidad que da el modelo a que el
siguiente token sea "1", "2", "3"... Se elige la opción más probable.
Así el resultado es determinista y nunca falla al leer la respuesta.

Uso:
  python src/baseline.py --split dev --n 200
  python src/baseline.py --split test --out /content/drive/MyDrive/TAPLN/results
Salida:
  <out>/baseline_<modelo>_<split>.jsonl  (una fila por pregunta)
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42


def build_prompt(row) -> str:
    """Pregunta + opciones numeradas. En el RAG se añadirá aquí el contexto."""
    options = "\n".join(f"{a['aid']}. {a['atext']}" for a in row["answers"])
    return (
        "Responde a la siguiente pregunta de examen eligiendo una opción.\n\n"
        f"Pregunta: {row['qtext']}\n{options}\n\n"
        "Responde solo con el número de la opción correcta."
    )


def load_model(model_id):
    """Con GPU (Colab) carga el modelo en 4 bits; sin GPU, en precisión normal (para pruebas)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
    else:
        print("AVISO: no hay GPU, el modelo se carga en CPU (solo sirve para pruebas con modelos pequeños)")
        model = AutoModelForCausalLM.from_pretrained(model_id)
    print(f"Modelo cargado en: {model.device}")
    model.eval()
    return tokenizer, model


def option_token_ids(tokenizer, max_options=5):
    """Id del token de cada número de opción ("1" -> id, "2" -> id...)."""
    ids = []
    for i in range(1, max_options + 1):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        assert len(tokens) == 1, f"El número {i} no es un único token en este tokenizer"
        ids.append(tokens[0])
    return ids


@torch.no_grad()
def predict(row, tokenizer, model, option_ids):
    """Devuelve la opción elegida y la probabilidad de cada opción."""
    messages = [{"role": "user", "content": build_prompt(row)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]  # logits del siguiente token

    n = len(row["answers"])  # 4 opciones, o 5 en dev 2013-2014
    probs = torch.softmax(logits[option_ids[:n]].float(), dim=0).tolist()
    pred = probs.index(max(probs)) + 1
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--n", type=int, help="usar solo una muestra aleatoria de n preguntas")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    df = pd.read_json(DATA_DIR / f"{args.split}.jsonl", lines=True)
    if args.n:
        df = df.sample(n=min(args.n, len(df)), random_state=SEED)

    tokenizer, model = load_model(args.model)
    option_ids = option_token_ids(tokenizer)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pred, probs = predict(row, tokenizer, model, option_ids)
        rows.append({
            "year": row["year"], "category": row["category"], "qid": row["qid"],
            "ra": row["ra"], "pred": pred, "correct": pred == row["ra"], "probs": probs,
        })
    results = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"baseline_{args.model.split('/')[-1]}_{args.split}.jsonl"
    results.to_json(out_file, orient="records", lines=True)

    print(f"\nModelo: {args.model} | Split: {args.split} | Preguntas: {len(results)}")
    print(f"Precisión global: {results['correct'].mean():.3f}")
    for cat, acc in results.groupby("category")["correct"].mean().items():
        print(f"  {cat}: {acc:.3f}")
    print(f"Guardado en {out_file}")


if __name__ == "__main__":
    main()
