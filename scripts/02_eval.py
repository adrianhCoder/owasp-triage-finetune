"""Evalua el modelo sobre el test set: linea base vs fine-tuneado.

    python 02_eval.py --base                      # ANTES de entrenar
    python 02_eval.py --adapter ./qwen-owasp-triage-lora   # DESPUES

Metricas (van al README del repo y se guardan en ../results/):
  1. json_valid        — % de salidas parseables. Suele ser la mejora mas visible.
  2. category_acc      — accuracy de la categoria OWASP con la etiqueta exacta.
  3. category_code_acc — accuracy por codigo (A01..A10), sin importar como se escribio la
                         etiqueta. Separa "sabe la categoria" de "sabe el formato".
  4. fp_precision/recall — la clase falso positivo, la que importa para el caso de uso.
  5. macro_f1          — F1 macro sobre las categorias del test set.

Una salida que no se puede parsear cuenta como error en TODAS las metricas: categoria
fallada y "no es falso positivo". Si se saltara, inflaria macro_f1 y fp_recall justo en el
modelo que peor sigue el formato, que es la linea base.
"""

import argparse
import json
import pathlib
import re
from collections import defaultdict

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a security finding triage assistant. Given a raw security finding, "
    "respond with a single JSON object and nothing else, using exactly these keys: "
    "owasp_category, severity, is_false_positive, rationale."
)

OWASP_CODE = re.compile(r"\bA(0[1-9]|10)\b")


def owasp_code(label):
    """'A03:2021-Injection' -> 'A03'. None si no hay un codigo reconocible."""
    match = OWASP_CODE.search(str(label or ""))
    return match.group(0) if match else None


def parse_bool(value):
    """El modelo a veces responde "false" como texto, y bool("false") seria True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "si", "sí", "1")
    return bool(value)


def load_model(adapter):
    # Los imports pesados viven aqui: las funciones de metricas se prueban sin torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"  # transformers 5: dtype, ya no torch_dtype
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def predict(tokenizer, model, finding):
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": finding},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end]), text
    except (ValueError, json.JSONDecodeError):
        return None, text


def macro_f1(per_class, labels):
    """F1 macro sobre las categorias reales del test set (labels)."""
    scores = []
    for label in labels:
        tp, fp, fn = per_class[label]
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def score(rows, preds):
    """Metricas puras. rows: ejemplos con la verdad; preds: dict parseado o None."""
    labels = sorted({row["owasp_category"] for row in rows})
    valid = cat_hits = code_hits = 0
    fp_tp = fp_fp = fp_fn = 0
    per_class = defaultdict(lambda: [0, 0, 0])  # categoria -> [tp, fp, fn]

    for row, pred in zip(rows, preds):
        gold_cat = row["owasp_category"]
        gold_fp = parse_bool(row["is_false_positive"])

        if pred is None:
            # no parseable = error en todo: categoria fallada y "no es falso positivo"
            per_class[gold_cat][2] += 1
            if gold_fp:
                fp_fn += 1
            continue
        valid += 1

        pred_cat = pred.get("owasp_category")
        if pred_cat == gold_cat:
            cat_hits += 1
            per_class[gold_cat][0] += 1
        else:
            per_class[gold_cat][2] += 1
            if pred_cat in labels:
                per_class[pred_cat][1] += 1
        if owasp_code(pred_cat) and owasp_code(pred_cat) == owasp_code(gold_cat):
            code_hits += 1

        pred_fp = parse_bool(pred.get("is_false_positive"))
        if gold_fp and pred_fp:
            fp_tp += 1
        elif pred_fp and not gold_fp:
            fp_fp += 1
        elif gold_fp and not pred_fp:
            fp_fn += 1

    n = len(rows)
    return {
        "n": n,
        "json_valid": valid / n,
        "category_acc": cat_hits / n,
        "category_code_acc": code_hits / n,
        "macro_f1": macro_f1(per_class, labels),
        "fp_precision": fp_tp / (fp_tp + fp_fp) if fp_tp + fp_fp else None,
        "fp_recall": fp_tp / (fp_tp + fp_fn) if fp_tp + fp_fn else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="ruta de los adaptadores LoRA")
    parser.add_argument("--base", action="store_true", help="evaluar el modelo base")
    parser.add_argument("--test", default="../data/test.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluar solo los primeros N ejemplos (prueba rapida)")
    args = parser.parse_args()

    if not args.base and not args.adapter:
        raise SystemExit("pasa --base o --adapter")

    here = pathlib.Path(__file__).parent
    rows = [json.loads(line) for line in (here / args.test).open()]
    if args.limit:
        rows = rows[: args.limit]
    tokenizer, model = load_model(args.adapter)

    preds, detail = [], []
    for row in rows:
        pred, raw = predict(tokenizer, model, row["finding"])
        if pred is None:
            print(f"  [no parseable] {raw[:90]!r}")
        preds.append(pred)
        detail.append({
            "gold_category": row["owasp_category"],
            "gold_fp": row["is_false_positive"],
            "pred": pred,
            "raw": raw[:300],
        })

    m = score(rows, preds)
    label = "BASE" if args.base else f"TUNED ({args.adapter})"
    print(f"\n=== {label} · {m['n']} ejemplos ===")
    for key in ("json_valid", "category_acc", "category_code_acc"):
        print(f"{key:18s}{m[key]:.1%}")
    print(f"{'macro_f1':18s}{m['macro_f1']:.3f}")
    for key in ("fp_precision", "fp_recall"):
        print(f"{key:18s}" + ("n/a" if m[key] is None else f"{m[key]:.1%}"))

    # Se guarda con el detalle por ejemplo: sirve para el analisis de errores del README.
    name = "base" if args.base else "tuned"
    if args.limit:
        name += f"_limit{args.limit}"
    out = here / ".." / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": label, "metrics": m, "examples": detail},
                              ensure_ascii=False, indent=2))
    print(f"\nresultados guardados en {out.resolve()}")


if __name__ == "__main__":
    main()
