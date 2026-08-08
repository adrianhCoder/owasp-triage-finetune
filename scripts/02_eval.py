"""Evalua el modelo sobre el test set: linea base vs fine-tuneado.

    python 02_eval.py --base                      # ANTES de entrenar
    python 02_eval.py --adapter ./qwen-owasp-triage-lora   # DESPUES

Metricas (las cuatro van al README del repo):
  1. json_valid      — % de salidas parseables. Suele ser la mejora mas visible.
  2. category_acc    — accuracy de la categoria OWASP.
  3. fp_precision/recall — la clase falso-positivo, que es la que le importa a Assail.
  4. macro_f1        — F1 macro sobre las categorias.
"""

import argparse
import json
import pathlib
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a security finding triage assistant. Given a raw security finding, "
    "respond with a single JSON object and nothing else, using exactly these keys: "
    "owasp_category, severity, is_false_positive, rationale."
)


def load_model(adapter):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def predict(tokenizer, model, finding):
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


def macro_f1(per_class):
    scores = []
    for tp, fp, fn in per_class.values():
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="ruta de los adaptadores LoRA")
    parser.add_argument("--base", action="store_true", help="evaluar el modelo base")
    parser.add_argument("--test", default="../data/test.jsonl")
    args = parser.parse_args()

    if not args.base and not args.adapter:
        raise SystemExit("pasa --base o --adapter")

    here = pathlib.Path(__file__).parent
    rows = [json.loads(line) for line in (here / args.test).open()]
    tokenizer, model = load_model(args.adapter)

    valid = cat_hits = 0
    fp_tp = fp_fp = fp_fn = 0
    per_class = defaultdict(lambda: [0, 0, 0])  # categoria -> [tp, fp, fn]

    for row in rows:
        pred, raw = predict(tokenizer, model, row["finding"])
        if pred is None:
            print(f"  [no parseable] {raw[:90]!r}")
            continue
        valid += 1

        gold_cat, pred_cat = row["owasp_category"], pred.get("owasp_category")
        if pred_cat == gold_cat:
            cat_hits += 1
            per_class[gold_cat][0] += 1
        else:
            per_class[gold_cat][2] += 1
            if pred_cat:
                per_class[pred_cat][1] += 1

        gold_fp, pred_fp = row["is_false_positive"], bool(pred.get("is_false_positive"))
        if gold_fp and pred_fp:
            fp_tp += 1
        elif pred_fp and not gold_fp:
            fp_fp += 1
        elif gold_fp and not pred_fp:
            fp_fn += 1

    n = len(rows)
    label = "BASE" if args.base else f"TUNED ({args.adapter})"
    print(f"\n=== {label} · {n} ejemplos ===")
    print(f"json_valid      {valid / n:.1%}")
    print(f"category_acc    {cat_hits / n:.1%}")
    print(f"macro_f1        {macro_f1(per_class):.3f}")
    if fp_tp + fp_fp:
        print(f"fp_precision    {fp_tp / (fp_tp + fp_fp):.1%}")
    if fp_tp + fp_fn:
        print(f"fp_recall       {fp_tp / (fp_tp + fp_fn):.1%}")


if __name__ == "__main__":
    main()
