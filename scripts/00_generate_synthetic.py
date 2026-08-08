"""Genera ejemplos sinteticos a partir de las semillas y los filtra con un juez LLM.

Este script es la evidencia de "data curation + synthetic data generation" que pide
la JD de Assail. Lo importante para una entrevista NO es el dataset resultante, sino
los dos prompts (generador y juez) y la tasa de rechazo del juez: eso demuestra que
entiendes el ciclo generar -> filtrar, no solo que corriste un tutorial.

Uso:
    pip install google-genai
    export GEMINI_API_KEY=...   # gratis en https://aistudio.google.com/apikey
    python 00_generate_synthetic.py --n 300 --out ../data/synthetic.jsonl
"""

import argparse
import json
import pathlib
import random
import time

from google import genai
from google.genai import errors, types

MODEL = "gemini-2.5-flash"

OWASP_CATEGORIES = [
    "A01:2021-Broken Access Control",
    "A02:2021-Cryptographic Failures",
    "A03:2021-Injection",
    "A04:2021-Insecure Design",
    "A05:2021-Security Misconfiguration",
    "A06:2021-Vulnerable and Outdated Components",
    "A07:2021-Identification and Authentication Failures",
    "A08:2021-Software and Data Integrity Failures",
    "A09:2021-Security Logging and Monitoring Failures",
    "A10:2021-Server-Side Request Forgery",
]

SEVERITIES = ["critical", "high", "medium", "low", "informational"]

EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "owasp_category": {"type": "string", "enum": OWASP_CATEGORIES},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "is_false_positive": {"type": "boolean"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "finding",
                    "owasp_category",
                    "severity",
                    "is_false_positive",
                    "rationale",
                ],
            },
        }
    },
    "required": ["examples"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label_is_correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["label_is_correct", "reason"],
}

GENERATOR_PROMPT = """You are helping build a training dataset for a security finding
triage model. Given the example findings below, write {n} NEW findings in the same
style and register.

Rules:
- Each finding is a short, neutral description of what a scanner or a code review
  reported. Describe the observation, never write a working exploit payload.
- About one third of the examples must be FALSE POSITIVES: cases where the scanner
  flagged something but a control is already in place, the code path is unreachable,
  or the check is misapplied. These are the most valuable examples, so make the
  reasoning genuinely subtle rather than obvious.
- Vary the OWASP category. Do not cluster on injection.
- The rationale must explain WHY the label holds, in one or two sentences.
- Do not copy the seed examples. Write new scenarios.

Seed examples:
{seeds}
"""

JUDGE_PROMPT = """You are auditing one labelled example from a security triage dataset.
Decide whether the label is defensible for a security engineer.

Reject the example if any of the following is true:
- The OWASP category does not match the described weakness.
- The is_false_positive flag contradicts the finding text or the rationale.
- The severity is clearly wrong for the described impact.
- The finding is vague enough that no confident label is possible.

Be strict. A noisy training set is worse than a small one.

Finding: {finding}
Category: {owasp_category}
Severity: {severity}
Is false positive: {is_false_positive}
Rationale: {rationale}
"""


def call_json(client, prompt, schema):
    """Llama a Gemini con salida JSON forzada al esquema. Reintenta ante limite de tasa,
    que en la capa gratuita se alcanza rapido (~10 peticiones/minuto)."""
    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return json.loads(response.text)
        except errors.APIError as e:
            if e.code == 429 and attempt < 5:
                wait = 15 * (attempt + 1)
                print(f"  limite de tasa, esperando {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate_batch(client, seeds, n):
    seed_text = "\n".join(
        json.dumps(s, ensure_ascii=False) for s in random.sample(seeds, min(6, len(seeds)))
    )
    prompt = GENERATOR_PROMPT.format(n=n, seeds=seed_text)
    return call_json(client, prompt, EXAMPLE_SCHEMA)["examples"]


def judge(client, example):
    return call_json(client, JUDGE_PROMPT.format(**example), VERDICT_SCHEMA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="ejemplos a generar")
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--seeds", default="../data/seed_examples.jsonl")
    parser.add_argument("--out", default="../data/synthetic.jsonl")
    args = parser.parse_args()

    here = pathlib.Path(__file__).parent
    seeds = [json.loads(line) for line in (here / args.seeds).open()]
    client = genai.Client()  # lee GEMINI_API_KEY (o GOOGLE_API_KEY) del entorno

    kept, rejected = [], 0
    while len(kept) < args.n:
        for example in generate_batch(client, seeds, args.batch):
            verdict = judge(client, example)
            if verdict["label_is_correct"]:
                kept.append(example)
            else:
                rejected += 1
        print(f"aceptados {len(kept)} / rechazados por el juez {rejected}")

    out_path = here / args.out
    with out_path.open("w") as f:
        for example in kept[: args.n]:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    total = len(kept) + rejected
    print(f"\n{len(kept[:args.n])} ejemplos escritos en {out_path}")
    print(f"tasa de rechazo del juez: {rejected / total:.1%}  <- este numero va en el README")


if __name__ == "__main__":
    main()
