"""SFT con LoRA sobre un modelo chico, para triage de hallazgos OWASP.

Pensado para correr en Colab gratis (GPU T4). Pegar en una celda o subir el repo.

    !pip install -q "transformers>=4.44" "trl>=0.12" "peft>=0.13" datasets accelerate bitsandbytes
    !python 01_train_sft_lora.py

⚠️ ORDEN CORRECTO: correr 02_eval.py con --base ANTES de entrenar. Sin la linea base
no puedes decir "paso de X a Y", y esa frase es la mitad del valor del proyecto.
"""

import json
import pathlib

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # bajar a 0.5B si la T4 se queda sin VRAM
OUTPUT_DIR = "./qwen-owasp-triage-lora"

SYSTEM_PROMPT = (
    "You are a security finding triage assistant. Given a raw security finding, "
    "respond with a single JSON object and nothing else, using exactly these keys: "
    "owasp_category, severity, is_false_positive, rationale."
)


def load_examples():
    here = pathlib.Path(__file__).parent
    rows = []
    for name in ("seed_examples.jsonl", "synthetic.jsonl"):
        path = here / ".." / "data" / name
        if path.exists():
            rows += [json.loads(line) for line in path.open()]
    if not rows:
        raise SystemExit("no hay datos en ../data/ — corre 00_generate_synthetic.py primero")
    return rows


def to_chat(example):
    answer = {
        "owasp_category": example["owasp_category"],
        "severity": example["severity"],
        "is_false_positive": example["is_false_positive"],
        "rationale": example["rationale"],
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["finding"]},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    }


def main():
    rows = load_examples()
    dataset = Dataset.from_list([to_chat(r) for r in rows])

    # Split estratificado barato: shuffle con semilla fija y corte 80/10/10.
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    holdout = splits["test"].train_test_split(test_size=0.5, seed=42)
    train_ds, val_ds, test_ds = splits["train"], holdout["train"], holdout["test"]

    here = pathlib.Path(__file__).parent
    test_ds.to_json(here / ".." / "data" / "test.jsonl")
    print(f"train {len(train_ds)} · val {len(val_ds)} · test {len(test_ds)} (test guardado)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # la T4 no soporta bf16
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_length=1024,
        fp16=True,          # bf16=True truena en T4
        report_to="none",
        assistant_only_loss=True,  # entrena solo sobre la respuesta, no sobre el prompt
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nadaptadores LoRA guardados en {OUTPUT_DIR}")
    print("siguiente paso: python 02_eval.py --adapter", OUTPUT_DIR)


if __name__ == "__main__":
    main()
