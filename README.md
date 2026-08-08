# OWASP Finding Triage — LoRA fine-tuning

Fine-tuning de un modelo de lenguaje pequeño para hacer **triage de hallazgos de seguridad**:
dado un hallazgo crudo (salida de un scanner, una observación de revisión de código), el
modelo devuelve JSON estructurado con la categoría OWASP, la severidad, **si es un falso
positivo**, y la justificación.

```json
{
  "owasp_category": "A03:2021-Injection",
  "severity": "critical",
  "is_false_positive": false,
  "rationale": "The single quote reaches the SQL parser and the raw driver error is reflected to the client, confirming unsanitized input concatenated into a query."
}
```

Entrenado con **LoRA** sobre `Qwen2.5-1.5B-Instruct` usando Hugging Face TRL y PEFT, en una
sola GPU T4 de Colab gratis. El dataset se construyó a mano y se amplió con **generación
sintética filtrada por un juez LLM**. Se mide línea base contra modelo entrenado sobre un
conjunto de prueba intocado.

---

## 1. Por qué esta tarea y no un clasificador cualquiera

Tres razones, en orden de importancia.

**La detección de falsos positivos es el problema real.** Un scanner de seguridad produce
cientos de alertas y la mayor parte del trabajo humano se va en descartar las que no son
explotables: la cabecera que "falta" pero está cubierta por otra política, la dependencia
vulnerable que solo vive en las herramientas de desarrollo y nunca llega a producción, el
error 500 que viene de una validación de tipo y no del parser de SQL. Ese descarte es
exactamente lo que consume el tiempo de un equipo de seguridad, y es una tarea de
clasificación con matiz, no de reglas.

**La salida estructurada lo convierte en un componente, no en un juguete.** El modelo
devuelve JSON con un esquema fijo, que es el formato que un agente consume como resultado
de una herramienta. No es un clasificador aislado: es una pieza que encaja en un pipeline.

**El dominio permite curar la data con criterio.** Etiquetar estos ejemplos requiere saber
distinguir un hallazgo real de una alerta de scanner. Un dataset genérico sacado de internet
no enseña eso.

---

## 2. Qué cubre este proyecto, técnicamente

| Área | Cómo se cubre aquí |
|---|---|
| **SFT** (supervised fine-tuning) | `SFTTrainer` de TRL sobre pares prompt → respuesta curados |
| **LoRA / PEFT** | Adaptadores de bajo rango (`r=16`, `alpha=32`) sobre el modelo base congelado |
| **Data curation** | 26 ejemplos semilla escritos a mano, balanceados por categoría y por clase |
| **Synthetic data generation** | Un LLM fuerte genera variaciones; un segundo pase de **juez** descarta las mal etiquetadas |
| **Diseño de evals** | Línea base antes de entrenar, mismas métricas después, conjunto de prueba intocado |
| **Salida estructurada** | Esquema JSON fijo, con la validez del JSON como métrica de primera clase |

---

## 3. El dataset

### Semillas (`data/seed_examples.jsonl`)

26 ejemplos escritos a mano. Distribución verificada:

- **Las 10 categorías del OWASP Top 10 (2021)** están representadas.
- **27% son falsos positivos** (7 de 26), deliberadamente.
- Severidades repartidas entre `critical`, `high`, `medium`, `low` e `informational`.

Los falsos positivos son la parte valiosa del dataset y están escritos para que el matiz sea
genuino, no obvio. Algunos ejemplos de los que están incluidos:

- Un scanner reporta falta de `X-Frame-Options`, pero todas las respuestas llevan una CSP con
  `frame-ancestors: none`, que en navegadores modernos lo reemplaza. La protección existe; el
  check está desactualizado.
- Un scanner de dependencias reporta una vulnerabilidad alta en un paquete que solo aparece en
  `devDependencies` y nunca entra al bundle de producción. El aviso es real, la superficie de
  ataque no existe.
- Un endpoint que recibe una URL se marca como SSRF, pero el handler valida el host contra una
  lista blanca y rechaza con 400 antes de hacer cualquier petición saliente.

Los hallazgos son **descripciones de observaciones**, nunca payloads funcionales. El modelo
aprende a clasificar y justificar, no a explotar.

### Ampliación sintética (`data/synthetic.jsonl`)

Generada por `scripts/00_generate_synthetic.py`. El pipeline tiene dos etapas:

1. **Generador**: se le pasan 6 semillas al azar como referencia de estilo y se le piden N
   ejemplos nuevos, con la instrucción explícita de que un tercio sean falsos positivos con
   razonamiento sutil, y de variar la categoría OWASP en vez de amontonarse en inyección.
2. **Juez**: cada ejemplo generado pasa por un segundo modelo que verifica que la categoría
   corresponda a la debilidad descrita, que la bandera de falso positivo no contradiga el
   texto, que la severidad sea razonable, y que el hallazgo sea lo bastante concreto para
   etiquetarse con confianza. Rechaza si algo falla.

Lo importante de esta etapa no es el archivo resultante sino **los dos prompts y la tasa de
rechazo del juez**, que el script imprime al terminar. Esa tasa es la evidencia de que el
filtro está haciendo algo y no es decorativo.

---

## 4. Cómo correrlo

### Instalación

```bash
pip install "transformers>=4.44" "trl>=0.12" "peft>=0.13" datasets accelerate bitsandbytes
pip install google-genai       # solo para la generación sintética (Gemini, capa gratuita)
```

### Los cuatro pasos, en orden

```bash
# 1. Ampliar el dataset: 26 semillas -> ~300 ejemplos filtrados por el juez
export GEMINI_API_KEY=...      # gratis en https://aistudio.google.com/apikey
python scripts/00_generate_synthetic.py --n 300

# 2. LÍNEA BASE. Antes de entrenar. Este paso no se salta.
python scripts/02_eval.py --base

# 3. Entrenar con LoRA (Colab T4, aproximadamente 1 a 2 horas)
python scripts/01_train_sft_lora.py

# 4. Evaluar el modelo entrenado con exactamente las mismas métricas
python scripts/02_eval.py --adapter ./qwen-owasp-triage-lora
```

### Por qué el paso 2 es obligatorio

Medir el modelo base **antes** de tocarlo es lo que convierte el proyecto en algo defendible.
Sin esa medición no existe la frase *"la exactitud pasó de X a Y"*, y esa frase es la mitad
del valor del ejercicio. Un modelo entrenado sin línea base no demuestra nada: no se sabe si
mejoró, si empeoró, o si el modelo base ya lo hacía bien con un buen prompt.

El script de entrenamiento guarda el conjunto de prueba en `data/test.jsonl` con una semilla
fija, de modo que las dos evaluaciones corren exactamente sobre los mismos ejemplos.

---

## 5. Métricas

`scripts/02_eval.py` reporta cuatro cosas, y cada una responde una pregunta distinta:

**`json_valid`** — porcentaje de salidas que se pueden parsear como JSON. Suele ser la mejora
más grande y más visible del fine-tuning: un modelo base pequeño divaga, agrega texto antes y
después, o se olvida de cerrar la llave. Después de entrenar, obedece el formato.

**`category_acc`** — exactitud sobre la categoría OWASP. La métrica obvia.

**`macro_f1`** — F1 macro sobre las categorías. Importa porque las clases están
desbalanceadas: sin esto, un modelo que siempre dice "Injection" se vería mejor de lo que es.

**`fp_precision` y `fp_recall`** — precisión y recall específicamente sobre la clase falso
positivo, reportadas por separado. Es la métrica que de verdad importa para el caso de uso.
Los dos errores tienen costos muy distintos: marcar como falso positivo algo que sí era
explotable es mucho peor que lo contrario.

### Resultados

| Métrica | Base | Fine-tuneado | Δ |
|---|---|---|---|
| `json_valid` | | | |
| `category_acc` | | | |
| `macro_f1` | | | |
| `fp_precision` | | | |
| `fp_recall` | | | |

*Tabla pendiente de llenar con la salida real de los pasos 2 y 4.*

---

## 6. Configuración y detalles que ahorran horas

**Modelo base**: `Qwen/Qwen2.5-1.5B-Instruct`. Se usa la variante *instruct* y no la base
porque ya sigue formato de conversación, así que aprende el esquema JSON con muchos menos
ejemplos. Si la T4 se queda sin VRAM, bajar a `Qwen2.5-0.5B-Instruct` cambiando una constante.

**La T4 de Colab no soporta bf16.** Los scripts usan `fp16=True`. Cambiarlo a `bf16` hace
que el entrenamiento falle. Es el error número uno al seguir tutoriales escritos para GPUs
más nuevas.

**Se entrena solo sobre la respuesta**, no sobre el prompt (`assistant_only_loss=True`). Sin
esto, el modelo gasta capacidad aprendiendo a reproducir los hallazgos de entrada en vez de
aprender a clasificarlos.

**LoRA**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, aplicado a las proyecciones de
atención (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Tres épocas, learning rate `2e-4` con
scheduler coseno.

**El split** es 80/10/10 con semilla fija en 42, para que sea reproducible entre corridas.

---

## 7. Estructura del repositorio

```
owasp-triage-finetune/
├── README.md
├── data/
│   ├── seed_examples.jsonl     ← 26 ejemplos escritos a mano
│   ├── synthetic.jsonl         ← generado por el paso 1
│   └── test.jsonl              ← guardado por el paso 3, intocado
└── scripts/
    ├── 00_generate_synthetic.py  ← generador + juez LLM
    ├── 01_train_sft_lora.py      ← SFT con LoRA (TRL + PEFT)
    └── 02_eval.py                ← línea base vs fine-tuneado
```

---

## 8. Límites honestos

Esto importa tanto como los resultados. Lo que este proyecto **no** es:

- **No es un pipeline de entrenamiento en producción.** Es un proyecto personal que corre en
  Colab. No hay orquestación, ni versionado de datasets, ni reentrenamiento programado.
- **No involucra entrenamiento multi-GPU.** No se usa DeepSpeed ni FSDP, porque un modelo de
  1.5B con LoRA cabe en una sola GPU y no hay forma honesta de ejercitar sharding aquí.
- **El dataset es chico y en parte sintético.** Sirve para demostrar el ciclo completo de
  extremo a extremo, no para desplegar un sistema de triage real.
- **No se hizo preference tuning en esta fase.** DPO sobre las justificaciones y servido con
  vLLM más cuantización son extensiones naturales, pero todavía no están hechas.

---

## 9. Extensiones planeadas

En orden, cada una es aproximadamente medio fin de semana:

1. **DPO** sobre pares de justificaciones (una precisa que cita evidencia contra una vaga),
   corriendo `DPOTrainer` sobre el modelo de la fase 1. Lo interesante es medir si la
   exactitud de clasificación se degrada al mejorar el estilo, y reportarlo.
2. **Servido e inferencia**: levantar el modelo con vLLM, cuantizar a 4 bits (AWQ o GGUF) y
   medir throughput y latencia p50/p95 antes y después.
3. **GRPO**, a nivel conceptual: el método de RL por grupos que popularizó DeepSeek, que
   encaja con rewards verificables. No requiere correrse para entenderse.

---

## Glosario rápido

**OWASP Top 10** — la lista de las diez categorías de vulnerabilidad web más críticas,
publicada por la OWASP Foundation. Es el vocabulario estándar del área: decir "A03 Injection"
comunica algo preciso. No confundir con **OSCP**, que es una certificación de pentesting y no
tiene relación con este proyecto.

**SFT** — supervised fine-tuning: seguir entrenando un modelo preentrenado con pares de
prompt y respuesta ideal para que imite ese comportamiento.

**LoRA** — Low-Rank Adaptation: se congelan los pesos del modelo base y solo se entrenan unas
matrices adaptadoras pequeñas, lo que permite hacer fine-tuning en una sola GPU modesta.

**Falso positivo** — en este contexto, un hallazgo que un scanner reporta pero que no
representa una vulnerabilidad explotable, porque ya existe un control, la ruta de código es
inalcanzable, o el check está mal aplicado.
