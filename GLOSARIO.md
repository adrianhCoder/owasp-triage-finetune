# Glosario — términos del stack de fine-tuning

Decoder de los términos que rodean a este proyecto. Cada entrada trae **qué es** en dos o tres
frases, **el ancla** con trabajo que ya hiciste cuando existe, y **la frase en inglés** para
decirla en voz alta.

Leer en voz alta. Si puedes explicar cada término sin ver el papel, estás listo.

---

## 0. El mapa mental en 30 segundos

Un modelo de lenguaje se construye por capas:

```
Pre-training          →   SFT                 →   Preference tuning        →   Inference
(internet entero,         (imita ejemplos         (aprende cuál respuesta      (servirlo rápido
 millones de dólares,      curados:                es MEJOR:                    y barato:
 no es tu problema)        prompt → respuesta)     RLHF · RLAIF · DPO · GRPO)   vLLM, quantization)
```

Tu mundo actual (prompting, RAG, orquestación, tools, evals) vive **encima de un modelo
congelado**. El fine-tuning **cambia los pesos del modelo**. Son dos mitades distintas del
mismo oficio, y este proyecto es el puente entre las dos.

---

## 1. Las etapas de entrenamiento

### SFT — Supervised Fine-Tuning

Seguir entrenando un modelo ya preentrenado con pares curados de `prompt → respuesta ideal`,
para que **imite** ese comportamiento, estilo o formato. Es la primera etapa de ajuste después
del preentrenamiento.

**Ancla:** tus *golden scenarios* del eval harness son exactamente ese tipo de dato. Tú los
usas para **evaluar**; SFT los usaría para **entrenar**. Ya sabes curar esa data, lo que
faltaba era el paso de entrenamiento, que es lo que hace este proyecto.

> *"Supervised fine-tuning: you keep training the model on curated prompt-response pairs so it
> imitates the demonstrated behavior. It's the first tuning stage after pre-training."*

### DPO — Direct Preference Optimization

Preference tuning simplificado. Le das **pares** de respuestas al mismo prompt, una **elegida**
y una **rechazada**, y una función de pérdida tipo clasificación hace que el modelo prefiera la
buena. **Sin reward model aparte y sin loop de aprendizaje por refuerzo.** Es la alternativa
barata y estable a RLHF (paper de 2023).

> *"DPO trains directly on preference pairs, chosen versus rejected, with a simple loss. No
> separate reward model and no RL loop. It's the cheap, stable alternative to RLHF."*

### RLHF — Reinforcement Learning from Human Feedback

El pipeline clásico, así se alineó ChatGPT. Tres etapas: primero SFT, luego se entrena un
**reward model** con comparaciones humanas ("A es mejor que B"), y finalmente se optimiza el
modelo con **aprendizaje por refuerzo (PPO)** contra ese reward model. Potente pero complejo,
caro e inestable.

> *"The classic three-stage pipeline: SFT, then train a reward model from human comparisons,
> then optimize the policy with PPO against that reward model."*

### RLAIF — Reinforcement Learning from AI Feedback

Igual que RLHF, pero las preferencias las genera **otro modelo actuando de juez** en vez de
humanos. Escala mucho más barato.

**Ancla:** es exactamente el patrón del juez que usa `scripts/00_generate_synthetic.py` en este
repo, aplicado a preferencias en vez de a validación de etiquetas.

> *"Same as RLHF, but an LLM judge provides the preference labels instead of humans. It scales
> much cheaper."*

### GRPO — Group Relative Policy Optimization

Variante de aprendizaje por refuerzo que popularizó **DeepSeek** (DeepSeekMath y R1). Por cada
prompt generas un **grupo** de respuestas, las puntúas con un reward, y usas **el promedio del
grupo como línea base**: cada respuesta se refuerza según qué tan arriba o abajo del promedio
quedó. Elimina el *value model* o *critic* que necesita PPO, así que es más simple y más
eficiente en memoria. Brilla cuando los rewards son **verificables**: ¿pasó el test? ¿funcionó
el exploit?

> *"Group Relative Policy Optimization, the RL method DeepSeek popularized. You sample a group
> of responses per prompt, score them, and use the group average as the baseline, so you don't
> need a separate critic model. It shines with verifiable rewards, which is why it fits
> self-play loops where 'did the exploit actually work' is the reward signal."*

### PPO — Proximal Policy Optimization

El algoritmo de aprendizaje por refuerzo que usa RLHF. Solo hace falta reconocerlo: es el que
GRPO simplifica al quitarle el critic model.

> *"The RL algorithm behind classic RLHF. GRPO is the simplification that drops its critic."*

---

## 2. Cómo se entrena sin un cluster

### LoRA — Low-Rank Adaptation

Congelas los pesos del modelo base y entrenas solo unas **matrices adaptadoras pequeñas de
bajo rango**. Resultado: fine-tuning en una sola GPU modesta. Es lo que hace este proyecto.

> *"You freeze the base weights and train small low-rank adapter matrices, which lets you
> fine-tune on a single modest GPU."*

### QLoRA

Lo mismo que LoRA pero sobre un modelo base **cuantizado a 4 bits**, así que necesita todavía
menos memoria de video.

> *"QLoRA does the same over a 4-bit quantized base, so it needs even less VRAM."*

### Data curation y synthetic data generation

**Curation** es construir, limpiar y filtrar el conjunto de entrenamiento. **Synthetic data
generation** es usar un modelo fuerte para **generar** ejemplos candidatos, que luego filtras
con jueces o reglas. Así consiguen datasets grandes los equipos chicos.

**Ancla:** es literalmente lo que hace el paso 1 de este repo, y el mismo patrón invertido de
tus evals: ahí curas escenarios a mano para medir, aquí los generas a escala para entrenar.

> *"Building and filtering the training set, and using a strong LLM to generate candidate
> examples that get filtered by judges or rules. That's how small teams get large datasets."*

---

## 3. El stack de librerías

| Herramienta | Qué es en una línea | Nivel a declarar |
|---|---|---|
| **PyTorch** | Framework base de tensores y autograd; todo lo demás corre encima | Operar básico |
| **Hugging Face `transformers`** | Modelos preentrenados y tokenizers estándar | Operar |
| **HF `TRL`** | La librería de tuning: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer` | **Operar — es la de este repo** |
| **HF `PEFT`** | Implementa LoRA y QLoRA | Operar |
| **HF `datasets`** | Carga y transforma datasets | Operar |
| **`accelerate`** | Capa que hace que el mismo script corra en 1 GPU, multi-GPU, FSDP o DeepSpeed | Reconocer |
| **DeepSpeed** (Microsoft) | *Sharding* del modelo y del optimizador entre varias GPUs, para modelos que no caben en una | ⚠️ **Solo reconocer** |
| **FSDP** (PyTorch) | Fully Sharded Data Parallel: lo mismo, nativo de PyTorch | ⚠️ **Solo reconocer** |
| **`bitsandbytes`** | Cuantización durante el entrenamiento, lo que habilita QLoRA | Operar básico |

**Sobre DeepSpeed y FSDP, la respuesta honesta:** *"I know what they do — sharding model and
optimizer state across GPUs for models that don't fit on one. I haven't operated multi-GPU
clusters."* No se opera sin cluster y decirlo así es más creíble que fingir.

---

## 4. Servir el modelo (inferencia)

### vLLM

Servidor de inferencia open source de alto throughput. Sus dos ideas clave son
**PagedAttention** (gestiona el KV cache en páginas, como la memoria virtual de un sistema
operativo, así que desperdicia mucha menos memoria) y **continuous batching** (mete peticiones
nuevas al lote sin esperar a que termine el anterior).

> *"An open-source high-throughput inference server. PagedAttention manages the KV cache in
> pages so you waste far less memory, and continuous batching keeps the GPU busy."*

### TensorRT-LLM

El equivalente de NVIDIA, con kernels compilados. Solo reconocerlo.

### Quantization

Comprimir los pesos del modelo a 8 o 4 bits (**GPTQ**, **AWQ**, **GGUF**). Menos memoria, más
velocidad, con pérdida mínima de calidad.

> *"Compressing model weights to 8 or 4 bits — GPTQ, AWQ, GGUF — for less VRAM and higher
> throughput, with minimal quality loss."*

### KV cache

Las claves y valores de la atención que el modelo guarda de los tokens ya generados, para no
recalcularlos en cada token nuevo. Es lo que consume la memoria durante la generación, y lo
que PagedAttention administra.

### Throughput vs latencia

**Throughput** es cuántos tokens por segundo produce el servidor en total; **latencia** es
cuánto tarda una petición individual. Se optimizan en direcciones distintas: batches grandes
suben throughput y suben latencia.

**Ancla:** tu model routing por costo y latencia es exactamente este tradeoff, resuelto en la
capa de orquestación en vez de en la de servido.

---

## 5. Arquitectura de transformers, lo mínimo

La JD de varios roles pide *"strong grasp of transformer architectures"*. Lo básico defendible:

**Self-attention** — cada token mira a todos los demás y pondera cuánto le importa cada uno.
Es lo que reemplazó a las redes recurrentes: se puede paralelizar sobre la secuencia completa.

**Multi-head attention** — se hacen varias atenciones en paralelo, cada "cabeza" aprende a
fijarse en relaciones distintas.

**Los tres tipos de arquitectura**, que la JD del cliente "HY" nombra explícitamente:

- **Encoder** (BERT): lee toda la secuencia de golpe y en ambas direcciones. Bueno para
  clasificación y para **embeddings**.
- **Decoder** (GPT, Llama, Qwen): genera token por token mirando solo hacia atrás. Es lo que
  usas para generación, y lo que fine-tuneas en este proyecto.
- **Encoder-decoder** (T5): el encoder lee la entrada, el decoder produce la salida. Traducción,
  resumen.

**Embeddings** — representar texto como vectores donde la cercanía significa similitud
semántica. **Ancla:** es la base de tu Visual RAG con Qdrant, así que este lo tienes de verdad.

**Reranking** — un modelo que reordena los resultados que devolvió la búsqueda vectorial,
mirando la consulta y el documento juntos en vez de por separado. Más caro y más preciso.

---

## 6. Seguridad, para no mezclar dos siglas parecidas

**OWASP** — una fundación. Lo que todos usan de ellos es el **OWASP Top 10**, la lista de las
diez categorías de vulnerabilidad web más críticas. Es un **vocabulario**, no una certificación.
Tu experiencia con SQLmap es A03 Injection.

**OSCP** — una **certificación** de pentesting (Offensive Security Certified Professional):
examen práctico de 24 horas. **No la tienes y nunca se afirma.** En las JDs aparece como *nice
to have*. Tu experiencia real de pentesting se cuenta como historia, jamás como credencial.

**BYOK** — *Bring Your Own Key*: el cliente usa su propia API key o su propio KMS en vez de las
credenciales del proveedor. Solo reconocerlo.

---

## 7. Chuleta de 10 líneas

Repasar cinco minutos antes de cualquier llamada.

1. **SFT** — entrenar con ejemplos curados prompt → respuesta. Imitación.
2. **DPO** — pares preferido/rechazado, pérdida directa, sin reward model ni RL.
3. **RLHF** — SFT, luego reward model humano, luego PPO. Clásico y caro.
4. **RLAIF** — RLHF con un juez LLM en vez de humanos.
5. **GRPO** — RL por grupos contra el promedio del grupo, sin critic. DeepSeek. Ideal con
   rewards verificables.
6. **LoRA / QLoRA** — adaptadores de bajo rango sobre base congelada, 4 bits en QLoRA.
   Fine-tuning en una GPU.
7. **Synthetic data** — un LLM fuerte genera, jueces filtran.
8. **DeepSpeed / FSDP** — sharding multi-GPU para modelos que no caben en una. Solo reconocer.
9. **vLLM** — servido de alto throughput, PagedAttention y continuous batching.
   **Quantization** — pesos a 8 o 4 bits, GPTQ, AWQ, GGUF.
10. **Encoder / decoder / encoder-decoder** — clasificación y embeddings / generación /
    traducción y resumen.
