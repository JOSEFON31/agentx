# AgentX — Plan para un modelo de IA experimental

> Objetivo: diseñar, construir y evaluar una arquitectura de IA propia que combine
> las líneas de investigación experimentales más prometedoras, con la ambición de
> superar a los modelos actuales **en eficiencia, razonamiento y aprendizaje continuo**.

---

## 0. Expectativas realistas (léelo primero)

Superar a los modelos de frontera (GPT, Claude, Gemini) en capacidad general cuesta
hoy cientos de millones de dólares en cómputo y datos. Un proyecto pequeño **no** va a
ganar por fuerza bruta. Donde sí se puede ganar es en:

1. **Eficiencia**: igual rendimiento con 10–100× menos parámetros o cómputo.
2. **Capacidades que los Transformers actuales hacen mal**: memoria a largo plazo,
   aprendizaje continuo sin reentrenar, razonamiento de profundidad variable,
   verificación de sus propias respuestas.
3. **Dominios acotados**: superar a modelos grandes en matemáticas, código o un
   dominio concreto con un modelo pequeño bien diseñado.

El plan está pensado para escalar: se valida cada idea en pequeño (10M–100M
parámetros) y solo lo que demuestre ventaja pasa a la siguiente escala.

---

## 1. La arquitectura imaginada: **AgentX-Core**

Idea central: un modelo que **no piensa en tokens, sino en un espacio latente**, que
**decide cuánto pensar**, **recuerda entre sesiones** y **verifica lo que produce**.

```
                ┌──────────────────────────────────────────────┐
  entrada ──►   │ 1. Codificador multimodal (bytes / parches)   │
                └───────────────┬──────────────────────────────┘
                                ▼
                ┌──────────────────────────────────────────────┐
                │ 2. Columna híbrida: SSM (Mamba) + Atención    │
                │    local + Mezcla de Expertos (MoE)           │
                └───────────────┬──────────────────────────────┘
                                ▼
   ┌────────────┐  ┌──────────────────────────────────────────┐
   │ 4. Memoria │◄─┤ 3. Núcleo de razonamiento recurrente     │
   │ persistente│─►│    (profundidad adaptativa en latente)   │◄──┐
   └────────────┘  └───────────────┬──────────────────────────┘   │
                                   ▼                              │
                ┌──────────────────────────────────────────────┐  │
                │ 5. Modelo del mundo predictivo (estilo JEPA)  │  │
                └───────────────┬──────────────────────────────┘  │
                                ▼                                 │
                ┌──────────────────────────────────────────────┐  │
                │ 6. Verificador / crítico interno              │──┘ (si falla,
                └───────────────┬──────────────────────────────┘     sigue pensando)
                                ▼
                             salida
```

### Componentes y método experimental del que parte cada uno

| # | Componente | Base experimental | Qué aporta |
|---|------------|-------------------|------------|
| 1 | Codificador sin tokenizador | Byte Latent Transformer, parches dinámicos por entropía | Sin vocabulario fijo; texto, código, imagen y audio en el mismo espacio |
| 2 | Columna híbrida SSM + atención + MoE | Mamba-2, Jamba, Mixture-of-Experts, Mixture-of-Depths | Contexto muy largo a coste casi lineal; solo se activa una fracción de parámetros por token |
| 3 | Razonamiento recurrente latente | Recurrent-depth / Universal Transformers, "Coconut" (chain-of-thought continuo), Adaptive Computation Time | Piensa varias iteraciones sin escribir tokens; problemas difíciles reciben más cómputo |
| 4 | Memoria persistente | Titans (memoria neuronal que aprende en inferencia), Memorizing Transformers, test-time training | Aprende de la conversación y la recuerda después, sin reentrenar |
| 5 | Modelo del mundo | JEPA / V-JEPA, modelos de mundo tipo Dreamer | Predice consecuencias en espacio abstracto, no solo la siguiente palabra |
| 6 | Verificador interno | Process Reward Models, RL con recompensas verificables, métodos neuro-simbólicos (ejecución de código, demostradores) | Detecta sus propios errores y decide si responder o seguir pensando |

### Lo original de la propuesta

Las piezas existen por separado; lo nuevo es **el bucle 3 → 5 → 6 → 3**:

- El núcleo recurrente propone un estado de "pensamiento".
- El modelo del mundo simula qué pasaría si ese pensamiento fuera cierto.
- El verificador puntúa la coherencia; si la puntuación es baja, el núcleo itera
  otra vez (con un límite de presupuesto).
- Lo que resulta útil se escribe en la memoria persistente.

Es decir: **"pensar → imaginar → comprobar → recordar"** dentro de la red, en lugar de
hacerlo con texto externo como los agentes actuales.

---

## 2. Fases del proyecto

### Fase 0 — Infraestructura (semanas 1–3)
- Repositorio con PyTorch 2.x, `uv`/`pip`, pruebas y CI.
- Pipeline de datos: FineWeb-Edu (texto), The Stack v2 (código), OpenWebMath,
  conjuntos sintéticos generados (aritmética, lógica, laberintos).
- Entrenamiento distribuido (FSDP), mezcla de precisión bf16, registro con
  Weights & Biases o TensorBoard.
- **Línea base**: un Transformer estilo GPT/Llama de 50M y 125M parámetros.
  Todo se compara contra él con el **mismo presupuesto de cómputo (FLOPs)**.

### Fase 1 — Validar cada pieza por separado (semanas 4–10)
Un experimento por componente, a 50M–125M parámetros, midiendo contra la línea base:

| Experimento | Pregunta | Métrica de éxito |
|-------------|----------|------------------|
| E1 Codificador de bytes | ¿Igual perplejidad por byte sin tokenizador? | bits-por-byte ≤ línea base |
| E2 Columna híbrida | ¿Mejor en contexto largo por FLOP? | recuperación en 64k tokens (needle, RULER) |
| E3 Razonamiento recurrente | ¿Más iteraciones = más acierto? | curva acierto vs. iteraciones en GSM8K / aritmética de varios dígitos |
| E4 Memoria persistente | ¿Recuerda hechos entre sesiones? | tareas de recuerdo tras N sesiones sin reentrenar |
| E5 Modelo del mundo | ¿Ayuda la pérdida predictiva latente? | planificación en laberintos / Sokoban; transferencia a texto |
| E6 Verificador | ¿Detecta sus errores? | AUROC del verificador; ganancia best-of-N |

Regla: **una pieza que no mejore a la línea base se descarta o se rediseña.**

### Fase 2 — Integración de AgentX-Core (semanas 11–18)
- Unir las piezas ganadoras en un modelo de ~350M parámetros activos.
- Entrenamiento en tres etapas:
  1. **Preentrenamiento** predictivo (siguiente parche + pérdida JEPA latente).
  2. **Ajuste supervisado** con trazas de razonamiento destiladas a latente
     (curriculum estilo Coconut: se sustituyen pasos de texto por pasos latentes).
  3. **Aprendizaje por refuerzo con recompensas verificables** (matemáticas con
     respuesta comprobable, código con tests), entrenando conjuntamente el
     verificador.
- Estudios de ablación: quitar cada componente y medir la caída.

### Fase 3 — Automejora y aprendizaje continuo (semanas 19–26)
- El modelo genera problemas, los resuelve, el verificador (y ejecución real de
  código / comprobación simbólica) filtra, y se reentrena con lo correcto
  (tipo STaR / AlphaZero para lenguaje).
- Evaluar olvido catastrófico: rendimiento en tareas viejas tras aprender nuevas.
- Búsqueda automática de arquitectura sobre hiperparámetros clave (número de
  expertos, iteraciones máximas, tamaño de memoria).

### Fase 4 — Escalado (a partir del mes 7, según resultados y presupuesto)
- Ajustar leyes de escalado con 3–4 tamaños (50M → 1.3B) y extrapolar.
- Solo si la pendiente es mejor que la del Transformer, escalar a 3B–7B.

---

## 3. Evaluación: cómo sabremos si "supera" a la IA actual

Comparar contra modelos abiertos **del mismo tamaño y cómputo** (Llama, Qwen,
SmolLM, Mamba) y, en dominios acotados, contra modelos mucho mayores.

- **Razonamiento**: GSM8K, MATH, ARC-AGI (clave: mide generalización, no memoria).
- **Código**: HumanEval+, MBPP+, LiveCodeBench.
- **Contexto largo**: RULER, BABILong.
- **Conocimiento general**: MMLU-Pro, HellaSwag.
- **Aprendizaje continuo** (propio): recuerdo y acierto tras N sesiones.
- **Eficiencia**: acierto por FLOP de inferencia; latencia; memoria.
- **Calibración**: ¿sabe cuándo no sabe? (error de calibración esperado).

Hitos de "superación" concretos:
1. Mismo acierto que un Transformer 3× más grande (Fase 2).
2. Superar a un modelo 10× más grande en matemáticas o código (Fase 3).
3. Aprender tareas nuevas en inferencia sin degradar las anteriores (Fase 3).

---

## 4. Recursos estimados

| Fase | Cómputo | Coste aproximado (nube) |
|------|---------|-------------------------|
| 0–1 | 1–8 GPU (A100/H100), días por experimento | 500–3.000 USD |
| 2 | 8–32 GPU, 1–3 semanas | 5.000–20.000 USD |
| 3 | 8–32 GPU continuo | 10.000–40.000 USD |
| 4 (1–7B) | 64–512 GPU | 50.000–500.000+ USD |

Fases 0–1 se pueden empezar con una sola GPU de consumo (RTX 3090/4090) o créditos
académicos / de nube a escala de 10M–50M parámetros.

---

## 5. Riesgos y mitigación

| Riesgo | Mitigación |
|--------|------------|
| Las piezas funcionan solas pero no juntas | Integración incremental, una pieza a la vez, con ablaciones |
| Razonamiento recurrente inestable | Normalización, límite de iteraciones, pérdida de parada aprendida |
| El verificador aprende a ser complaciente (*reward hacking*) | Anclar recompensas a comprobaciones externas (tests, solvers) |
| La memoria acumula errores o información falsa | Puntuación de confianza antes de escribir; olvido controlado |
| Resultados que no escalan | Leyes de escalado antes de gastar en grande |
| Evaluación contaminada | Benchmarks recientes y conjuntos propios nunca vistos |

## 6. Seguridad y alineamiento (desde el día 1)

Un sistema que aprende en inferencia y se automejora es más difícil de controlar:

- Memoria **auditable** e inspeccionable; posibilidad de borrarla.
- Límites duros de cómputo por respuesta y de automodificación.
- Evaluaciones de rechazo de peticiones dañinas y de honestidad en cada checkpoint.
- Herramientas de interpretabilidad (sondas lineales, autoencoders dispersos)
  sobre el estado latente del razonamiento, ya que no se ve como texto.
- Registro de todos los datos autogenerados usados para reentrenar.

---

## 7. Estructura propuesta del repositorio

```
agentx/
├── PLAN.md
├── configs/             # YAML por experimento (E1–E6, integración)
├── agentx/
│   ├── data/            # carga, parches de bytes, datos sintéticos
│   ├── modules/
│   │   ├── byte_encoder.py
│   │   ├── hybrid_block.py     # Mamba + atención local + MoE
│   │   ├── latent_reasoner.py  # núcleo recurrente con ACT
│   │   ├── memory.py           # memoria neuronal persistente
│   │   ├── world_model.py      # predictor JEPA
│   │   └── verifier.py
│   ├── model.py         # ensamblado de AgentX-Core
│   ├── train/           # preentrenamiento, SFT, RL
│   └── eval/            # benchmarks y métricas
├── baselines/           # Transformer de referencia
└── tests/
```

---

## 8. Siguientes pasos inmediatos

1. Montar la Fase 0: esqueleto del repositorio, línea base Transformer de ~10M
   parámetros entrenable en una GPU, y datos sintéticos de aritmética.
2. Implementar E3 (razonamiento recurrente latente) primero: es el componente
   más distintivo y el que más barato se prueba.
3. Publicar resultados de cada experimento en `results/` con configuración y
   semilla para que sean reproducibles.
