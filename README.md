# AgentX — tu IA personal local

IA que corre **en tu ordenador**, sin enviar nada a internet, construida sobre un modelo
abierto (vía [Ollama](https://ollama.com)) y reforzada con lo que a los chatbots normales les falta:

- **Memoria persistente**: aprende hechos sobre ti y los recuerda en cada conversación
  (`~/.agentx/memoria.db`, puedes verla y borrarla).
- **Razonamiento verificado** (`/pensar`): un verificador revisa la respuesta y el modelo la
  corrige antes de dártela.
- **Herramientas**: ejecuta Python para calcular y comprobar (siempre te pide permiso),
  lee archivos y carpetas, consulta la fecha.
- **Privacidad total**: sin cuentas, sin API keys, sin coste por uso.

El plan de investigación completo (arquitectura AgentX-Core y fases) está en [PLAN.md](PLAN.md).

## Instalación

1. Instala Ollama: <https://ollama.com/download>
2. Descarga un modelo (elige según tu hardware, ver tabla):
   ```bash
   ollama pull qwen3:8b
   ```
3. Instala AgentX (Python 3.10+, sin dependencias):
   ```bash
   git clone <este repositorio> && cd agentx
   pip install -e .
   ```

## Uso

```bash
agentx                       # abre el chat
agentx "¿qué tiempo hace en Marte?"   # pregunta única
agentx -m qwen3:14b          # otro modelo
agentx --profundo            # verifica todas las respuestas
```

Dentro del chat: `/pensar <pregunta>`, `/memoria`, `/olvidar <id|todo>`, `/recordar <hecho>`,
`/nuevo`, `/modelo <nombre>`, `/auto`, `/ayuda`, `/salir`.

Variables de entorno: `AGENTX_MODEL` (modelo por defecto), `OLLAMA_HOST`, `AGENTX_HOME`.

## ¿Qué modelo uso?

Cuanto más grande, más inteligente, pero necesita más memoria. Todos los de esta tabla
soportan herramientas.

| Tu equipo | Modelo recomendado |
|-----------|--------------------|
| Sin GPU / 8 GB RAM | `qwen3:4b` |
| GPU 8 GB o Mac 16 GB | `qwen3:8b` (por defecto) |
| GPU 12–16 GB o Mac 32 GB | `qwen3:14b` |
| GPU 24 GB o Mac 64 GB | `qwen3:32b` o `qwen3:30b-a3b` |

## Pruebas

```bash
pip install -e ".[dev]" && pytest
```
