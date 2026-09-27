"""El agente: une modelo local, memoria, herramientas y verificación."""

from __future__ import annotations

import datetime as dt
import json
from typing import Callable

from agentx.llm import parse_json
from agentx.memory import MemoryStore
from agentx.reasoning import refine
from agentx.tools import TOOL_SCHEMAS, ToolBox

SYSTEM_PROMPT = """Eres AgentX, la IA personal de tu usuario. Corres en su propio ordenador.
Responde en el idioma del usuario, de forma clara y directa.

Cómo trabajas:
- Para cualquier cálculo, dato numérico o código, usa la herramienta ejecutar_python y comprueba \
el resultado en lugar de calcularlo de cabeza.
- Si no sabes algo o no puedes comprobarlo, dilo; nunca inventes datos.
- Cuando el usuario te cuente algo importante y duradero sobre sí mismo (nombre, trabajo, \
proyectos, preferencias), guárdalo con la herramienta recordar.

Fecha actual: {today}

Lo que recuerdas del usuario:
{memories}"""

EXTRACT_PROMPT = """Extrae de este intercambio los hechos DURADEROS sobre el usuario que valga la \
pena recordar en futuras conversaciones (nombre, profesión, proyectos, gustos, objetivos, \
personas cercanas). NO guardes preguntas, cálculos, resultados ni lo que la IA respondió; solo \
lo que el usuario contó de sí mismo. Cada hecho, una frase breve \
en tercera persona ("El usuario ...").

Responde SOLO con JSON: {"hechos": ["..."]} (lista vacía si no hay nada)."""

MAX_TOOL_STEPS = 8
MAX_HISTORY = 40


class Agent:
    def __init__(
        self,
        llm,
        memory: MemoryStore,
        tools: ToolBox | None = None,
        auto_memory: bool = True,
        use_tools: bool = True,
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools or ToolBox(memory)
        self.auto_memory = auto_memory
        self.use_tools = use_tools
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history = []

    def _system(self, user_text: str) -> dict:
        found = {m.id: m for m in self.memory.search(user_text)}
        for m in self.memory.recent(5):
            found.setdefault(m.id, m)
        memories = "\n".join(f"- {m.content}" for m in found.values()) or "(nada todavía)"
        today = dt.date.today().strftime("%d/%m/%Y")
        return {"role": "system", "content": SYSTEM_PROMPT.format(today=today, memories=memories)}

    def respond(
        self,
        user_text: str,
        deep: bool = False,
        on_token: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Responde a `user_text`. Con `deep=True` verifica y corrige la respuesta antes de darla."""
        self.history.append({"role": "user", "content": user_text})
        messages = [self._system(user_text)] + self.history[-MAX_HISTORY:]
        tools = TOOL_SCHEMAS if self.use_tools else None
        # En modo profundo no se transmite el borrador: el usuario solo ve la versión verificada.
        stream = None if deep else on_token

        answer = ""
        for _ in range(MAX_TOOL_STEPS):
            reply = self.llm.chat(messages, tools=tools, on_token=stream)
            calls = reply.get("tool_calls") or []
            if not calls:
                answer = reply["content"]
                break
            messages.append(
                {"role": "assistant", "content": reply["content"], "tool_calls": calls}
            )
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = parse_json(args)
                if on_status:
                    on_status(f"usando herramienta: {name}")
                result = self.tools.run(name, args)
                messages.append({"role": "tool", "content": result, "tool_name": name})
        else:
            answer = reply["content"] or "(Se alcanzó el límite de pasos con herramientas.)"

        if deep:
            answer = refine(self.llm, messages, user_text, answer, on_status=on_status)
            if on_token:
                on_token(answer)

        self.history.append({"role": "assistant", "content": answer})
        if self.auto_memory:
            self._learn(user_text, answer)
        return answer

    def _learn(self, user_text: str, answer: str) -> list[str]:
        """Extrae hechos duraderos del intercambio y los guarda en memoria."""
        exchange = json.dumps({"usuario": user_text, "asistente": answer[:2000]}, ensure_ascii=False)
        try:
            reply = self.llm.chat(
                [
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": exchange},
                ],
                json_mode=True,
                temperature=0,
            )
        except Exception:
            return []  # aprender es opcional; nunca debe romper la conversación
        facts = parse_json(reply["content"]).get("hechos", [])
        saved = []
        for fact in facts if isinstance(facts, list) else []:
            if isinstance(fact, str) and self.memory.add(fact):
                saved.append(fact)
        return saved
