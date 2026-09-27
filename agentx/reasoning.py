"""Bucle "pensar → comprobar → corregir" (versión a nivel de sistema de AgentX-Core).

Un modelo local pequeño se equivoca más que uno de frontera; este bucle le da una segunda
(y tercera) oportunidad: un verificador revisa la respuesta y, si encuentra fallos, el modelo
la reescribe teniéndolos en cuenta.
"""

from __future__ import annotations

from typing import Callable

from agentx.llm import parse_json

VERIFIER_PROMPT = """Eres un verificador riguroso. Revisa si la RESPUESTA resuelve correcta y \
completamente la PREGUNTA. Busca errores de cálculo, de lógica, datos inventados, partes sin \
responder y código que no funcionaría. Si la respuesta es correcta, no inventes problemas.

Responde SOLO con JSON: {"correcta": true|false, "problemas": ["..."]}"""

REVISE_PROMPT = """Un verificador encontró estos problemas en tu respuesta anterior:
{problems}

Escribe una respuesta final corregida y completa (no menciones al verificador)."""


def verify(llm, question: str, answer: str) -> tuple[bool, list[str]]:
    reply = llm.chat(
        [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": f"PREGUNTA:\n{question}\n\nRESPUESTA:\n{answer}"},
        ],
        json_mode=True,
        temperature=0,
    )
    data = parse_json(reply["content"])
    problems = [str(p) for p in data.get("problemas", []) if str(p).strip()]
    ok = bool(data.get("correcta", not problems))
    return ok and not problems, problems


def refine(
    llm,
    messages: list[dict],
    question: str,
    answer: str,
    max_rounds: int = 2,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Verifica `answer` y la corrige hasta `max_rounds` veces. Devuelve la mejor versión."""
    for i in range(max_rounds):
        if on_status:
            on_status(f"verificando (ronda {i + 1}/{max_rounds})...")
        ok, problems = verify(llm, question, answer)
        if ok:
            if on_status:
                on_status("respuesta verificada ✓")
            return answer
        if on_status:
            on_status(f"corrigiendo {len(problems)} problema(s)...")
        revision = llm.chat(
            messages
            + [
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": REVISE_PROMPT.format(
                        problems="\n".join(f"- {p}" for p in problems)
                    ),
                },
            ]
        )
        if revision["content"].strip():
            answer = revision["content"]
    return answer
