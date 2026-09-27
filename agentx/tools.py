"""Herramientas que la IA puede usar: ejecutar Python, leer archivos, memoria, fecha."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from agentx.memory import MemoryStore

MAX_OUTPUT = 6000


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _fn(
        "ejecutar_python",
        "Ejecuta código Python 3 y devuelve lo que imprime. Úsalo para cálculos, "
        "comprobar respuestas, procesar datos o verificar código. Usa print() para ver resultados.",
        {"codigo": {"type": "string", "description": "Código Python a ejecutar"}},
        ["codigo"],
    ),
    _fn(
        "leer_archivo",
        "Lee un archivo de texto del ordenador del usuario.",
        {"ruta": {"type": "string", "description": "Ruta del archivo"}},
        ["ruta"],
    ),
    _fn(
        "listar_directorio",
        "Lista los archivos de un directorio del ordenador del usuario.",
        {"ruta": {"type": "string", "description": "Ruta del directorio"}},
        ["ruta"],
    ),
    _fn(
        "recordar",
        "Guarda en la memoria a largo plazo un hecho importante sobre el usuario o sus proyectos.",
        {"hecho": {"type": "string", "description": "Hecho breve y autocontenido"}},
        ["hecho"],
    ),
    _fn(
        "buscar_memoria",
        "Busca en la memoria a largo plazo recuerdos relacionados con un tema.",
        {"consulta": {"type": "string", "description": "Palabras clave"}},
        ["consulta"],
    ),
    _fn("fecha_actual", "Devuelve la fecha y hora actuales.", {}, []),
]


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [salida truncada, {len(text) - MAX_OUTPUT} caracteres más]"


class ToolBox:
    """Ejecuta las llamadas a herramientas que pide el modelo.

    `confirm` se llama antes de acciones con efectos (ejecutar código); si devuelve False
    la acción se cancela. Por defecto se niega todo, para que nada se ejecute sin permiso.
    """

    def __init__(
        self,
        memory: MemoryStore,
        confirm: Callable[[str, str], bool] = lambda action, detail: False,
        python_timeout: float = 30,
    ):
        self.memory = memory
        self.confirm = confirm
        self.python_timeout = python_timeout

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return f"Error: la herramienta '{name}' no existe."
        try:
            return _truncate(handler(**args))
        except TypeError as e:
            return f"Error: argumentos inválidos para '{name}': {e}"
        except Exception as e:  # la IA debe ver el error para poder corregirse
            return f"Error ejecutando '{name}': {type(e).__name__}: {e}"

    def _t_ejecutar_python(self, codigo: str) -> str:
        if not self.confirm("ejecutar_python", codigo):
            return "El usuario no autorizó ejecutar este código."
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "script.py"
            script.write_text(codigo, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.python_timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return f"Error: el código superó el límite de {self.python_timeout:g} s."
        out = proc.stdout
        if proc.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + proc.stderr
        return out.strip() or f"(sin salida, código de salida {proc.returncode})"

    def _t_leer_archivo(self, ruta: str) -> str:
        path = Path(ruta).expanduser()
        return path.read_text(encoding="utf-8", errors="replace")

    def _t_listar_directorio(self, ruta: str) -> str:
        path = Path(ruta).expanduser()
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        return "\n".join(p.name + ("/" if p.is_dir() else "") for p in entries) or "(vacío)"

    def _t_recordar(self, hecho: str) -> str:
        return "Guardado en memoria." if self.memory.add(hecho) else "Ya lo sabía."

    def _t_buscar_memoria(self, consulta: str) -> str:
        found = self.memory.search(consulta)
        return "\n".join(f"- {m.content}" for m in found) or "No hay recuerdos sobre eso."

    def _t_fecha_actual(self) -> str:
        return dt.datetime.now().strftime("%A %d/%m/%Y %H:%M")
