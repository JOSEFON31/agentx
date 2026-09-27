"""Interfaz de terminal de AgentX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agentx import __version__
from agentx.agent import Agent
from agentx.llm import LLMError, OllamaClient
from agentx.memory import MemoryStore
from agentx.tools import ToolBox

DEFAULT_MODEL = os.environ.get("AGENTX_MODEL", "qwen3:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_HOME = Path(os.environ.get("AGENTX_HOME", Path.home() / ".agentx"))

HELP = """Comandos:
  /pensar <pregunta>   responde verificando y corrigiendo la respuesta (más lento, más fiable)
  /profundo            activa/desactiva el modo verificado para todos los mensajes
  /memoria             muestra lo que recuerda de ti
  /olvidar <id|todo>   borra un recuerdo (o todos)
  /recordar <hecho>    guarda un hecho a mano
  /nuevo               empieza una conversación nueva (la memoria se conserva)
  /modelo [nombre]     muestra o cambia el modelo (ver `ollama list`)
  /auto                activa/desactiva la ejecución de código sin pedir permiso
  /ayuda               esta ayuda
  /salir               salir"""

DIM, BOLD, CYAN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[0m"
if not sys.stdout.isatty():
    DIM = BOLD = CYAN = RESET = ""


def status(msg: str) -> None:
    print(f"{DIM}  · {msg}{RESET}", flush=True)


def token(piece: str) -> None:
    print(piece, end="", flush=True)


class Session:
    def __init__(self, args):
        self.auto_run = args.auto
        self.deep = args.profundo
        self.memory = MemoryStore(Path(args.home) / "memoria.db")
        self.llm = OllamaClient(args.modelo, host=args.host)
        tools = ToolBox(self.memory, confirm=self.confirm)
        self.agent = Agent(self.llm, self.memory, tools, auto_memory=not args.sin_memoria)

    def confirm(self, action: str, detail: str) -> bool:
        if self.auto_run:
            return True
        print(f"\n{DIM}--- La IA quiere ejecutar este código ---\n{detail}\n---{RESET}")
        try:
            return input("¿Ejecutar? [s/N] ").strip().lower() in ("s", "si", "sí", "y", "yes")
        except EOFError:
            return False

    def ask(self, text: str, deep: bool) -> None:
        print(f"{CYAN}{BOLD}AgentX:{RESET} ", end="", flush=True)
        try:
            self.agent.respond(text, deep=deep, on_token=token, on_status=status)
        except LLMError as e:
            print(f"\n[error] {e}")
            self.agent.history.pop()  # descarta el mensaje que no obtuvo respuesta
        print()

    def command(self, line: str) -> bool:
        """Procesa un comando /x. Devuelve False si hay que salir."""
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("salir", "exit", "quit"):
            return False
        if cmd == "ayuda":
            print(HELP)
        elif cmd == "pensar":
            if arg:
                self.ask(arg, deep=True)
            else:
                print("Uso: /pensar <pregunta>")
        elif cmd == "profundo":
            self.deep = not self.deep
            print(f"Modo verificado: {'activado' if self.deep else 'desactivado'}")
        elif cmd == "memoria":
            items = self.memory.all()
            print("\n".join(f"  [{m.id}] {m.content}" for m in items) or "  (vacía)")
        elif cmd == "olvidar":
            if arg == "todo":
                self.memory.clear()
                print("Memoria borrada.")
            elif arg.isdigit():
                print("Olvidado." if self.memory.delete(int(arg)) else "No existe ese recuerdo.")
            else:
                print("Uso: /olvidar <id|todo>")
        elif cmd == "recordar":
            print("Guardado." if arg and self.memory.add(arg) else "Nada que guardar.")
        elif cmd == "nuevo":
            self.agent.reset()
            print("Conversación nueva.")
        elif cmd == "modelo":
            if arg:
                self.llm.model = arg
            print(f"Modelo: {self.llm.model}")
        elif cmd == "auto":
            self.auto_run = not self.auto_run
            print(f"Ejecutar código sin preguntar: {'sí' if self.auto_run else 'no'}")
        else:
            print(f"Comando desconocido: /{cmd}. Escribe /ayuda.")
        return True

    def loop(self) -> None:
        print(f"{BOLD}AgentX {__version__}{RESET} · modelo {self.llm.model} · /ayuda para comandos")
        try:
            available = self.llm.list_models()
            if not any(m == self.llm.model or m.split(":")[0] == self.llm.model for m in available):
                print(f"[aviso] El modelo '{self.llm.model}' no está descargado. "
                      f"Ejecuta: ollama pull {self.llm.model}")
        except LLMError as e:
            print(f"[aviso] {e}")
        while True:
            try:
                line = input(f"\n{BOLD}Tú:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    break
                continue
            try:
                self.ask(line, deep=self.deep)
            except KeyboardInterrupt:
                print("\n[interrumpido]")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agentx", description="IA personal local con memoria.")
    p.add_argument("pregunta", nargs="*", help="pregunta única (sin ella, abre el chat)")
    p.add_argument("-m", "--modelo", default=DEFAULT_MODEL, help=f"modelo de Ollama ({DEFAULT_MODEL})")
    p.add_argument("--host", default=DEFAULT_HOST, help="URL de Ollama")
    p.add_argument("--home", default=str(DEFAULT_HOME), help="carpeta de datos (memoria)")
    p.add_argument("-p", "--profundo", action="store_true", help="verificar todas las respuestas")
    p.add_argument("--auto", action="store_true", help="ejecutar código sin pedir permiso")
    p.add_argument("--sin-memoria", action="store_true", help="no aprender hechos automáticamente")
    args = p.parse_args(argv)

    session = Session(args)
    if args.pregunta:
        session.ask(" ".join(args.pregunta), deep=args.profundo)
    else:
        session.loop()


if __name__ == "__main__":
    main()
