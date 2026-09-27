"""Cliente mínimo para la API HTTP de Ollama (sin dependencias externas)."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Quita los bloques <think>...</think> que emiten algunos modelos (Qwen3, DeepSeek-R1)."""
    return _THINK_RE.sub("", text).strip()


class LLMError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 600):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise LLMError(f"Ollama respondió {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"No se pudo conectar con Ollama en {self.host}. "
                "¿Está en marcha? (ejecuta `ollama serve`)"
            ) from e

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> dict:
        """Envía una conversación y devuelve el mensaje del asistente.

        El resultado tiene la forma {"role": "assistant", "content": str, "tool_calls": [...]}.
        Si se pasa `on_token`, la respuesta se transmite token a token.
        """
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["format"] = "json"
        if temperature is not None:
            body["options"] = {"temperature": temperature}

        content: list[str] = []
        tool_calls: list[dict] = []
        in_think = False
        with self._post("/api/chat", body) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(chunk["error"])
                msg = chunk.get("message", {})
                piece = msg.get("content", "")
                if piece:
                    content.append(piece)
                    # No mostramos el razonamiento interno <think> por pantalla.
                    if "<think>" in piece:
                        in_think = True
                    if on_token and not in_think:
                        on_token(piece)
                    if "</think>" in piece:
                        in_think = False
                tool_calls.extend(msg.get("tool_calls") or [])
                if chunk.get("done"):
                    break

        return {
            "role": "assistant",
            "content": strip_think("".join(content)),
            "tool_calls": tool_calls,
        }

    def list_models(self) -> list[str]:
        req = urllib.request.Request(self.host + "/api/tags")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise LLMError(f"No se pudo conectar con Ollama en {self.host}.") from e
        return [m["name"] for m in data.get("models", [])]


def parse_json(text: str) -> dict:
    """Extrae el primer objeto JSON de un texto; devuelve {} si no hay ninguno válido."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}
