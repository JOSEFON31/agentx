import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentx.agent import Agent
from agentx.llm import OllamaClient, parse_json, strip_think
from agentx.memory import MemoryStore
from agentx.reasoning import refine
from agentx.tools import ToolBox


class FakeLLM:
    """Devuelve respuestas predefinidas y registra las llamadas."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, on_token=None, json_mode=False, temperature=None):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = {"role": "assistant", "content": reply, "tool_calls": []}
        if on_token and reply["content"]:
            on_token(reply["content"])
        return reply


@pytest.fixture
def memory():
    return MemoryStore(":memory:")


def test_memory_add_search_delete(memory):
    assert memory.add("El usuario se llama Juan y programa en Python")
    assert not memory.add("El usuario se llama Juan y programa en Python")
    memory.add("Al usuario le gusta el café")
    found = memory.search("¿qué lenguaje uso? python")
    assert [m.content for m in found] == ["El usuario se llama Juan y programa en Python"]
    assert memory.search("") == []
    assert memory.delete(found[0].id)
    assert memory.search("python") == []


def test_memory_persists_on_disk(tmp_path):
    MemoryStore(tmp_path / "m.db").add("Vive en Bogotá")
    assert MemoryStore(tmp_path / "m.db").search("bogotá")[0].content == "Vive en Bogotá"


def test_strip_think_and_parse_json():
    assert strip_think("<think>hmm</think>\nHola") == "Hola"
    assert parse_json('texto {"a": 1} más') == {"a": 1}
    assert parse_json("nada") == {}


def test_python_tool_requires_confirmation(memory):
    denied = ToolBox(memory)
    assert "no autorizó" in denied.run("ejecutar_python", {"codigo": "print(1)"})
    allowed = ToolBox(memory, confirm=lambda a, d: True)
    assert allowed.run("ejecutar_python", {"codigo": "print(2**10)"}) == "1024"
    assert "ZeroDivisionError" in allowed.run("ejecutar_python", {"codigo": "1/0"})
    assert "no existe" in allowed.run("volar", {})
    assert "argumentos inválidos" in allowed.run("leer_archivo", {})


def test_agent_tool_loop_and_memory_injection(memory):
    memory.add("El usuario se llama Juan")
    llm = FakeLLM(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "ejecutar_python", "arguments": {"codigo": "print(6*7)"}}}
                ],
            },
            "Son 42, Juan.",
        ]
    )
    agent = Agent(llm, memory, ToolBox(memory, confirm=lambda a, d: True), auto_memory=False)
    assert agent.respond("¿Cuánto es 6*7? Soy Juan") == "Son 42, Juan."
    system = llm.calls[0]["messages"][0]["content"]
    assert "El usuario se llama Juan" in system
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "content": "42", "tool_name": "ejecutar_python"}
    assert agent.history[-1] == {"role": "assistant", "content": "Son 42, Juan."}


def test_agent_learns_facts(memory):
    llm = FakeLLM(["¡Encantado!", '{"hechos": ["El usuario es médico"]}'])
    agent = Agent(llm, memory)
    agent.respond("Soy médico")
    assert [m.content for m in memory.all()] == ["El usuario es médico"]
    assert llm.calls[1]["json_mode"]


def test_refine_corrects_until_verified():
    llm = FakeLLM(
        [
            '{"correcta": false, "problemas": ["17*23 no es 381"]}',
            "17*23 = 391",
            '{"correcta": true, "problemas": []}',
        ]
    )
    statuses = []
    out = refine(llm, [{"role": "user", "content": "17*23?"}], "17*23?", "381", on_status=statuses.append)
    assert out == "17*23 = 391"
    assert "17*23 no es 381" in llm.calls[1]["messages"][-1]["content"]
    assert statuses[-1] == "respuesta verificada ✓"


def test_deep_mode_streams_only_final_answer(memory):
    llm = FakeLLM(["borrador", '{"correcta": false, "problemas": ["x"]}', "final", '{"correcta": true}'])
    agent = Agent(llm, memory, auto_memory=False)
    shown = []
    assert agent.respond("pregunta", deep=True, on_token=shown.append) == "final"
    assert shown == ["final"]


def test_ollama_client_streaming():
    chunks = [
        {"message": {"content": "<think>pienso</think>"}, "done": False},
        {"message": {"content": "Hola"}, "done": False},
        {"message": {"content": " mundo", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            for c in chunks:
                self.wfile.write((json.dumps(c) + "\n").encode())

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaClient("qwen3:8b", host=f"http://127.0.0.1:{server.server_port}")
        shown = []
        reply = client.chat([{"role": "user", "content": "hola"}], tools=[{"x": 1}], on_token=shown.append, json_mode=True)
    finally:
        server.shutdown()
    assert reply["content"] == "Hola mundo"
    assert reply["tool_calls"][0]["function"]["name"] == "f"
    assert "".join(shown) == "Hola mundo"
    assert received["body"]["model"] == "qwen3:8b"
    assert received["body"]["format"] == "json"
    assert received["body"]["tools"] == [{"x": 1}]
