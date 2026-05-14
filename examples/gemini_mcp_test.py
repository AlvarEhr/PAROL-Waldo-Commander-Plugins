"""Interactive Gemini chat driving the parol6_mcp server.

Reads GEMINI_API_KEY from parol6-vision/.env, connects to a running
parol6_mcp server (default port 8765, override with MCP_DEMO_PORT),
prints the tool list Gemini can see, and drops you into a REPL where
you type prompts and Gemini autonomously routes through MCP tools.

Prerequisites:
    Terminal 1: parol6-server (sim/real toggled via the GUI button)
    Terminal 2: waldo-commander GUI (toggle SIM mode before running this)
    Terminal 3: set MCP_DEMO_PORT=8765 (Windows: ``set ...``)
                python examples/run_mcp_demo.py --connect-parol6
    Terminal 4: python examples/gemini_mcp_test.py     (this script)

Install (one-time):
    pip install google-genai mcp python-dotenv

Type /quit (or /exit, or Ctrl+C) to leave.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


_ENV_FILE = (
    Path(__file__).resolve().parent.parent.parent / "parol6-vision" / ".env"
)
_MCP_PORT = int(os.environ.get("MCP_DEMO_PORT", "8765"))
_MCP_URL = f"http://127.0.0.1:{_MCP_PORT}/mcp"
_MODEL = "gemini-2.5-pro"

_QUIT = {"/quit", "/exit", "quit", "exit"}

_EXAMPLE_PROMPTS = [
    "What are the robot's current joint angles?",
    "What's the TCP pose in the WRF frame right now?",
    "Move J1 by +20 degrees from current. Use speed 0.25, accel 0.5.",
    "Now move it back to where it was.",
    "Check whether moving from current joints to all zeros would collide.",
    "Halt the robot.",
]


def _load_api_key() -> str:
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
        print(f"Loaded env from {_ENV_FILE}")
    else:
        print(f"Note: {_ENV_FILE} not found; relying on existing env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GEMINI_API_KEY not set. Add it to parol6-vision/.env or your env."
        )
    return key


def _print_tools(tool_objects: list[Any]) -> None:
    """Print MCP tools available to Gemini with one-line descriptions."""
    print(f"\nMCP tools advertised ({len(tool_objects)}):")
    width = max(len(t.name) for t in tool_objects) + 2
    for t in tool_objects:
        desc_first = (t.description or "").strip().split("\n")[0]
        if len(desc_first) > 90:
            desc_first = desc_first[:87] + "..."
        print(f"  {t.name:<{width}} {desc_first}")


def _print_examples() -> None:
    print("\nExample prompts to try:")
    for p in _EXAMPLE_PROMPTS:
        print(f"  > {p}")
    print()


def _print_afc_steps(history: Any) -> None:
    """Best-effort summary of Gemini's tool-call steps for the last turn."""
    if not history:
        return
    for step in history:
        parts = getattr(step, "parts", []) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            if fc:
                args_repr = dict(fc.args) if fc.args else {}
                print(f"  [CALL]   {fc.name}({args_repr})")
            elif fr:
                payload = str(fr.response)
                if len(payload) > 240:
                    payload = payload[:237] + "..."
                print(f"  [RESULT] {fr.name}: {payload}")


async def _chat_loop(
    gemini: genai.Client,
    session: ClientSession,
) -> None:
    chat = gemini.aio.chats.create(
        model=_MODEL,
        config=types.GenerateContentConfig(tools=[session]),
    )
    print(
        "\nReady. Type your prompts at the > prompt. "
        "/quit (or /exit, Ctrl+C) to leave.\n"
    )
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not text:
            continue
        if text.lower() in _QUIT:
            print("Bye.")
            return
        try:
            resp = await chat.send_message(text)
        except KeyboardInterrupt:
            print("\nBye.")
            return
        except Exception as e:  # noqa: BLE001
            print(f"\nERROR: {type(e).__name__}: {e}\n")
            continue
        _print_afc_steps(
            getattr(resp, "automatic_function_calling_history", None),
        )
        reply = (resp.text or "").strip()
        print(f"\nGEMINI: {reply or '<no text reply>'}\n")


async def main() -> None:
    api_key = _load_api_key()
    print(f"Connecting to MCP server at {_MCP_URL} ...")
    try:
        async with streamablehttp_client(_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                _print_tools(tools.tools)
                _print_examples()
                gemini = genai.Client(api_key=api_key)
                await _chat_loop(gemini, session)
    except (OSError, ConnectionError) as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print(
            f"\nIs the MCP server running on port {_MCP_PORT}?\n"
            '  cd /d "C:\\Users\\alvar\\OneDrive\\Desktop\\Project Files\\Waldo-Commander-plugins"\n'
            f"  set MCP_DEMO_PORT={_MCP_PORT}\n"
            "  python examples/run_mcp_demo.py --connect-parol6"
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
