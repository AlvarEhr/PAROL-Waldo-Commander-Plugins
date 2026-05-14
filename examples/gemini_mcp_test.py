"""Demo: Gemini API driving the parol6_mcp server.

Reads GEMINI_API_KEY from parol6-vision/.env, connects to a running
parol6_mcp server at http://127.0.0.1:8080/mcp, and runs four prompts
that exercise read + motion tools. Watch the sim robot in
Waldo-Commander while Gemini autonomously routes through MCP.

Prerequisites:
    Terminal 1: parol6-server (sim/real toggled via the GUI button)
    Terminal 2: waldo-commander GUI (toggle sim mode before running this)
    Terminal 3: python examples/run_mcp_demo.py --connect-parol6
    Terminal 4: python examples/gemini_mcp_test.py     (this script)

Install (one-time):
    pip install google-genai mcp python-dotenv

The four prompts go from read-only (joints, pose) to a motion command
that should be visible in the WC GUI. Gemini decides which tools to
call; this script prints every tool invocation + result and the final
text reply per prompt.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


_ENV_FILE = (
    Path(__file__).resolve().parent.parent.parent / "parol6-vision" / ".env"
)
# Avoid clashing with Waldo-Commander on 8080; run the demo on 8765
# unless MCP_DEMO_PORT is set (matches run_mcp_demo.py's convention).
_MCP_PORT = int(os.environ.get("MCP_DEMO_PORT", "8765"))
_MCP_URL = f"http://127.0.0.1:{_MCP_PORT}/mcp"
_MODEL = "gemini-2.5-pro"

_PROMPTS = [
    "What are the robot's current joint angles?",
    "What's the TCP pose in the WRF frame right now?",
    (
        "Get the current joint angles. Then call parol6_move_j to drive the "
        "robot to those same joints but with J1 (the second joint, index 1) "
        "increased by 20 degrees. Use a moderate speed around 0.25 and "
        "accel around 0.5. Wait for motion to finish."
    ),
    (
        "Now return the robot to the joint angles you read at the very "
        "start. Same speed and accel as before."
    ),
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


def _print_afc_step(afc_history: object) -> None:
    """Best-effort summary of the auto-function-calling steps Gemini took."""
    if not afc_history:
        return
    print("  --- tool-call trace ---")
    for step in afc_history:
        parts = getattr(step, "parts", []) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            fr = getattr(part, "function_response", None)
            if fc:
                print(f"  [CALL]   {fc.name}({dict(fc.args)})")
            elif fr:
                payload = str(fr.response)
                if len(payload) > 200:
                    payload = payload[:197] + "..."
                print(f"  [RESULT] {fr.name}: {payload}")


async def _run_prompts(
    gemini: "genai.Client",
    session: ClientSession,
) -> None:
    for i, prompt in enumerate(_PROMPTS, 1):
        print(f"\n{'=' * 60}\nPROMPT {i}/{len(_PROMPTS)}\n{'=' * 60}")
        print(f"USER: {prompt}\n")
        resp = await gemini.aio.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(tools=[session]),
        )
        _print_afc_step(getattr(resp, "automatic_function_calling_history", None))
        text = (resp.text or "").strip()
        print(f"\nGEMINI: {text or '<no text reply>'}")


async def main() -> None:
    api_key = _load_api_key()
    print(f"Connecting to MCP server at {_MCP_URL} ...")
    try:
        async with streamablehttp_client(_MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(t.name for t in tools.tools)
                print(f"MCP tools advertised ({len(names)}):")
                for n in names:
                    print(f"  - {n}")

                gemini = genai.Client(api_key=api_key)
                await _run_prompts(gemini, session)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print(
            "\nIs the MCP server running with --connect-parol6?\n"
            '  cd /d "C:\\Users\\alvar\\OneDrive\\Desktop\\Project Files\\Waldo-Commander-plugins"\n'
            "  python examples/run_mcp_demo.py --connect-parol6"
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
