"""Minimal example plugin demonstrating the WaldoPlugin contract. See PLUGIN_CONTRACT.md for the full surface."""

from __future__ import annotations

import logging
import time
from typing import Any

from nicegui import ui
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "hello_world"
    version: str = "0.1.0"
    display_name: str = "Hello World"
    description: str = "Minimal example plugin. Says hello and shows when it loaded."


class Plugin:
    manifest = PluginManifest()

    def __init__(self) -> None:
        self._loaded_at: float = 0.0

    async def on_load(self, host: Any) -> None:
        self._loaded_at = time.time()
        logger.info("hello_world: loaded at %.0f", self._loaded_at)

    def register_panels(self, host: Any, panels: Any) -> None:
        def _build() -> None:
            ui.label("Hello from hello_world plugin").classes("text-sm font-medium")
            elapsed = ui.label("").classes("text-xs opacity-70")

            def _refresh() -> None:
                elapsed.text = f"loaded {time.time() - self._loaded_at:.0f}s ago"

            _refresh()
            ui.button("Refresh", icon="refresh", on_click=_refresh).props("flat dense size=sm")

        panels.add("plugins", self.manifest.display_name, _build)

    async def on_unload(self, host: Any) -> None:
        logger.info("hello_world: unloaded after %.0fs", time.time() - self._loaded_at)
