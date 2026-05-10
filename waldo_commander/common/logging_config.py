import logging
import logging.handlers
import os
import sys
import threading
import weakref
from pathlib import Path
from typing import Any

from nicegui import ui, Client

_LEVEL_COLORS = {
    "TRACE": "\033[32m",  # green
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[37m",  # light gray
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[41m",  # red background
}

# CSS classes used for NiceGUI log lines (mirrors CLI color semantics)
_LEVEL_CLASSES = {
    "TRACE": "log-trace",
    "DEBUG": "log-debug",
    "INFO": "log-info",
    "WARNING": "log-warning",
    "ERROR": "log-error",
    "CRITICAL": "log-critical",
}

_RESET = "\033[0m"
_DIM = "\033[2m"

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class TraceLogger(logging.Logger):
    """Logger subclass with a ``trace()`` method for sub-DEBUG verbosity."""

    def trace(self, msg: object, *args: object, **kwargs: Any) -> None:
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


# Install trace on the base Logger class so *all* loggers support it
if not hasattr(logging.Logger, "trace"):
    logging.Logger.trace = TraceLogger.trace  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    logging.TRACE = TRACE  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]

# Environment guard to make hot-path trace zero-cost unless explicitly enabled
TRACE_ENABLED = str(os.getenv("WALDO_TRACE", "0")).lower() in ("1", "true", "yes", "on")


class AnsiColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors and a compact timestamp."""

    def __init__(self, colored: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
        self.colored = colored and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.colored:
            return base
        # Colorize level name and dim the timestamp
        level = record.levelname.upper()
        color = _LEVEL_COLORS.get(level, "")
        # Expect format "HH:MM:SS LEVEL logger: msg"
        try:
            ts, rest = base.split(" ", 1)
            if color:
                rest = rest.replace(level, f"{color}{level}{_RESET}", 1)
            return f"{_DIM}{ts}{_RESET} {rest}"
        except Exception:
            return base


# ---- NiceGUI UI log handler ----

_ui_log_targets: set[weakref.ref] = set()
_ui_lock = threading.Lock()


class NiceGuiLogHandler(logging.Handler):
    """Push log records into one or more NiceGUI ui.log widgets."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        # Basic, non-colored format for UI (timestamp + level + message)
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        if not _ui_log_targets:
            return
        msg = self.format(record)
        level = record.levelname.upper()
        classes = _LEVEL_CLASSES.get(level, "log-info")
        stale: list[weakref.ref] = []
        with _ui_lock:
            for ref in list(_ui_log_targets):
                widget = ref()
                if widget is None:
                    stale.append(ref)
                    continue
                # Check if the widget's client is still valid before pushing
                # This prevents the "Client has been deleted" warning from NiceGUI
                try:
                    client_ref = widget._client()
                    if client_ref is None or client_ref.id not in Client.instances:
                        stale.append(ref)
                        continue
                except (RuntimeError, AttributeError):
                    stale.append(ref)
                    continue
                try:
                    widget.push(msg, classes=classes)
                except Exception:
                    # If widget is gone or not available, mark as stale
                    stale.append(ref)
            for ref in stale:
                _ui_log_targets.discard(ref)


def attach_ui_log(log_widget) -> None:
    """Register a ui.log widget as a sink for log records."""
    if ui is None:
        return
    try:
        ref = weakref.ref(log_widget)
    except TypeError:
        return
    with _ui_lock:
        _ui_log_targets.add(ref)


def _have_console_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _have_ui_handler(logger: logging.Logger) -> bool:
    return any(isinstance(h, NiceGuiLogHandler) for h in logger.handlers)


def _have_file_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        for h in logger.handlers
    )


def _resolve_log_file_path() -> Path | None:
    """Resolve the destination path for the rotating-file log.

    Order of precedence:

    1. ``WALDO_LOG_FILE`` env var (explicit override).
    2. ``WALDO_LOG_DIR`` env var (write ``waldo-commander.log`` inside).
    3. The OneDrive-synced ``Project Files\\Waldo Logs\\`` directory if
       it exists — keeps logs visible to the development workflow that
       frequently inspects them across machines.
    4. ``~/.waldo-commander/waldo-commander.log`` as the universal
       fallback (mirrors the workspace-hull cache location).

    Returns None if path resolution + parent-dir creation both fail
    (callers treat that as "no file logging").
    """
    explicit = os.environ.get("WALDO_LOG_FILE")
    if explicit:
        try:
            p = Path(explicit).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            pass

    env_dir = os.environ.get("WALDO_LOG_DIR")
    if env_dir:
        try:
            d = Path(env_dir).expanduser().resolve()
            d.mkdir(parents=True, exist_ok=True)
            return d / "waldo-commander.log"
        except OSError:
            pass

    onedrive_logs = (
        Path.home() / "OneDrive" / "Desktop" / "Project Files" / "Waldo Logs"
    )
    if onedrive_logs.parent.exists():
        try:
            onedrive_logs.mkdir(parents=True, exist_ok=True)
            return onedrive_logs / "waldo-commander.log"
        except OSError:
            pass

    fallback = Path.home() / ".waldo-commander"
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / "waldo-commander.log"
    except OSError:
        return None


def configure_logging(
    level: int = logging.INFO, use_color: bool = True, add_ui_handler: bool = True
) -> logging.Logger:
    """
    Configure root logger with:
      - ANSI-colored console handler (stderr) with timestamps and levels
      - Rotating file handler (waldo-commander.log, 10MB x 5 backups)
      - Optional NiceGUI UI log handler (messages mirrored to web log)
    Idempotent across multiple calls.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if not _have_console_handler(logger):
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(level)
        console.setFormatter(AnsiColorFormatter(colored=use_color))
        logger.addHandler(console)

    if not _have_file_handler(logger):
        log_path = _resolve_log_file_path()
        if log_path is not None:
            try:
                # 10 MB per file, keep 5 backups → up to 60 MB on disk.
                # Plain text (no ANSI colors) so the file is readable
                # without a terminal that interprets escape codes.
                file_handler = logging.handlers.RotatingFileHandler(
                    str(log_path),
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(
                    logging.Formatter(
                        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    ),
                )
                logger.addHandler(file_handler)
            except OSError as exc:
                # Don't fail startup just because the log file can't
                # be opened (e.g. permission issue on a locked-down
                # path). Console + UI handlers still work.
                sys.stderr.write(
                    f"[logging] could not open log file {log_path}: {exc}\n",
                )

    if add_ui_handler and not _have_ui_handler(logger):
        ui_handler = NiceGuiLogHandler(level=level)
        logger.addHandler(ui_handler)

    # Silence verbose third-party loggers. At INFO and above, push noisy
    # libraries up to WARNING so a clean startup stays quiet; honor DEBUG
    # by leaving them at the requested level.
    quiet_level = max(level, logging.WARNING) if level >= logging.INFO else level
    for name in (
        "numba",
        "numba.core",
        "numba.cuda",
        "toppra",
        "uvicorn.access",
        "nicegui",
    ):
        logging.getLogger(name).setLevel(quiet_level)

    return logger
