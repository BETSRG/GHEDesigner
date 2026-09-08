#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from werkzeug.serving import make_server

from ghedesigner.gui.log import LOGGER, configure_gui_logging, default_log_path
from ghedesigner.gui.server import create_app


def main_gui(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the GHEDesigner input editor.")
    parser.add_argument("--host", default="127.0.0.1", help="Local interface to bind.")
    parser.add_argument("--port", default=0, type=int, help="Port to bind; zero selects an available port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the editor in a browser.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_log_path(),
        help="Diagnostic log path (default: ~/.ghedesigner/gui.log).",
    )
    parser.add_argument("--debug-log", action="store_true", help="Include detailed debug events in the log.")
    args = parser.parse_args(argv)

    log_path = configure_gui_logging(args.log_file, args.debug_log)
    app = create_app()
    server = make_server(args.host, args.port, app, threaded=True)
    url = f"http://{args.host}:{server.server_port}"
    print(f"GHEDesigner GUI running at {url}")
    print(f"Diagnostic log: {log_path}")
    LOGGER.info("session_started url=%s debug=%s log_file=%s", url, args.debug_log, log_path)
    if not args.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("session_interrupted")
    except Exception:
        LOGGER.exception("session_failed")
        raise
    finally:
        app.extensions["ghedesigner_simulations"].shutdown()
        server.server_close()
        LOGGER.info("session_stopped")
        logging.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_gui())
