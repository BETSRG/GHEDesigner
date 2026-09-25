from __future__ import annotations

import ipaddress
import json
import time
import uuid
from pathlib import Path
from typing import Any, cast

from flask import Flask, Response, g, jsonify, request, send_from_directory

from ghedesigner.gui.file_picker import (
    DirectoryOpener,
    DirectoryOpenError,
    PathChooser,
    PathChooserError,
    PathKind,
    choose_path,
    open_directory,
)
from ghedesigner.gui.log import LOGGER
from ghedesigner.gui.simulation import SimulationBusyError, SimulationCommandFactory, SimulationManager
from ghedesigner.network import compile_network, component_type_map
from ghedesigner.validate import load_input_schema, validate_input_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIRECTORY = Path(__file__).with_name("frontend") / "dist"
DEMOS_DIRECTORY = PROJECT_ROOT / "demos"
PACKAGED_EXAMPLES_DIRECTORY = Path(__file__).with_name("examples")
MAX_CLIENT_LOG_LENGTH = 4_000


def _json_body() -> dict[str, Any]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    return body


def _example_paths() -> dict[str, Path]:
    paths = {path.name: path for path in sorted(PACKAGED_EXAMPLES_DIRECTORY.glob("*.json"))}
    if DEMOS_DIRECTORY.is_dir():
        paths.update({path.name: path for path in sorted(DEMOS_DIRECTORY.glob("*.json"))})
    return paths


def _resolve_example_file_paths(value: Any, base_directory: Path) -> int:
    """Resolve relative load paths before an example is copied to a temporary run file."""
    resolved_count = 0
    if isinstance(value, dict):
        for name, child in value.items():
            if name == "file_path" and isinstance(child, str):
                path = Path(child).expanduser()
                if not path.is_absolute():
                    value[name] = str((base_directory / path).resolve())
                    resolved_count += 1
            else:
                resolved_count += _resolve_example_file_paths(child, base_directory)
    elif isinstance(value, list):
        for child in value:
            resolved_count += _resolve_example_file_paths(child, base_directory)
    return resolved_count


def _document_summary(document: dict[str, Any]) -> str:
    network = document.get("network")
    network_type = network.get("type") if isinstance(network, dict) else None
    component_count = sum(
        len(value)
        for key, value in document.items()
        if key
        in {
            "building",
            "ground_heat_exchanger",
            "source_sink_heat_exchanger",
        }
        and isinstance(value, dict)
    )
    return f"root_keys={len(document)} components={component_count} network_type={network_type or 'none'}"


def _workflow_mode(document: dict[str, Any]) -> str:
    """Classify an input using the same execution-path distinctions shown by the GUI."""
    ghes = document.get("ground_heat_exchanger", {})
    ghe_values = list(ghes.values()) if isinstance(ghes, dict) else []
    if isinstance(document.get("network"), dict):
        if ghe_values and all(
            isinstance(ghe, dict) and isinstance(ghe.get("pre_designed"), dict) for ghe in ghe_values
        ):
            return "district_simulation"
        controls = document.get("simulation_control", {})
        if isinstance(controls, dict) and controls.get("search_method") == "SIMULATION_ONLY":
            return "district_simulation"
        return "district_design"
    if document.get("building"):
        return "building_design"
    if ghe_values and all(isinstance(ghe, dict) and isinstance(ghe.get("pre_designed"), dict) for ghe in ghe_values):
        return "g_function"
    return "standalone_design"


def _client_log_details(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(value)
    if len(encoded) > MAX_CLIENT_LOG_LENGTH:
        return encoded[:MAX_CLIENT_LOG_LENGTH] + "...[truncated]"
    return encoded


def _request_is_local() -> bool:
    remote_address = request.remote_addr
    if not remote_address:
        return False
    try:
        return ipaddress.ip_address(remote_address).is_loopback
    except ValueError:
        return remote_address == "localhost"


def _network_preview(document: dict[str, Any]) -> dict[str, Any]:
    network_data = document.get("network")
    if not isinstance(network_data, dict):
        raise ValueError("The document does not contain a network object.")

    graph = compile_network(network_data, component_type_map(document), document)
    return {
        "nodes": [{"id": node.id} for node in graph.nodes.values()],
        "branches": [
            {
                "id": branch.id,
                "type": branch.branch_type.value,
                "node_a": branch.node_a,
                "node_b": branch.node_b,
                "component": branch.component_id,
            }
            for branch in graph.branches.values()
        ],
    }


def create_app(
    static_directory: Path | None = None,
    simulation_command_factory: SimulationCommandFactory | None = None,
    path_chooser: PathChooser = choose_path,
    directory_opener: DirectoryOpener = open_directory,
) -> Flask:
    """Create the local GUI service without starting a server."""
    static_root = (static_directory or DEFAULT_STATIC_DIRECTORY).resolve()
    app = Flask(__name__, static_folder=None)
    simulation_manager = SimulationManager(simulation_command_factory)
    app.extensions["ghedesigner_simulations"] = simulation_manager

    @app.before_request
    def log_request_started() -> None:
        g.request_started = time.perf_counter()
        g.request_id = uuid.uuid4().hex[:10]
        LOGGER.info(
            "request_started id=%s method=%s path=%s content_length=%s user_agent=%s",
            g.request_id,
            request.method,
            request.path,
            request.content_length or 0,
            request.headers.get("User-Agent", "unknown"),
        )

    @app.after_request
    def log_request_finished(response: Response) -> Response:
        started = getattr(g, "request_started", time.perf_counter())
        request_id = getattr(g, "request_id", "unknown")
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-GHEDesigner-Request-ID"] = request_id
        # Route identity is stable across platforms, while MIME type detection is not.
        # In particular, Windows may report JavaScript files as ``text/plain``.
        if request.path.startswith("/api/") or request.endpoint in {"index", "assets"}:
            response.headers["Cache-Control"] = "no-store"
        LOGGER.info(
            "request_finished id=%s method=%s path=%s status=%s duration_ms=%.1f response_length=%s",
            request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            response.calculate_content_length() or 0,
        )
        return response

    @app.teardown_request
    def log_request_exception(error: BaseException | None) -> None:
        if error is not None:
            LOGGER.exception(
                "request_failed id=%s method=%s path=%s",
                getattr(g, "request_id", "unknown"),
                request.method,
                request.path,
                exc_info=error,
            )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/run-settings")
    def run_settings() -> Response:
        return jsonify({"default_output_directory": str((Path.cwd() / "ghedesigner-output").resolve())})

    @app.post("/api/choose-path")
    def select_local_path() -> Response:
        if not _request_is_local():
            return jsonify({"error": "Path browsing is available only from the computer running GHEDesigner."}), 403
        try:
            body = _json_body()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        kind = body.get("kind")
        initial_path = body.get("initial_path")
        if kind not in {"file", "directory"}:
            return jsonify({"error": "Path kind must be 'file' or 'directory'."}), 400
        if initial_path is not None and not isinstance(initial_path, str):
            return jsonify({"error": "The initial path must be a string."}), 400
        try:
            selected_path = path_chooser(cast(PathKind, kind), initial_path)
        except PathChooserError as error:
            LOGGER.exception("path_chooser_failed kind=%s", kind)
            return jsonify({"error": str(error)}), 501
        LOGGER.info("path_chooser_finished kind=%s selected=%s", kind, selected_path is not None)
        return jsonify({"path": selected_path})

    @app.post("/api/open-directory")
    def reveal_local_directory() -> Response:
        if not _request_is_local():
            return jsonify(
                {"error": "Opening directories is available only from the computer running GHEDesigner."}
            ), 403
        try:
            body = _json_body()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        requested_path = body.get("path")
        if not isinstance(requested_path, str) or not requested_path.strip():
            return jsonify({"error": "A directory path is required."}), 400
        directory = Path(requested_path.strip()).expanduser().resolve()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return jsonify({"error": f"Unable to create the output directory: {error}"}), 422
        if not directory.is_dir():
            return jsonify({"error": "The output path is not a directory."}), 422
        try:
            directory_opener(directory)
        except DirectoryOpenError as error:
            LOGGER.exception("directory_open_failed path=%s", directory)
            return jsonify({"error": str(error)}), 501
        LOGGER.info("directory_opened path=%s", directory)
        return jsonify({"path": str(directory)})

    @app.post("/api/client-log")
    def client_log() -> Response:
        try:
            body = _json_body()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        level_name = str(body.get("level", "info")).upper()
        level = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
        }.get(level_name, 20)
        event = str(body.get("event", "browser_event"))[:160]
        LOGGER.log(
            level,
            "browser_event event=%s details=%s",
            event,
            _client_log_details(body.get("details", {})),
        )
        return Response(status=204)

    @app.get("/api/schema")
    def schema() -> Response:
        return jsonify(load_input_schema())

    @app.get("/api/examples")
    def examples() -> Response:
        items = []
        for name, path in _example_paths().items():
            document = json.loads(path.read_text())
            items.append(
                {
                    "name": name,
                    "label": path.stem.replace("_", " "),
                    "network_type": document.get("network", {}).get("type"),
                    "has_buildings": bool(document.get("building")),
                    "workflow_mode": _workflow_mode(document),
                }
            )
        return jsonify(items)

    @app.get("/api/examples/<name>")
    def example(name: str) -> Response:
        path = _example_paths().get(name)
        if path is None:
            return jsonify({"error": "Unknown example."}), 404
        document = json.loads(path.read_text())
        resolved_paths = _resolve_example_file_paths(document, path.parent)
        LOGGER.info(
            "example_loaded name=%s resolved_paths=%s %s",
            name,
            resolved_paths,
            _document_summary(document),
        )
        return jsonify(document)

    @app.post("/api/validate")
    def validate() -> Response:
        try:
            document = _json_body()
        except ValueError as error:
            return jsonify({"valid": False, "diagnostics": [], "error": str(error)}), 400
        LOGGER.debug("validation_started %s", _document_summary(document))
        diagnostics = [diagnostic.as_dict() for diagnostic in validate_input_data(document)]
        LOGGER.info("validation_finished diagnostics=%s %s", len(diagnostics), _document_summary(document))
        return jsonify({"valid": not diagnostics, "diagnostics": diagnostics})

    @app.post("/api/compile-network")
    def network_preview() -> Response:
        try:
            preview = _network_preview(_json_body())
        except (KeyError, TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 422
        return jsonify(preview)

    @app.post("/api/simulations")
    def start_simulation() -> Response:
        try:
            body = _json_body()
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        document = body.get("document")
        output_directory = body.get("output_directory")
        input_name = body.get("input_name", "ghedesigner-input.json")
        if not isinstance(document, dict):
            return jsonify({"error": "The document must be a JSON object."}), 400
        if not isinstance(output_directory, str) or not output_directory.strip():
            return jsonify({"error": "An output directory is required."}), 400
        if not isinstance(input_name, str):
            return jsonify({"error": "The input name must be a string."}), 400

        diagnostics = [diagnostic.as_dict() for diagnostic in validate_input_data(document)]
        if diagnostics:
            return jsonify({"error": "Resolve input diagnostics before running.", "diagnostics": diagnostics}), 422
        try:
            job = simulation_manager.start(document, output_directory.strip(), input_name)
        except SimulationBusyError as error:
            return jsonify({"error": str(error)}), 409
        except ValueError as error:
            return jsonify({"error": str(error)}), 422
        return jsonify(job), 202

    @app.get("/api/simulations/<job_id>")
    def simulation_status(job_id: str) -> Response:
        job = simulation_manager.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown simulation job."}), 404
        return jsonify(job)

    @app.delete("/api/simulations/<job_id>")
    def cancel_simulation(job_id: str) -> Response:
        job = simulation_manager.cancel(job_id)
        if job is None:
            return jsonify({"error": "Unknown simulation job."}), 404
        return jsonify(job)

    @app.get("/")
    def index() -> Response:
        index_path = static_root / "index.html"
        if not index_path.is_file():
            return Response(
                "GHEDesigner GUI assets are not built. Run the frontend build before launching the GUI.",
                status=503,
                mimetype="text/plain",
            )
        return send_from_directory(static_root, "index.html")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.get("/<path:asset_path>")
    def assets(asset_path: str) -> Response:
        candidate = static_root / asset_path
        if candidate.is_file():
            return send_from_directory(static_root, asset_path)
        if (static_root / "index.html").is_file():
            return send_from_directory(static_root, "index.html")
        return Response("GUI asset not found.", status=404, mimetype="text/plain")

    return app
