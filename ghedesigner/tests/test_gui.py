from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from ghedesigner.gui.file_picker import PathKind
from ghedesigner.gui.log import LOGGER, configure_gui_logging
from ghedesigner.gui.server import create_app
from ghedesigner.validate import validate_input_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMOS_DIRECTORY = PROJECT_ROOT / "demos"


def _demo(name: str) -> dict:
    return json.loads((DEMOS_DIRECTORY / name).read_text())


def _successful_simulation_command(_input_path: Path, output_directory: Path) -> list[str]:
    script = (
        "from pathlib import Path; import sys; Path(sys.argv[1], 'result.txt').write_text('done'); print('fake run')"
    )
    return [sys.executable, "-c", script, str(output_directory)]


def _slow_simulation_command(_input_path: Path, _output_directory: Path) -> list[str]:
    return [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"]


def _wait_for_job(client, job_id: str, terminal_statuses: set[str] | None = None) -> dict:
    terminal_statuses = terminal_statuses or {"completed", "failed", "cancelled"}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/simulations/{job_id}")
        assert response.status_code == 200
        job = response.get_json()
        if job["status"] in terminal_statuses:
            return job
        time.sleep(0.01)
    pytest.fail("Simulation job did not reach a terminal status.")
    raise AssertionError


def test_validate_input_data_returns_structured_diagnostics() -> None:
    diagnostics = validate_input_data({"version": 4})

    assert diagnostics
    assert diagnostics[0].severity == "error"
    assert diagnostics[0].pointer.startswith("/")
    assert diagnostics[0].source == "schema"


def test_gui_serves_assets_and_api(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<main>GHEDesigner editor</main>")
    app = create_app(tmp_path)
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert b"GHEDesigner editor" in client.get("/network").data
    health = client.get("/api/health")
    assert health.get_json() == {"status": "ok"}
    assert health.headers["X-GHEDesigner-Request-ID"]
    assert "$defs" in client.get("/api/schema").get_json()
    assert client.get("/api/examples/not-an-example.json").status_code == 404


def test_gui_schema_gives_every_field_a_natural_language_title() -> None:
    schema = create_app().test_client().get("/api/schema").get_json()
    missing: list[str] = []
    untitled_variants: list[str] = []
    titles: list[str] = []

    def inspect(value: object, path: str = "") -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                for name, field_schema in properties.items():
                    field_path = f"{path}/properties/{name}"
                    if not isinstance(field_schema, dict) or not isinstance(field_schema.get("title"), str):
                        missing.append(field_path)
                    else:
                        titles.append(field_schema["title"])
            for keyword in ("oneOf", "anyOf"):
                variants = value.get(keyword)
                if isinstance(variants, list):
                    for index, variant in enumerate(variants):
                        if isinstance(variant, dict) and not isinstance(variant.get("title"), str):
                            untitled_variants.append(f"{path}/{keyword}/{index}")
            for name, child in value.items():
                inspect(child, f"{path}/{name}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}/{index}")

    inspect(schema)

    assert schema["title"] == "GHEDesigner input file"
    assert all(definition.get("title") for definition in schema["$defs"].values())
    assert missing == []
    assert untitled_variants == []
    assert all("_" not in title for title in titles)
    assert schema["$defs"]["rho_cp"]["title"] == "Volumetric heat capacity"
    assert (
        schema["properties"]["ground_heat_exchanger"]["additionalProperties"]["properties"]["flow_rate"]["title"]
        == "Design flow per borehole"
    )
    assert schema["properties"]["fluid"]["properties"]["concentration_percent"]["title"] == ("Antifreeze concentration")
    assert schema["$defs"]["file_path"]["format"] == "file-path"


def test_gui_schema_gives_every_numeric_field_units() -> None:
    schema = create_app().test_client().get("/api/schema").get_json()
    missing: list[str] = []

    def is_numeric(field_schema: dict) -> bool:
        field_type = field_schema.get("type")
        if field_type in {"integer", "number"}:
            return True
        items = field_schema.get("items")
        return field_type == "array" and isinstance(items, dict) and is_numeric(items)

    def inspect(value: object, path: str = "") -> None:
        if isinstance(value, dict):
            if isinstance(value.get("title"), str) and is_numeric(value):
                description = value.get("description")
                if not isinstance(description, str) or "Units:" not in description:
                    missing.append(path)
            for name, child in value.items():
                inspect(child, f"{path}/{name}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}/{index}")

    inspect(schema)

    assert missing == []
    assert schema["$defs"]["rho_cp"]["description"].endswith("Units: joules per cubic meter-kelvin (J/m³·K).")
    assert schema["$defs"]["ground_temperature_model"]["properties"]["phase_lag_1"]["description"].endswith(
        "Units: days."
    )
    assert schema["$defs"]["network_pump"]["required"] == ["wire_to_water_efficiency"]


def test_gui_schema_exposes_only_prescribed_flow_network_inputs() -> None:
    schema = create_app().test_client().get("/api/schema").get_json()

    assert "hydraulic_mode" not in schema["$defs"]["compact_network"]["properties"]
    assert "control" not in schema["$defs"]["pump_hydraulics"]["properties"]
    assert "maximum_pressure" not in schema["$defs"]["network_pump"]["properties"]
    assert "pressure_curve" not in schema["$defs"]["network_pump"]["properties"]
    assert "minimum_speed_fraction" not in schema["$defs"]["network_pump"]["properties"]
    assert "maximum_speed_fraction" not in schema["$defs"]["network_pump"]["properties"]
    assert "passive_components" not in schema["$defs"]["compact_network"]["properties"]
    assert "bypass_components" not in schema["$defs"]["compact_network"]["properties"]
    assert "graph_network" not in schema["$defs"]
    assert "network_pipe" not in schema["properties"]
    assert "circulation_pump" not in schema["properties"]
    assert "bypass" not in schema["properties"]


def test_single_u_tube_schema_omits_redundant_pipe_count() -> None:
    schema = create_app().test_client().get("/api/schema").get_json()
    single_u_tube = next(
        option
        for option in schema["$defs"]["pipe"]["oneOf"]
        if option["properties"]["arrangement"].get("const") == "SINGLEUTUBE"
    )

    assert "num_pipes" not in single_u_tube["properties"]


def test_gui_chooses_files_and_directories_on_the_local_computer() -> None:
    calls: list[tuple[PathKind, str | None]] = []

    def choose(kind: PathKind, initial_path: str | None = None) -> str | None:
        calls.append((kind, initial_path))
        return "/data/loads.csv" if kind == "file" else None

    client = create_app(path_chooser=choose).test_client()

    selected = client.post("/api/choose-path", json={"kind": "file", "initial_path": "/data/old.csv"})
    cancelled = client.post("/api/choose-path", json={"kind": "directory", "initial_path": "/output"})

    assert selected.status_code == 200
    assert selected.get_json() == {"path": "/data/loads.csv"}
    assert cancelled.status_code == 200
    assert cancelled.get_json() == {"path": None}
    assert calls == [("file", "/data/old.csv"), ("directory", "/output")]


def test_gui_rejects_remote_path_chooser_requests() -> None:
    client = create_app().test_client()

    response = client.post(
        "/api/choose-path",
        json={"kind": "file"},
        environ_base={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert response.status_code == 403


def test_gui_opens_the_selected_output_directory(tmp_path: Path) -> None:
    opened: list[Path] = []
    output_directory = tmp_path / "new-output"

    def open_selected_directory(directory: Path) -> None:
        opened.append(directory)

    client = create_app(directory_opener=open_selected_directory).test_client()

    response = client.post("/api/open-directory", json={"path": str(output_directory)})

    assert response.status_code == 200
    assert response.get_json() == {"path": str(output_directory)}
    assert output_directory.is_dir()
    assert opened == [output_directory]


def test_gui_rejects_remote_directory_open_requests(tmp_path: Path) -> None:
    client = create_app(directory_opener=lambda _path: None).test_client()

    response = client.post(
        "/api/open-directory",
        json={"path": str(tmp_path)},
        environ_base={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert response.status_code == 403


def test_gui_includes_packaged_examples() -> None:
    client = create_app().test_client()

    response = client.get("/api/examples/pre_designed_manual.json")

    assert response.status_code == 200
    assert response.get_json()["version"] == 4


def test_packaged_examples_resolve_to_packaged_load_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ghedesigner.gui.server.DEMOS_DIRECTORY", Path("/not-installed"))
    client = create_app().test_client()

    for name in (
        "simulate_1_pipe_1_ghe_1_bldg_district.json",
        "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json",
    ):
        response = client.get(f"/api/examples/{name}")

        assert response.status_code == 200
        first_building = next(iter(response.get_json()["building"].values()))
        load_path = Path(first_building["heating_load"]["file_path"])
        assert load_path.is_absolute()
        assert load_path.is_file()


def test_gui_resolves_example_load_paths_before_returning_document() -> None:
    client = create_app().test_client()

    response = client.get("/api/examples/Network_Sizing_3GHE_6HP_RowWise.json")

    assert response.status_code == 200
    document = response.get_json()
    load_path = Path(document["building"]["bldg_1_zone_1"]["heating_load"]["file_path"])
    assert load_path.is_absolute()
    assert load_path == (DEMOS_DIRECTORY / "../ghedesigner/tests/test_data/Test_Case_1_Loads.csv").resolve()
    assert load_path.is_file()
    stored_path = _demo("Network_Sizing_3GHE_6HP_RowWise.json")["building"]["bldg_1_zone_1"]["heating_load"][
        "file_path"
    ]
    assert not Path(stored_path).is_absolute()


def test_gui_classifies_examples_by_workflow() -> None:
    examples = create_app().test_client().get("/api/examples").get_json()
    workflows = {example["name"]: example["workflow_mode"] for example in examples}

    assert workflows["pre_designed_manual.json"] == "g_function"
    assert workflows["find_design_rectangle_single_u_tube.json"] == "standalone_design"
    assert workflows["simulate_1_pipe_1_ghe_1_bldg_district.json"] == "district_simulation"


def test_packaged_gui_has_pre_module_startup_diagnostics() -> None:
    response = create_app().test_client().get("/")

    assert response.status_code == 200
    assert b"browser_startup_failed" in response.data


def test_gui_disables_browser_caching_for_frontend_assets(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<main>Editor</main>")
    (tmp_path / "application.js").write_text("console.log('loaded');")
    client = create_app(tmp_path).test_client()

    assert client.get("/").headers["Cache-Control"] == "no-store"
    assert client.get("/application.js").headers["Cache-Control"] == "no-store"


def test_gui_validation_reports_bad_request() -> None:
    client = create_app().test_client()

    response = client.post("/api/validate", data="[]", content_type="application/json")

    assert response.status_code == 400
    assert response.get_json()["valid"] is False


def test_gui_accepts_browser_diagnostics() -> None:
    client = create_app().test_client()

    response = client.post(
        "/api/client-log",
        json={"level": "warning", "event": "browser_event_loop_stall", "details": {"delay_ms": 4200}},
    )

    assert response.status_code == 204


def test_gui_runs_simulation_without_blocking_request_thread(tmp_path: Path) -> None:
    app = create_app(simulation_command_factory=_successful_simulation_command)
    client = app.test_client()

    started = client.post(
        "/api/simulations",
        json={
            "document": _demo("pre_designed_manual.json"),
            "output_directory": str(tmp_path / "output"),
            "input_name": "edited-input.json",
        },
    )

    assert started.status_code == 202
    assert started.get_json()["status"] in {"queued", "running"}
    job = _wait_for_job(client, started.get_json()["id"])
    assert job["status"] == "completed"
    assert job["return_code"] == 0
    assert "fake run" in job["output"]
    assert (tmp_path / "output" / "result.txt").read_text() == "done"
    app.extensions["ghedesigner_simulations"].shutdown()


def test_gui_rejects_invalid_simulation_input(tmp_path: Path) -> None:
    client = create_app(simulation_command_factory=_successful_simulation_command).test_client()

    response = client.post(
        "/api/simulations",
        json={"document": {"version": 4}, "output_directory": str(tmp_path)},
    )

    assert response.status_code == 422
    assert response.get_json()["diagnostics"]


def test_gui_cancels_active_simulation(tmp_path: Path) -> None:
    app = create_app(simulation_command_factory=_slow_simulation_command)
    client = app.test_client()
    started = client.post(
        "/api/simulations",
        json={"document": _demo("pre_designed_manual.json"), "output_directory": str(tmp_path)},
    ).get_json()

    _wait_for_job(client, started["id"], {"running"})
    response = client.delete(f"/api/simulations/{started['id']}")

    assert response.status_code == 200
    assert _wait_for_job(client, started["id"])["status"] == "cancelled"
    app.extensions["ghedesigner_simulations"].shutdown()


def test_gui_writes_rotating_diagnostic_log(tmp_path: Path) -> None:
    log_path = configure_gui_logging(tmp_path / "gui.log", debug=True)

    LOGGER.info("test_diagnostic_event")
    for handler in LOGGER.handlers:
        handler.flush()

    assert log_path.is_file()
    assert "test_diagnostic_event" in log_path.read_text()


@pytest.mark.parametrize(
    "name",
    [
        "simulate_1_pipe_1_ghe_1_bldg_district.json",
        "simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json",
    ],
)
def test_gui_compiles_representative_networks(name: str) -> None:
    document = _demo(name)
    client = create_app().test_client()

    response = client.post("/api/compile-network", json=document)

    assert response.status_code == 200
    preview = response.get_json()
    assert preview["nodes"]
    assert preview["branches"]
