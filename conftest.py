from datetime import datetime


def is_controller(config):
    # controller has no workerinput attribute
    return not hasattr(config, "workerinput")


def pytest_configure(config):
    # only runs once, on the controller process
    if is_controller(config):
        config.time_str = datetime.now().strftime("%Y%m%d_%H%M%S")


def pytest_configure_node(node):
    # xdist hook: before a worker starts, copy the controller's timestamp in
    node.workerinput["time_str"] = node.config.time_str


def pytest_generate_tests(metafunc):
    # for any test that asks for a `time_str` argument,
    # inject exactly one parameter (the same timestamp in all workers)
    if "time_str" in metafunc.fixturenames:
        timestamp = (
            metafunc.config.time_str if is_controller(metafunc.config) else metafunc.config.workerinput["time_str"]
        )
        metafunc.parametrize("time_str", [timestamp], scope="session")
