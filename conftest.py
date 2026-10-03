"""The verification suite under pytest, set up as check.py sets up a run.

    py -3 -m pytest checks [--docker | --real]

Without a flag the container checks skip, as under `check.py --no-docker`.
`--docker` runs them when Docker is up and the image is built; `--real` also
runs every other check in a container, as `check.py --real` does. Checks run
one after another in this process, as under `check.py -j 1`.
"""

from __future__ import annotations

import os

import pytest

import check
from checks.lanes import Skip, configure, docker_ready

# Fixtures and the fake API; neither module holds a check.
collect_ignore = ["checks/fake.py", "checks/lanes.py"]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("checks")
    group.addoption("--docker", action="store_true", help="run the checks that need a container")
    group.addoption(
        "--real", action="store_true", help="run every check in a container, including the ones that need not be"
    )


def pytest_configure(config: pytest.Config) -> None:
    # The provider adapters refuse a redirected endpoint, and every check hands
    # them a fake client.
    for variable in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
        os.environ.pop(variable, None)
    real = config.getoption("real")
    configure(real, (real or config.getoption("docker")) and docker_ready(), os.getpid())


def pytest_unconfigure(config: pytest.Config) -> None:
    if config.getoption("real") or config.getoption("docker"):
        check.sweep()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """A check raising Skip is reported as skipped."""
    try:
        return (yield)
    except Skip:
        pytest.skip("needs Docker + the image")
