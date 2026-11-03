"""Fixtures and helpers shared by the tests that talk to a fake SpatialReal backend."""

import asyncio
from collections.abc import Callable

import pytest
from fake_driveningress import FakeDriveningress

from pipecat_spatialreal.driven import DrivenIngress

# Clients built by make(); the fake fixture closes them before it shuts the fake down.
_clients: list[DrivenIngress] = []


def pcm_ms(ms: int, rate: int) -> bytes:
    """Return ``ms`` milliseconds of digital silence as PCM16 mono at ``rate`` Hz.

    Args:
        ms: Length in ms.
        rate: Sample rate in Hz.

    Returns:
        ``rate * ms // 1000`` zeroed samples, two bytes each.
    """
    return bytes(2 * (rate * ms // 1000))


def make(fake: FakeDriveningress, **overrides) -> DrivenIngress:
    """Build a ``DrivenIngress`` for the fake, not yet connected.

    The ``fake`` fixture closes it at teardown, so a test need not.

    Args:
        fake: The entered fake backend to point the client at.
        **overrides: Constructor arguments that replace the defaults, for example
            ``on_segment=got.append``. The defaults are credentials ``k``, ``a`` and ``av``,
            24 kHz audio, and callbacks that do nothing.

    Returns:
        The client.
    """
    arguments = {
        "api_key": "k",
        "app_id": "a",
        "avatar_id": "av",
        "sample_rate": 24000,
        "console_endpoint_url": fake.console_url,
        "ingress_endpoint_url": fake.ingress_url,
        "on_segment": lambda segment: None,
        "on_down": lambda reason: None,
    }
    client = DrivenIngress(**{**arguments, **overrides})
    _clients.append(client)
    return client


async def until(condition: Callable[[], object], timeout: float = 2.0) -> None:
    """Wait until ``condition()`` is true; fail the test if it is not within ``timeout`` seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out waiting for the condition"
        await asyncio.sleep(0.01)


@pytest.fixture
async def fake():
    """An entered ``FakeDriveningress`` with the default segment sizes (200 ms and 200 ms)."""
    async with FakeDriveningress() as backend:
        yield backend
        while _clients:
            await _clients.pop().close()
