"""Fixtures and helpers shared by the tests that talk to a fake SpatialReal backend."""

import pytest
from fake_driveningress import FakeDriveningress


def pcm_ms(ms: int, rate: int) -> bytes:
    """Return ``ms`` milliseconds of digital silence as PCM16 mono at ``rate`` Hz.

    Args:
        ms: Length in ms.
        rate: Sample rate in Hz.

    Returns:
        ``rate * ms // 1000`` zeroed samples, two bytes each.
    """
    return bytes(2 * (rate * ms // 1000))


@pytest.fixture
async def fake():
    """An entered ``FakeDriveningress`` with the default segment sizes (200 ms and 200 ms)."""
    async with FakeDriveningress() as backend:
        yield backend
