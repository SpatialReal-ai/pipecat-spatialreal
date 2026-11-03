"""``play``, the runner in conftest.py, fails a test instead of hanging it.

Both tests shorten ``WAIT_S``, the longest ``play`` waits for the pipeline, so that they take a
fraction of a second where a failing test would take ten.
"""

import conftest
import pytest
from conftest import Until, play
from pipecat.frames.frames import EndFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor


class SwallowsEnd(FrameProcessor):
    """Passes every frame on except the ``EndFrame``, so the pipeline never ends by itself."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)


async def test_a_pipeline_that_does_not_end_fails_the_test(monkeypatch):
    monkeypatch.setattr(conftest, "WAIT_S", 0.5)

    with pytest.raises(AssertionError, match="did not end within 0.5 s"):
        await play(SwallowsEnd(), [TextFrame("hello")])


async def test_a_step_that_never_comes_true_fails_the_test(monkeypatch):
    monkeypatch.setattr(conftest, "WAIT_S", 0.3)

    with pytest.raises(AssertionError, match="timed out waiting for the moon"):
        await play(SwallowsEnd(), [Until("the moon", lambda: False)])
