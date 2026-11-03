"""``play``, the runner in conftest.py, fails a test instead of hanging it.

Both tests shorten ``wait_s``, the longest ``play`` waits for a step and for the pipeline to end,
so that they take a fraction of a second where a failing test would take ten. The wait for the
pipeline to start keeps its full length.
"""

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


async def test_a_pipeline_that_does_not_end_fails_the_test():
    with pytest.raises(AssertionError, match="did not end within 0.5 s"):
        await play(SwallowsEnd(), [TextFrame("hello")], wait_s=0.5)


async def test_a_step_that_never_comes_true_fails_the_test():
    with pytest.raises(AssertionError, match="timed out waiting for the moon"):
        await play(SwallowsEnd(), [Until("the moon", lambda: False)], wait_s=0.3)
