"""Fixtures and helpers shared by the tests that talk to a fake SpatialReal backend.

``play`` and its steps run a processor in a Pipecat test pipeline and wait for the pipeline instead
of for a time; they are at the end of this file.
"""

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from fake_driveningress import FakeDriveningress
from pipecat.frames.frames import EndFrame, InterruptionFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.tests.utils import QueuedFrameProcessor, SleepFrame
from pipecat.workers.runner import WorkerRunner

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


# --- Waiting for the pipeline instead of for a time
#
# run_test feeds its frames on a timetable, and a SleepFrame is a guess at how long the pipeline
# needs. On a loaded machine the guess is wrong: the first connect to the fake took 600 ms and more,
# which moved every scenario that began with a short sleep. play() feeds frames the same way, but
# its steps can also wait for the pipeline itself.

# The longest a step waits for the pipeline. Only a test that is going to fail waits this long.
WAIT_S = 10.0


@dataclass
class Until:
    """A step for ``play``: wait here until ``condition()`` is true."""

    what: str  # named in the failure
    condition: Callable[[], object]


@dataclass
class Do:
    """A step for ``play``: call ``action()`` here, and await what it returns if it is awaitable."""

    action: Callable[[], object]


async def play(processor, steps, *, send_end_frame=True):
    """Run ``processor`` in a test pipeline and feed it ``steps``, as ``run_test`` does.

    The result is the same too: the frames that came out downstream and upstream. A step is a frame
    to queue, a ``SleepFrame``, an ``Until`` that waits for the pipeline to get somewhere, or a
    ``Do`` that acts. Use a sleep to set a duration that is the point of the scenario, and an
    ``Until`` wherever the scenario needs the processor to have got somewhere first.

    The body mirrors Pipecat's ``run_test`` (``pipecat.tests.utils``) and is written against the
    same ``PipelineWorker`` and ``WorkerRunner``. If a Pipecat release renames or reshapes them,
    this helper breaks first, and the weekly compatibility job turns red for a reason that lies in
    the tests, not in the package: port the body of ``run_test`` here.

    Args:
        processor: The processor, or a pipeline of processors, under test.
        steps: What to do, in order.
        send_end_frame: Whether to end the pipeline with an ``EndFrame`` after the last step.

    Returns:
        ``(downstream frames, upstream frames)``, the ``EndFrame`` not included.

    Raises:
        AssertionError: If a step waits longer than ``WAIT_S`` for the pipeline.
    """
    received_up, received_down = asyncio.Queue(), asyncio.Queue()
    source = QueuedFrameProcessor(queue=received_up, queue_direction=FrameDirection.UPSTREAM)
    sink = QueuedFrameProcessor(queue=received_down, queue_direction=FrameDirection.DOWNSTREAM)
    worker = PipelineWorker(
        Pipeline([source, processor, sink]), cancel_on_idle_timeout=False, enable_rtvi=False
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_started(worker, frame):
        started.set()

    async def feed():
        await asyncio.wait_for(started.wait(), timeout=WAIT_S)
        for step in steps:
            if isinstance(step, SleepFrame):
                await asyncio.sleep(step.sleep)
            elif isinstance(step, Until):
                try:
                    await until(step.condition, timeout=WAIT_S)
                except AssertionError:
                    raise AssertionError(f"timed out waiting for {step.what}") from None
            elif isinstance(step, Do):
                result = step.action()
                if inspect.isawaitable(result):
                    await result
            else:
                await worker.queue_frame(step)
        if send_end_frame:
            await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await feed()
        await running
    except BaseException:
        running.cancel()  # a step failed: do not leave the pipeline running behind the test
        await asyncio.gather(running, return_exceptions=True)
        raise

    down = []
    while not received_down.empty():
        frame = received_down.get_nowait()
        if not (send_end_frame and isinstance(frame, EndFrame)):
            down.append(frame)
    up = []
    while not received_up.empty():
        up.append(received_up.get_nowait())
    return down, up


def link_up(proc):
    """The step that waits for the processor's driveningress session to be up."""
    return Until("the driveningress session to be up", lambda: proc._driven.connected)


def tts_audio_released(pushed, count=1):
    """The step that waits until ``count`` frames of TTS audio have been pushed."""
    return Until(
        f"{count} frame(s) of TTS audio to be released",
        lambda: sum(isinstance(f, TTSAudioRawFrame) for _, f in pushed) >= count,
    )


def message_pushed(pushed, member, count=1):
    """The step that waits until ``count`` avatar messages of one kind have been pushed."""
    return Until(
        f"{count} {member} message(s) to be pushed",
        lambda: (
            sum(isinstance(f, RTVIServerMessageFrame) and member in f.data for _, f in pushed)
            >= count
        ),
    )


def interruption_pushed(pushed):
    """The step that waits until the processor has passed an ``InterruptionFrame`` on."""
    return Until(
        "the InterruptionFrame to be passed on",
        lambda: any(isinstance(f, InterruptionFrame) for _, f in pushed),
    )
