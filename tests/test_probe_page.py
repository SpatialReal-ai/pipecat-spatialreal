"""The probe page: where the bot serves it, and its analysis run against what the package sends.

``examples/probe/index.html`` turns the recorded envelope messages and the onsets found in the
bot's audio into its table, in plain JavaScript. ``probe_check.mjs`` runs that code under Node on
messages built by :mod:`pipecat_spatialreal.envelope` and on random audio scanned by the package's
``OnsetTracker``, so the page's reading of the contract and its onset rule cannot drift from the
processor's. It also holds ``examples/probe/sync.js``, the page's port of the envelope alignment
rule, to the normative oracle's numbers in ``tests/fixtures/sync/oracle.json``, and its port of the
stall rule to the oracle's ticks in ``tests/fixtures/sync/stall.json`` (synthetic stalls) and
``tests/fixtures/sync/daily-stall/oracle-ticks.json`` (a real one); each fixture is checked against
a fresh run of its generator. Without ``node`` the Node test is skipped locally
and fails in CI, where the workflow installs Node; the ones that ask the bot's server for the page
are skipped when the example extra is not installed.
"""

import base64
import importlib.util
import json
import math
import os
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from pipecat_spatialreal import envelope
from pipecat_spatialreal.onset import OnsetTracker
from pipecat_spatialreal.proto.generated import avatar_pb2

PAGE = Path(__file__).resolve().parent.parent / "examples" / "probe" / "index.html"
SYNC = PAGE.with_name("sync.js")
CHECK = Path(__file__).with_name("probe_check.mjs")
ORACLE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sync" / "oracle.json"
STALL_FIXTURE = ORACLE_FIXTURE.with_name("stall.json")
# What a gate_only case of oracle.json is compared on.
GATE_ONLY_KEYS = ("speech", "window_ok", "accepted")

TURN_A = "aaaaaaaa-1111-4111-8111-111111111111"
TURN_B = "bbbbbbbb-2222-4222-8222-222222222222"
TURN_C = "cccccccc-3333-4333-8333-333333333333"


def _envelopes() -> dict:
    """Messages of three turns (a two-segment one, one chained to it, one interrupted), and turn A's
    and turn C's envelopes."""
    return {
        "A_start": envelope.turn_start(TURN_A, sample_rate=24000, onsets_ms=[180.0, 1400.0]),
        "A_seg0": envelope.frames(TURN_A, 0, bytes(range(256)) * 20, end=False),  # 3 chunks
        "A_seg1": envelope.frames(TURN_A, 1, b"\x01" * 1000, end=True),
        "A_end": envelope.turn_end(TURN_A, pcm_ms=3000.0, segment_count=2),
        "B_start": envelope.turn_start(
            TURN_B,
            sample_rate=24000,
            onsets_ms=[0.0],
            prev_turn_id=TURN_A,
            prev_turn_pcm_ms=3000.0,
        ),
        "B_seg0": envelope.frames(TURN_B, 0, b"\x02" * 700, end=True),
        "B_end": envelope.turn_end(TURN_B, pcm_ms=1500.0, segment_count=1),
        "C_start": envelope.turn_start(TURN_C, sample_rate=24000, onsets_ms=[300.0]),
        "C_seg0": envelope.frames(TURN_C, 0, b"\x03" * 3000, end=False),
        "C_abort": envelope.turn_abort(TURN_C),
        "C_onsets": envelope.turn_onsets(TURN_C, [900.0]),
        # Two messages for A (2 400 + 160 values); one for C, sent after its abort in one test.
        "A_env": envelope.turn_envelope(TURN_A, 0, bytes(range(256)) * 10),
        "C_env": envelope.turn_envelope(TURN_C, 0, b"\x05" * 100),
        "degraded": envelope.state(avatar_pb2.State.STATUS_DEGRADED, "driveningress closed"),
    }


def _audio(rate: int = 48000, threshold: int = 328, min_silence_ms: int = 150) -> dict:
    """Random PCM16 of pauses and bursts, with the onsets the processor's rule finds in it.

    The pauses stay well below the threshold and the bursts well above it, and some pauses sit one
    ms either side of the minimum silence, so only the silent-run rule decides where onsets are.
    """
    rng = random.Random(20261102)
    pcm: list[int] = []
    for _ in range(16):
        pause_ms = rng.choice([0, 20, 90, 149, 150, 151, 260, 400])
        pcm += [rng.randint(-200, 200) for _ in range(rate * pause_ms // 1000)]
        burst = rate * rng.randint(50, 500) // 1000
        pcm += [rng.choice([-1, 1]) * rng.randint(450, 9000) for _ in range(burst)]
    wire = struct.pack(f"<{len(pcm)}h", *pcm)
    tracker = OnsetTracker(rate, threshold=threshold, min_silence_ms=min_silence_ms)
    onsets_ms: list[float] = []
    for at in range(0, len(wire), 1554):  # an odd chunking: where the audio is cut must not matter
        onsets_ms += tracker.feed(wire[at : at + 1554])
    return {
        "pcm": base64.b64encode(wire).decode(),
        "rate": rate,
        "threshold": threshold,
        "min_silence_ms": min_silence_ms,
        "onsets": [round(ms * rate / 1000) for ms in onsets_ms],
    }


def _assert_same(fresh, committed, where="oracle.json"):
    """``fresh`` and ``committed`` hold the same JSON, floats within 1e-9 (relative)."""
    if isinstance(fresh, dict):
        assert isinstance(committed, dict) and fresh.keys() == committed.keys(), where
        for key in fresh:
            _assert_same(fresh[key], committed[key], f"{where}.{key}")
    elif isinstance(fresh, list):
        assert isinstance(committed, list) and len(fresh) == len(committed), where
        for i, (a, b) in enumerate(zip(fresh, committed)):
            _assert_same(a, b, f"{where}[{i}]")
    elif isinstance(fresh, float) and not isinstance(committed, bool):
        assert isinstance(committed, (int, float)), where
        assert math.isclose(fresh, committed, rel_tol=1e-9, abs_tol=1e-9), where
    else:
        assert fresh == committed, where


def test_the_oracle_fixture_is_what_its_generator_writes():
    # probe_check.mjs trusts tests/fixtures/sync/oracle.json as the oracle's numbers: it must be a
    # current run of make_oracle_fixture.py over hack/sync-sim/dsp.py.
    pytest.importorskip("numpy")
    if not (ORACLE_FIXTURE.parents[3] / "hack" / "sync-sim" / "dsp.py").exists():
        pytest.skip("hack/sync-sim is not in the sdist")
    spec = importlib.util.spec_from_file_location(
        "make_oracle_fixture", ORACLE_FIXTURE.with_name("make_oracle_fixture.py")
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    committed = json.loads(ORACLE_FIXTURE.read_text())
    fresh = generator.build()
    # A gate-boundary case's correlation curve is flat by construction, so the lag of its peak (and
    # everything measured there) is decided by rounding and differs between platforms' numpy;
    # probe_check.mjs compares those cases on the gate and the outcome only, and so does this.
    for document in (fresh, committed):
        for case in document["cases"]:
            if case.get("gate_only"):
                case["oracle_no_hp"] = {key: case["oracle_no_hp"][key] for key in GATE_ONLY_KEYS}

    _assert_same(fresh, committed)
    assert ORACLE_FIXTURE.stat().st_size < 200_000


def _stall_generator():
    pytest.importorskip("numpy")
    if not (ORACLE_FIXTURE.parents[3] / "hack" / "sync-sim" / "dsp.py").exists():
        pytest.skip("hack/sync-sim is not in the sdist")
    spec = importlib.util.spec_from_file_location(
        "make_stall_fixture", ORACLE_FIXTURE.with_name("make_stall_fixture.py")
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator


def test_the_stall_fixture_is_what_its_generator_writes():
    # probe_check.mjs holds the page's stall rule to tests/fixtures/sync/stall.json tick for tick:
    # it must be a current run of make_stall_fixture.py over hack/sync-sim (dsp.py's TurnTracker
    # and replay_capture.py's tick loop).
    generator = _stall_generator()
    committed = json.loads(STALL_FIXTURE.read_text())
    _assert_same(generator.build(), committed, "stall.json")
    assert STALL_FIXTURE.stat().st_size < 400_000


def test_the_stalled_capture_ticks_are_what_the_generator_writes():
    generator = _stall_generator()
    path = STALL_FIXTURE.with_name("daily-stall") / "oracle-ticks.json"
    if not path.exists():
        pytest.skip("tests/fixtures/sync/daily-stall is not in the sdist")
    _assert_same(generator.build_capture(), json.loads(path.read_text()), "oracle-ticks.json")


async def test_the_bot_serves_the_probe_at_probe():
    # The URL the README, examples/README.md and bot.py give for the probe.
    reason = "needs the example extra: uv sync --extra example"
    pytest.importorskip("fastapi", reason=reason)
    pytest.importorskip("dotenv", reason=reason)
    pytest.importorskip("pipecat.runner.run", reason=reason, exc_type=ImportError)
    import httpx
    from example_loader import load_bot
    from pipecat.runner.run import app

    load_bot()  # registers the pages on the runner's app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:7860") as client:
        reply = await client.get("/probe/", params={"mic": "1"})
        # The page imports the rule as a module next to it: ./sync.js.
        module = await client.get("/probe/sync.js")

    assert reply.status_code == 200
    assert reply.text == PAGE.read_text()
    assert "import { SyncTracker } from './sync.js'" in reply.text
    assert module.status_code == 200
    assert module.text == SYNC.read_text()
    assert "javascript" in module.headers["content-type"]


@pytest.mark.skipif(shutil.which("node") is None and not os.environ.get("CI"), reason="needs node")
def test_the_probe_reads_the_envelope_and_the_audio_as_the_processor_does(tmp_path):
    assert shutil.which("node"), "Node is not installed; the workflow must install it"
    audio = _audio()
    assert len(audio["onsets"]) >= 8  # the fixture has to exercise the rule
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"envelopes": _envelopes(), "audio": audio}))

    done = subprocess.run(
        ["node", str(CHECK), str(PAGE), str(fixture)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stdout + done.stderr
