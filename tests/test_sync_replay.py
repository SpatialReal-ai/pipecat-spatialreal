"""The alignment rule replayed over the committed paired captures: the restated design 4.1 minimum.

``tests/fixtures/sync/*-clean`` hold paired captures (the bot's track as Chrome decoded it, the
server's envelope and every message), one per transport. ``hack/sync-sim/replay_capture.py``
replays the web SDK's rule over each and measures it against each phrase's own alignment, which
is the truth the criteria use: the jitter buffer moves the audio by a few ms inside a call and
the rule tracks it. The criteria are the ones design 4.1 restates after the first measurement
(rulings R62/R63):

* zero false locks: no accepted tick more than 20 ms off the phrase truth;
* every turn locks within 750 ms of received speech;
* until its first lock a turn's frames follow the onset prior, which is within 10 ms of the truth
  on at least 95 % of the turns;
* p95 of the accepted ticks' error against the phrase truth is at most 5 ms.

"Every turn" means every turn the rule saw with at least 300 ms of received speech. Each capture
is replayed twice: with the oracle's zero-phase 30 Hz high-pass and with the browser's causal one.

The truth shares the reference with the rule: both correlate the same reassembled reference
envelope, so a wrong index origin in that reference would move both alike and not show here. The
origin is pinned at the processor level, in ``tests/test_processor.py``:
``test_envelope_values_equal_fingerprint_on_the_pushed_bytes`` and
``test_one_chunk_turn_has_seven_values``.

The fixtures are not in the sdist (see ``pyproject.toml``); without them the tests skip.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "hack" / "sync-sim" / "replay_capture.py"
_FIXTURES = sorted((_ROOT / "tests" / "fixtures" / "sync").glob("*-clean"))

FALSE_LOCK_MS = 20.0
FIRST_LOCK_MAX_MS = 750
PRIOR_TOL_MS, PRIOR_SHARE = 10.0, 0.95
P95_MAX_MS = 5.0
# The captures hold 9 and 8 turns with speech; fewer would mean the replay stopped seeing them.
MIN_SPEECH_TURNS = 8


def _load_replay():
    """Import the replay script by path; its own ``import dsp`` needs its directory on sys.path
    for the moment of the import only."""
    spec = importlib.util.spec_from_file_location("sync_sim_replay_capture", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SCRIPT.parent))
    return module


@pytest.fixture(scope="module")
def replay():
    if not _SCRIPT.exists():
        pytest.skip("hack/sync-sim is not in the sdist")
    return _load_replay()


def _run(replay, capture: Path, causal: bool):
    """The replay's per-turn results; its tables go to a buffer."""
    with contextlib.redirect_stdout(io.StringIO()):
        results = replay.main([str(capture)] + (["--causal"] if causal else []))
    seen = [r for r in results if r["accepted"] or r["held"]]
    speech = [r for r in seen if (r["rx_ms"] or 0) >= 300]
    return results, speech


_CASES = [(f, causal) for f in _FIXTURES for causal in (False, True)]


@pytest.fixture(
    scope="module",
    params=_CASES or [None],
    ids=lambda c: f"{c[0].name}-{'causal' if c[1] else 'zero-phase'}" if c else "none",
)
def capture(request, replay):
    if request.param is None:
        pytest.skip("tests/fixtures/sync/*-clean is not in the sdist")
    return _run(replay, *request.param)


def test_the_capture_still_holds_its_speech_turns(capture):
    _, speech = capture
    assert len(speech) >= MIN_SPEECH_TURNS


def test_zero_false_locks_against_the_phrase_truth(capture):
    results, _ = capture
    errors = [e for r in results for e in r["local_errors_ms"]]
    assert len(errors) > 100  # the rule did accept ticks
    late = [e for e in errors if e > FALSE_LOCK_MS]
    assert not late, f"{len(late)} accepted ticks more than {FALSE_LOCK_MS} ms off: {late}"


def test_every_turn_first_locks_within_750_ms_of_speech(capture):
    _, speech = capture
    late = {
        r["turn_id"]: r["first_lock_ms"]
        for r in speech
        if r["first_lock_ms"] is None or r["first_lock_ms"] > FIRST_LOCK_MAX_MS
    }
    assert not late, f"first lock later than {FIRST_LOCK_MAX_MS} ms (None = never): {late}"


def test_until_the_first_lock_the_onset_prior_is_within_10_ms_of_the_truth(capture):
    _, speech = capture
    prior = [abs(r["prior_error_ms"]) for r in speech if r["prior_error_ms"] is not None]
    assert len(prior) == len(speech)
    ok = sum(e <= PRIOR_TOL_MS for e in prior)
    assert ok >= PRIOR_SHARE * len(prior), f"prior errors in ms: {sorted(prior)}"


def test_p95_error_against_the_phrase_truth_is_at_most_5_ms(capture):
    results, _ = capture
    errors = [e for r in results for e in r["local_errors_ms"]]
    assert np.percentile(errors, 95) <= P95_MAX_MS
