"""The alignment rule replayed over the committed paired captures: the restated design 4.1 minimum.

``tests/fixtures/sync/`` holds paired captures (the bot's track as Chrome decoded it, the server's
envelope and every message):

* ``daily-clean``, ``livekit-clean``: one per transport, every turn after an idle (no splice);
* ``daily-splice``: 20 whole turns of a Daily run with chained turns (one TTS context per
  sentence) and follow-ups after 100-400 ms, so most turns are spliced onto the one before
  (design 3.3);
* ``daily-splice-failures``: one turn of the same run whose first 240 ms reached the browser
  65 ms ahead of the rest of it (below).

``hack/sync-sim/replay_capture.py`` replays the web SDK's rule over each the way the probe page
runs it (ticks from the turn's first ``turn_envelope``, the reference as it had arrived, the
page's prior) and measures every accepted tick against the **window content**: 400 ms windows of
the reference aligned on their own, so that the truth follows the received audio where Chrome's
jitter buffer moved it inside a turn (5-60 ms in the splice run) and the rule is not charged for
tracking it. The criteria are design 4.1's restated minimum (rulings R62/R63):

* zero false locks: no accepted tick more than 20 ms off the window-content truth;
* every turn locks within 750 ms of received speech;
* until its first lock a turn's frames follow the onset prior, which is within 10 ms of the truth
  on at least 95 % of the turns;
* p95 of the accepted ticks' error against the window-content truth is at most 5 ms.

"Every turn" means every turn the rule saw with at least 300 ms of received speech. Each capture
is replayed twice: with the oracle's zero-phase 30 Hz high-pass and with the browser's causal one.
With the causal one the replay must first-lock every turn where the page did live (within 1.5 ms),
which holds the replay's cadence and prior to the page's.
On the splice fixture, the joins themselves are checked too: the received distance between a
turn's last sample and the next one's first is the ``gap_ms`` the splice assumes, within a hop.

``daily-splice-failures`` fails two criteria, and those cases are ``xfail(strict=True)``: the
turn's lead-in and first syllable came out 65 ms early, the page's onset rule fired on them (so
the onset prior is 68 ms off the rest of the turn), and the rule, correctly, did not lock while its
1 s window straddled that step, so its first lock came after 1065 ms of speech. That is the
re-lock delay design 3.2 accepts after a jitter-buffer jump, here at the very start of a turn.
The thresholds are not changed for it.

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
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "hack" / "sync-sim" / "replay_capture.py"
_DIR = _ROOT / "tests" / "fixtures" / "sync"
# Turns with speech and accepted ticks each capture holds; fewer would mean the replay stopped
# seeing them.
_EXPECTED = {
    "daily-clean": (9, 100),
    "livekit-clean": (8, 100),
    "daily-splice": (20, 150),
    "daily-splice-failures": (1, 20),
}
_FIXTURES = [_DIR / name for name in _EXPECTED if (_DIR / name).is_dir()]
_SPLICE = _DIR / "daily-splice"

FALSE_LOCK_MS = 20.0
FIRST_LOCK_MAX_MS = 750
PRIOR_TOL_MS, PRIOR_SHARE = 10.0, 0.95
P95_MAX_MS = 5.0
HOP_MS = 5.0
MIN_CHAINED, MIN_GAP = 3, 3

_ONSET_STEP = (
    "daily-splice-failures: the turn's first 240 ms reached the browser 65 ms ahead of the rest; "
    "the rule holds while its 1 s window straddles the step (design 3.2's re-lock delay after a "
    "jitter-buffer jump) and the onset prior follows the early onset"
)


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
    return capture.name, results, speech, causal


_CASES = [(f, causal) for f in _FIXTURES for causal in (False, True)]


@pytest.fixture(
    scope="module",
    params=_CASES or [None],
    ids=lambda c: f"{c[0].name}-{'causal' if c[1] else 'zero-phase'}" if c else "none",
)
def capture(request, replay):
    if request.param is None:
        pytest.skip("tests/fixtures/sync/* is not in the sdist")
    return _run(replay, *request.param)


def _xfail_onset_step(request, name):
    if name == "daily-splice-failures":
        request.applymarker(pytest.mark.xfail(strict=True, reason=_ONSET_STEP))


def test_the_capture_still_holds_its_speech_turns(capture):
    name, results, speech, _ = capture
    turns, ticks = _EXPECTED[name]
    assert len(speech) >= turns
    assert sum(r["accepted"] for r in results) >= ticks


def test_every_accepted_tick_has_content_windows_for_its_truth(capture):
    # The own-window fallback is no independent truth; on the fixtures it is never needed.
    _, results, _, _ = capture
    fallback = {r["turn_id"]: r["window_fallback"] for r in results if r["window_fallback"]}
    assert not fallback


@pytest.mark.parametrize(
    "capture",
    [c for c in _CASES if c[1]] or [None],
    ids=lambda c: f"{c[0].name}-causal" if c else "none",
    indirect=True,
)
def test_causal_replay_first_locks_where_the_page_did(capture):
    # With the browser's high-pass the replay is the page's tracker: its first lock on every turn
    # is the one the page recorded live (it holds exactly today; 1.5 ms leaves room for float
    # rounding, not for a different tick). The page ran the causal high-pass only.
    _, results, _, causal = capture
    assert causal
    recorded = [r for r in results if r["page_first_lock_ms"] is not None]
    assert recorded
    off = {
        r["turn_id"]: (r["first_lock_ms"], r["page_first_lock_ms"])
        for r in recorded
        if r["first_lock_ms"] is None or abs(r["first_lock_ms"] - r["page_first_lock_ms"]) > 1.5
    }
    assert not off, f"replay vs page first lock, ms: {off}"


def test_zero_false_locks_against_the_window_content_truth(capture):
    _, results, _, _ = capture
    errors = [e for r in results for e in r["window_errors_ms"]]
    assert len(errors) == sum(r["accepted"] for r in results)
    late = [e for e in errors if e > FALSE_LOCK_MS]
    assert not late, f"{len(late)} accepted ticks more than {FALSE_LOCK_MS} ms off: {late}"


def test_every_turn_first_locks_within_750_ms_of_speech(capture, request):
    name, _, speech, _ = capture
    _xfail_onset_step(request, name)
    late = {
        r["turn_id"]: r["first_lock_ms"]
        for r in speech
        if r["first_lock_ms"] is None or r["first_lock_ms"] > FIRST_LOCK_MAX_MS
    }
    assert not late, f"first lock later than {FIRST_LOCK_MAX_MS} ms (None = never): {late}"


def test_until_the_first_lock_the_onset_prior_is_within_10_ms_of_the_truth(capture, request):
    name, _, speech, _ = capture
    _xfail_onset_step(request, name)
    prior = [
        abs(r["prior_error_window_ms"]) for r in speech if r["prior_error_window_ms"] is not None
    ]
    assert len(prior) == len(speech)
    ok = sum(e <= PRIOR_TOL_MS for e in prior)
    assert ok >= PRIOR_SHARE * len(prior), f"prior errors in ms: {sorted(prior)}"


def test_p95_error_against_the_window_content_truth_is_at_most_5_ms(capture):
    _, results, _, _ = capture
    errors = [e for r in results for e in r["window_errors_ms"]]
    assert np.percentile(errors, 95) <= P95_MAX_MS


@pytest.fixture(scope="module")
def splice(replay):
    if not _SPLICE.is_dir():
        pytest.skip("tests/fixtures/sync/daily-splice is not in the sdist")
    return _run(replay, _SPLICE, False)[1]


def _spliced(results, kind):
    return [r for r in results if r["kind"] and r["kind"][0] == kind]


def test_the_splice_fixture_holds_chained_and_gap_turns(splice):
    chained = _spliced(splice, "chained")
    gap = [r for r in _spliced(splice, "gap") if r["gap_ms"] > 0]
    assert len(chained) >= MIN_CHAINED
    assert len(gap) >= MIN_GAP


def test_every_join_holds_its_gap_within_one_hop(splice):
    joins = {
        r["turn_id"]: r["join_delta_ms"]
        for r in _spliced(splice, "chained") + _spliced(splice, "gap")
    }
    assert len(joins) >= MIN_CHAINED + MIN_GAP
    unmeasured = [t for t, d in joins.items() if d is None]
    assert not unmeasured, f"no content window on both sides of the join: {unmeasured}"
    off = {t: d for t, d in joins.items() if abs(d) > HOP_MS}
    assert not off, f"joins whose received gap is more than one hop off gap_ms (ms): {off}"


# R72: before its first lock a gap turn is placed at the previous turn's place + its
# turn_end.pcm_ms + gap_ms. The splice fixture was recorded before R72, so its ticks hold the arrival
# prior and the replay above uses that; the replay also computes the R72 prior (and with
# --r72-prior replays with it).
R72_PRIOR_TOL_MS = 2.0
MIN_R72_LOCKED_BEFORE_ONSET_PRIOR = 4


@pytest.fixture(scope="module")
def splice_r72(replay):
    if not _SPLICE.is_dir():
        pytest.skip("tests/fixtures/sync/daily-splice is not in the sdist")
    with contextlib.redirect_stdout(io.StringIO()):
        return replay.main([str(_SPLICE), "--causal", "--r72-prior"])


def test_r72_prior_puts_each_gap_turn_within_2_ms_of_its_first_content(splice):
    # The turns that locked before their onset prior existed: the recorded arrival prior had them
    # 78-182 ms off until the lock; the splice position is the join measured exact (|δ| <= 2.1 ms).
    r72 = {
        r["turn_id"]: r["r72_prior_error_ms"] for r in splice if r["r72_prior_error_ms"] is not None
    }
    assert len(r72) >= MIN_R72_LOCKED_BEFORE_ONSET_PRIOR, r72
    off = {t: e for t, e in r72.items() if abs(e) > R72_PRIOR_TOL_MS}
    assert not off, f"R72 prior errors in ms: {off}"
    recorded = [r["prelock_prior_error_ms"] for r in splice if r["turn_id"] in r72]
    assert max(abs(e) for e in recorded) > 50  # what R72 fixes: the arrival prior was far off


def test_r72_prior_leaves_every_first_lock_where_it_was(splice_r72, replay):
    # The prior only places frames until the lock; on the fixture it changes no lock.
    with contextlib.redirect_stdout(io.StringIO()):
        recorded = replay.main([str(_SPLICE), "--causal"])
    before = {r["turn_id"]: r["first_lock_ms"] for r in recorded}
    after = {r["turn_id"]: r["first_lock_ms"] for r in splice_r72}
    assert after == before
    errors = [e for r in splice_r72 for e in r["window_errors_ms"]]
    assert not [e for e in errors if e > FALSE_LOCK_MS]
    replayed = [
        r["prelock_prior_error_ms"] for r in splice_r72 if r["r72_prior_error_ms"] is not None
    ]
    assert replayed and all(abs(e) <= R72_PRIOR_TOL_MS for e in replayed)


def test_old_captures_without_prior_kind_infer_it_from_the_value(replay):
    default = 48000
    assert replay.recorded_kind({"prior_pos": 48000.2}, default) == "arrival"
    assert replay.recorded_kind({"prior_pos": 47000}, default) == "onset"
    assert replay.recorded_kind({"prior_pos": 47000, "prior_kind": "gap"}, default) == "gap"
    assert replay.recorded_kind({"prior_pos": 48000, "prior_kind": "onset"}, default) == "onset"


def test_a_capture_from_the_stall_rule_page_replays_the_onset_prior_it_recorded(replay):
    # Since the stall rule the page's prior can be a chained turn's place ('chain') while an onset
    # prior exists (R76 widens around the latter), so every tick records the onset prior itself.
    t = {
        "start_tap": 0,
        "onsets_ms": [],
        "onset_prior_pos": 30000,
        "ticks": [
            {
                "pos": 1000,
                "prior_pos": 9000,
                "prior_kind": "arrival",
                "onset_prior_pos": None,
                "accepted": False,
            },
            {
                "pos": 2000,
                "prior_pos": 50000,
                "prior_kind": "chain",
                "onset_prior_pos": 30000,
                "accepted": False,
            },
            {
                "pos": 3000,
                "prior_pos": 30000,
                "prior_kind": "onset",
                "onset_prior_pos": 30000,
                "accepted": True,
            },
        ],
    }
    at = replay.page_prior(t, 48000, 240)
    assert at(1500) is None
    assert at(2000) == 30000 and at(2999) == 30000  # not the chain place, not None
    old = {
        **t,
        "ticks": [{k: v for k, v in tick.items() if k != "onset_prior_pos"} for tick in t["ticks"]],
    }
    assert replay.page_prior(old, 48000, 240)(2000) is None  # the kind alone cannot say it


def test_a_capture_from_the_r72_page_replays_its_gap_priors_from_the_previous_turn(
    replay, splice_r72, tmp_path
):
    # The splice capture rewritten as the page records since R72: every tick has its prior_kind,
    # and the gap turns that R72 places carry 'gap' before their lock, with a recorded prior_pos
    # that is deliberately wrong. The replay must take them as gap priors (from the previous
    # turn's replayed place), not as onset priors frozen at the recorded value.
    if not _SPLICE.is_dir():
        pytest.skip("tests/fixtures/sync/daily-splice is not in the sdist")
    for name in ("capture.wav", "capture.jsonl"):
        (tmp_path / name).symlink_to(_SPLICE / name)
    sync = json.loads((_SPLICE / "capture.sync.json").read_text())
    fs, turns = sync["rate"], sync["turns"]
    rewritten, gap_at_lock = set(), set()  # turns with a 'gap' tick; those still gap at the lock
    for n, t in enumerate(turns):
        default = t["start_tap"] + round(200 * fs / 1000)
        gap = replay.gap_join(t, turns[n - 1] if n else None) is not None
        locked = False
        for k in t["ticks"]:
            if locked:
                k["prior_kind"] = "locked"
            else:
                kind = replay.recorded_kind(k, default)
                if gap and kind == "arrival":
                    kind, k["prior_pos"] = "gap", default + 12345.0
                    rewritten.add(t["turn_id"])
                k["prior_kind"] = kind
                if k["accepted"] and kind == "gap":
                    gap_at_lock.add(t["turn_id"])
            locked = locked or k["accepted"]
    assert len(gap_at_lock) >= MIN_R72_LOCKED_BEFORE_ONSET_PRIOR
    (tmp_path / "capture.sync.json").write_text(json.dumps(sync))
    with contextlib.redirect_stdout(io.StringIO()):
        results = replay.main([str(tmp_path), "--causal"])  # as recorded: no --r72-prior
    by_id = {r["turn_id"]: r for r in results}
    for tid in rewritten - gap_at_lock:  # an onset prior took over before the lock
        assert by_id[tid]["r72_prior_error_ms"] is None, tid
    for tid in gap_at_lock:
        r = by_id[tid]
        assert r["r72_prior_error_ms"] is not None, tid  # not taken for an onset prior
        assert r["prelock_prior_error_ms"] == r["r72_prior_error_ms"], tid  # replayed with it
        assert abs(r["r72_prior_error_ms"]) <= R72_PRIOR_TOL_MS, tid
    assert {r["turn_id"]: r["first_lock_ms"] for r in results} == {
        r["turn_id"]: r["first_lock_ms"] for r in splice_r72
    }
