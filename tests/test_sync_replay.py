"""The alignment rule replayed over the committed paired captures: the restated design 4.1 minimum.

``tests/fixtures/sync/`` holds paired captures (the bot's track as Chrome decoded it, the server's
envelope and every message):

* ``daily-clean``, ``livekit-clean``: one per transport, every turn after an idle (no splice);
* ``daily-splice``: 20 whole turns of a Daily run with chained turns (one TTS context per
  sentence) and follow-ups after 100-400 ms, so most turns are spliced onto the one before
  (design 3.3);
* ``daily-splice-failures``: one turn of the same run whose first 240 ms reached the browser
  65 ms ahead of the rest of it (below);
* ``daily-stall``: 8 whole turns of a Daily run around two host stalls (240 ms, 891 ms and 1.66 s
  late), replayed under the stall rule against today's rule (the last section).

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


# ---------------------------------------------------------------- a stalled capture, the stall rule
#
# ``daily-stall``: whole turns of a Daily run (2026-11-04, Chrome) around two host stalls, trimmed as
# the other captures (turns 9-12 and 42-45 of 54; tap positions rebased): at 15:52:14.6 the turn
# then playing (fixture turn 0) arrived 240 ms late from a point on, its chained follow-up (turn 1)
# 891 ms late, and at 15:54:38-46 turn 5 (chained) 1657 ms late after 1.66 s of concealment and
# zeros; Chrome then played the rest of that turn and the next (turn 6) faster in -20 to -45 ms steps.
# The replay runs with the browser's causal high-pass. The truth is the dense local truth of
# ``replay_capture.py --gaps`` (300 ms reference windows every 25 ms over +/- 2 s, so it follows a
# stall of seconds) and, for false locks, the window-content truth segmented at each re-align (the
# design 4.1 criterion). The stall rule must re-align after each real stall in the time the oracle
# measured (plus a margin), leave no turn worse than today's rule, and lock falsely nowhere; today's
# rule, on the same capture, holds through them for a second or never re-aligns.

_STALLED = _DIR / "daily-stall"
REAL_STALL_MS = 200.0  # a gap whose shift is at least this is one of the stalls above
# The oracle's re-align time after each real stall (from the resume until the face stays within
# 10 ms of the new truth for 100 ms), causal high-pass: fixture turn -> (shift ms, re-align ms).
STALL_REALIGN_MS = {0: (240, 280), 1: (891, 275), 5: (1657, 290)}
REALIGN_MARGIN_MS = 50.0
# Today's rule after the same stalls: turn 0 holds 1125 ms, turns 1 and 5 never re-align.
TODAY_HOLD_MIN_MS = 1000.0


@pytest.fixture(scope="module")
def stalled(replay):
    if not _STALLED.is_dir():
        pytest.skip("tests/fixtures/sync/daily-stall is not in the sdist")
    dsp = replay.dsp
    x, fs, sync, _ = replay.load(str(_STALLED))
    env, hop = replay.received_envelope(x, fs, True)
    rules = {"today": dsp.TODAY, "rule": dsp.STALL}
    report = replay.stall_capture_report(env, hop, fs, len(x), sync, rules)
    runs = {
        name: replay.replay_all_tracked(env, hop, fs, len(x), sync, dsp.DEFAULTS, st)
        for name, st in rules.items()
    }
    # per turn: ms after the first lock with the face more than 20 ms off the dense truth, + 2 s
    # when the turn reached sync-lost (thrsearch_stall.py's cost)
    cost = {name: {} for name in rules}
    for n, t in enumerate(sync["turns"]):
        own_db = dsp.dequantise(replay.b64(t["reference"]))
        base = runs["today"][t["turn_id"]]
        acc = [k["l"] for k in base["res"]["ticks"] if k["accepted"]]
        centre = acc[0] if acc else base["prior_l_at"](t["start_tap"])
        truth = replay.dense_truth_frames(replay.dense_track(env, own_db, centre), len(env))
        for name in rules:
            res = runs[name][t["turn_id"]]["res"]
            first = next((k["count"] for k in res["ticks"] if k["accepted"]), None)
            off = 0.0
            if first is not None:
                js = np.array(sorted(j for j in res["disp"] if j >= first))
                face = np.array([res["disp"][j] for j in js])
                ok = ~np.isnan(truth[js])
                off = float(
                    (np.abs(face[ok] - truth[js][ok]) * HOP_MS > FALSE_LOCK_MS).sum() * HOP_MS
                )
            cost[name][n] = off + 2000.0 * (len(res["lost_at"]) > 0)
    return report, cost


def _real_stalls(rows):
    return {
        (r["turn"], round(g["shift_ms"])): g
        for r in rows
        for g in r["gaps"]
        if g["shift_ms"] >= REAL_STALL_MS
    }


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "zero-phase"])
def test_the_stalled_capture_has_zero_false_locks_under_the_stall_rule(replay, causal):
    # The window-content truth, segmented at each re-align (the replay's default rule is the stall
    # rule): no accepted tick more than 20 ms off. Every turn has accepts to score but one (turn 6,
    # played faster after the 1.66 s stall, never locks under either rule).
    if not _STALLED.is_dir():
        pytest.skip("tests/fixtures/sync/daily-stall is not in the sdist")
    with contextlib.redirect_stdout(io.StringIO()):
        results = replay.main([str(_STALLED)] + (["--causal"] if causal else []))
    errors = [e for r in results for e in r["window_errors_ms"]]
    assert len(errors) == sum(r["accepted"] for r in results) >= 70
    assert not [e for e in errors if e > FALSE_LOCK_MS]
    assert sum(len(r["freezes"]) for r in results) >= 6
    assert sum(r["half_accepts"] for r in results) >= 4


def test_the_stalled_capture_has_no_false_accept_on_the_dense_truth(stalled):
    # Full-window, newer-half and confirm accepts, outside a gap and across one, for both rules.
    report, _ = stalled
    for name, rows in report.items():
        assert sum(r["scored"] for r in rows) >= 60, name
        assert not [f for r in rows for f in r["false"]], name
        assert not [f for r in rows for f in r["false_straddle"]], name


def test_the_stall_rule_realigns_after_each_real_stall_as_fast_as_the_oracle_measured(stalled):
    report, _ = stalled
    stalls = _real_stalls(report["rule"])
    assert sorted(stalls) == sorted((turn, shift) for turn, (shift, _) in STALL_REALIGN_MS.items())
    for (turn, shift), g in stalls.items():
        measured = STALL_REALIGN_MS[turn][1]
        assert g["realign_ms"] is not None, (turn, shift)
        assert g["realign_ms"] <= measured + REALIGN_MARGIN_MS, (turn, shift, g["realign_ms"])
    # the face runs ahead of the sound during the 1.66 s gap only for detection and concealment
    assert stalls[(5, 1657)]["max_ahead_ms"] <= 155


def test_today_holds_through_the_same_stalls(stalled):
    # What the rule is for: today's rule coasts at the old offset until its 1 s window clears the
    # jump, or never re-aligns (beyond its +/- 300 ms search), with the face running ahead of the
    # sound for the whole 1.66 s gap.
    report, _ = stalled
    stalls = _real_stalls(report["today"])
    assert sorted(stalls) == sorted((turn, shift) for turn, (shift, _) in STALL_REALIGN_MS.items())
    for key, g in stalls.items():
        assert g["realign_ms"] is None or g["realign_ms"] >= TODAY_HOLD_MIN_MS, (
            key,
            g["realign_ms"],
        )
    assert sum(g["realign_ms"] is None for g in stalls.values()) >= 2
    assert stalls[(5, 1657)]["max_ahead_ms"] >= 1500


def test_no_turn_of_the_stalled_capture_is_worse_under_the_stall_rule(stalled):
    report, cost = stalled
    worse = {n: (cost["today"][n], c) for n, c in cost["rule"].items() if c > cost["today"][n]}
    assert not worse, f"turns costlier under the stall rule (today, rule) in ms: {worse}"
    assert sum(cost["rule"].values()) < sum(cost["today"].values()) - 1000
    # and no gap re-aligns later than today's rule re-aligned it
    for today_row, rule_row in zip(report["today"], report["rule"]):
        for gt, gr in zip(today_row["gaps"], rule_row["gaps"]):
            if gt["realign_ms"] is not None:
                assert gr["realign_ms"] is not None and gr["realign_ms"] <= gt["realign_ms"], (
                    rule_row["turn"],
                    gt["shift_ms"],
                )
