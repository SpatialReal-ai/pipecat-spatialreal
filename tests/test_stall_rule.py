"""The stall rule in the normative oracle (``hack/sync-sim/dsp.py``: ``TurnTracker``,
``confirm_position``, ``newer_half_candidate``, ``chained_prior``) and its tick loop
(``hack/sync-sim/replay_capture.py``: ``replay_turn_tracked``).

When the bot's host or the receiver stalls, the browser plays a concealment fade and then exact
zeros, and the rest of the turn arrives late by about the stall. The rule (design 3.2, owner
2026-11-04): after the first lock, freeze the reference pointer while the received audio is silent
where the reference expects speech (20 ms to trigger, retroactive, at most 2 s); confirm the new
position with a small correlation once audio resumes; accept the newer half-window alone on two
agreeing ticks unless it holds digital silence; keep frozen time off the 2 s loss clock; and place
a chained turn from its locked predecessor (rule 5, R74).

The scenarios are synthetic (speech-like noise at 24 kHz, the received side computed from it) so
each piece can be switched off and seen to matter; the committed captures check that the rule
leaves clean data alone.
"""

import contextlib
import importlib.util
import io
import sys
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_ROOT = Path(__file__).resolve().parent.parent
_SIM = _ROOT / "hack" / "sync-sim"
if not (_SIM / "dsp.py").exists():
    pytest.skip("hack/sync-sim is not in the sdist", allow_module_level=True)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_SIM))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_SIM))
    return module


rc = _load("stall_rule_replay_capture", _SIM / "replay_capture.py")
dsp = rc.dsp  # the module the replay uses, so isinstance and defaults agree

RATE = 24000
HOP = RATE * 5 // 1000  # samples per envelope frame
LEAD_S = 0.5
_DIR = _ROOT / "tests" / "fixtures" / "sync"
_FIXTURES = [
    _DIR / n
    for n in ("daily-clean", "livekit-clean", "daily-splice", "daily-splice-failures")
    if (_DIR / n).is_dir()
]


def _speechlike(seconds, seed=11, pauses=()):
    """Noise in syllable-length bursts with short pauses (exact zeros between syllables), as
    float int16-scale samples; ``pauses`` = ``[(at_s, length_s)]`` of extra digital silence."""
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    env = np.zeros(n)
    pos = 0
    while pos < n:
        syllable = round(rng.uniform(0.08, 0.25) * RATE)
        pause = round(rng.uniform(0.02, 0.06) * RATE)
        t = np.linspace(0, np.pi, syllable)
        env[pos : pos + syllable] = rng.uniform(0.1, 0.6) * np.sin(t)[: n - pos]
        pos += syllable + pause
    pcm = np.round(rng.standard_normal(n) * env * 32767 * 0.5)
    for at, length in pauses:
        pcm[round(at * RATE) : round((at + length) * RATE)] = 0.0
    return pcm


def _envelope(x):
    """The received envelope as the browser has it (causal 30 Hz high-pass, 5 ms frames)."""
    return dsp._window_db(rc.causal_high_pass(x, RATE), HOP, 2 * HOP)


def _turn(reference_pcm, received_pcm, ticks=(), onset_prior_pos=None, turn_id="t"):
    """A fake turn whose whole envelope arrived with turn_start; the arrival prior is the true
    position (``LEAD_S`` into the track)."""
    ref = dsp.envelope_bytes(reference_pcm, RATE)
    start = round(LEAD_S * RATE) - round(0.2 * RATE)
    t = {
        "turn_id": turn_id,
        "start_tap": start,
        "end_tap": start + RATE // 10,
        "end_pcm_ms": len(reference_pcm) / RATE * 1000,
        "abort_tap": None,
        "ticks": list(ticks),
        "onsets_ms": [],
        "onset_prior_pos": onset_prior_pos,
    }
    return t, {turn_id: [(start - 1, 0, ref)]}, _envelope(received_pcm), len(received_pcm)


def _replay(t, arrivals, env, n, stall, chain_prior=None):
    return rc.replay_turn_tracked(
        t, env, HOP, RATE, n, dsp.DEFAULTS, {}, arrivals, None, stall, chain_prior
    )


def _with_stall(pcm, at_s, stall_s):
    """``pcm`` received after a lead, with ``stall_s`` of exact zeros inserted at ``at_s`` (the
    rest arrives that much later), and a quiet tail."""
    a = round(at_s * RATE)
    lead, tail = np.zeros(round(LEAD_S * RATE)), np.zeros(RATE)
    return np.concatenate([lead, pcm[:a], np.zeros(round(stall_s * RATE)), pcm[a:], tail])


L0 = LEAD_S * 1000 / 5  # the true position (frames) before any impairment


# ---------------------------------------------------------------- freeze


def _locked_tracker(stall=None):
    """A tracker locked at L0 on a clean 3 s turn, and the turn's envelopes."""
    pcm = _speechlike(3.0, seed=5)
    ref = dsp.envelope_bytes(pcm, RATE)
    env = _envelope(np.concatenate([np.zeros(round(LEAD_S * RATE)), pcm, np.zeros(RATE)]))
    tr = dsp.TurnTracker("t", stall=stall or dsp.STALL)
    tr.set_reference(ref)
    count = round((LEAD_S + 1.2) * 1000 / 5)
    rec = tr.tick(env, count, ref, 0, L0, 300.0, 0.0)
    assert rec["accepted"] and abs(tr.l - L0) < 0.5
    return tr, env, ref, count


def test_freeze_triggers_after_the_trigger_run_of_silence_where_the_reference_has_speech():
    tr, env, ref, count = _locked_tracker()
    own = dsp.dequantise(ref)
    # the first frame after the lock whose reference (at the pointer) is speech, with 50 frames
    # of speech after it: silence the received from there
    j0 = next(
        j
        for j in range(count, count + 300)
        if (own[j - round(L0) : j - round(L0) + 12] > dsp.STALL.ref_speech_db).all()
    )
    env = env.copy()
    env[j0:] = -100.0
    for j in range(count, j0 + tr.n_min - 1):
        tr.frame(env, j)
    assert not tr.frozen and tr.l == pytest.approx(L0, abs=0.5)  # one frame short: nothing yet
    tr.frame(env, j0 + tr.n_min - 1)
    assert tr.frozen
    assert tr.l == pytest.approx(L0 + tr.n_min, abs=0.5)  # retroactive over the run
    assert tr.events[-1][1] == {
        "kind": "frozen",
        "turnId": "t",
        "atMs": (j0 - round(L0)) * 5.0,
    }
    for j in range(j0 + tr.n_min, j0 + tr.n_min + 20):
        tr.frame(env, j)
    assert tr.l == pytest.approx(L0 + tr.n_min + 20, abs=0.5)  # the pointer stays where it was


def test_silence_where_the_reference_is_silent_never_freezes():
    tr, env, ref, count = _locked_tracker()
    own = dsp.dequantise(ref)
    env = env.copy()
    # silence the received only on frames whose reference (at the pointer) is itself quiet
    quiet = [j for j in range(count, count + 300) if own[j - round(L0)] <= dsp.STALL.ref_speech_db]
    assert len(quiet) > 20
    env[quiet] = -100.0
    for j in range(count, count + 300):
        tr.frame(env, j)
    assert not tr.freezes and tr._episode is None and tr.l == pytest.approx(L0, abs=0.5)


def test_the_reference_must_have_speech_on_every_frame_of_the_trigger_run():
    # Received silence that starts on the last frame of a syllable and runs into the reference's
    # own pause: only its first frame is over speech, so it is the pause, not a stall.
    tr, env, ref, count = _locked_tracker()
    own = dsp.dequantise(ref)
    thr, n = dsp.STALL.ref_speech_db, tr.n_min
    k0 = next(
        k
        for k in range(count - round(L0), count - round(L0) + 300)
        if own[k] > thr and (own[k + 1 : k + 1 + n + 4] <= thr).all()
    )
    j0 = k0 + round(L0)
    env = env.copy()
    env[j0 : j0 + n + 4] = -100.0
    for j in range(count, j0 + n + 8):
        tr.frame(env, j)
    assert not tr.freezes and tr._episode is None


def test_a_natural_pause_and_a_short_dropout_never_freeze_the_turn():
    # A 600 ms digital pause inside the reference (received the same), and 20 ms of the received
    # zeroed over speech without a shift (as long as the trigger run: the window and the
    # high-pass tail leave fewer silent frames than that).
    pcm = _speechlike(4.0, seed=3, pauses=[(1.6, 0.6)])
    rx = np.concatenate([np.zeros(round(LEAD_S * RATE)), pcm, np.zeros(RATE)])
    loud = np.flatnonzero(np.abs(pcm[round(2.9 * RATE) : round(3.3 * RATE)]) > 3000)
    a = round(LEAD_S * RATE) + round(2.9 * RATE) + loud[0]
    rx[a : a + round(dsp.STALL.freeze_after_ms / 1000 * RATE)] = 0.0
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    assert any(k["accepted"] for k in res["ticks"])
    assert res["freezes"] == [] and res["events"] == [] and res["lost_at"] == []


def test_a_stall_freezes_and_is_confirmed_at_the_new_position():
    pcm = _speechlike(4.0, seed=8)
    loud = np.flatnonzero(np.abs(pcm[round(1.8 * RATE) :]) > 3000)
    at = 1.8 + loud[0] / RATE + 0.05  # inside a syllable
    rx = _with_stall(pcm, at, 0.3)
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    assert len(res["freezes"]) == 1
    f = res["freezes"][0]
    assert f["how"] == "confirm"
    assert f["confirm_l"] == pytest.approx(L0 + 60, abs=1.0)  # 300 ms later
    assert 50 <= f["frames"] <= 60  # the zeros, less the frames whose window still has audio
    kinds = [e["kind"] for _, e in res["events"]]
    assert kinds == ["frozen", "unfrozen"]
    assert res["events"][1][1]["confirmed"] is True
    assert res["events"][1][1]["frozenMs"] == f["frames"] * 5.0
    # today's rule coasts at the old position through the gap
    today = _replay(*_turn(pcm, rx), dsp.TODAY)
    assert today["freezes"] == []
    j = f["resume"] + 10  # after the gap, before the confirm: the frozen pointer
    assert res["disp"][j] == pytest.approx(L0 + 60, abs=4.0)
    assert today["disp"][j] == pytest.approx(L0, abs=1.0)


def _syllable_at(pcm, from_s):
    """A position (seconds) 50 ms into the first loud stretch at or after ``from_s``."""
    loud = np.flatnonzero(np.abs(pcm[round(from_s * RATE) :]) > 3000)
    return from_s + loud[0] / RATE + 0.05


def test_a_burst_loss_is_undone_by_the_confirm():
    # 300 ms of content replaced by zeros, timeline unchanged: the freeze is wrong, and the
    # confirm (whose range starts 150 ms below the pointer before the freeze) puts it back. The
    # synthetic pauses are digital silence, so pauses heard under the shifted pointer may freeze
    # again before the confirm; those freezes keep the first bracket.
    pcm = _speechlike(4.0, seed=8)
    at = _syllable_at(pcm, 1.8)
    rx = np.concatenate([np.zeros(round(LEAD_S * RATE)), pcm, np.zeros(RATE)])
    a = round((LEAD_S + at) * RATE)
    rx[a : a + round(0.3 * RATE)] = 0.0
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    fz = res["freezes"]
    assert fz[0]["frames"] >= 50  # the wrong freeze
    assert all(f["l_pre"] == fz[0]["l_pre"] for f in fz)
    confirmed = [f for f in fz if f["how"] == "confirm"]
    assert confirmed and confirmed[0]["confirm_l"] == pytest.approx(L0, abs=1.0)
    j = max(res["disp"])
    assert res["disp"][j] == pytest.approx(L0, abs=1.0)


def test_a_burst_loss_is_confirmed_early_by_the_no_shift_check():
    # The same burst loss: "nothing moved" is tested alone from 200 ms after the resume, before the
    # confirm proper (300 ms).
    pcm = _speechlike(4.0, seed=8)
    at = _syllable_at(pcm, 1.8)
    rx = np.concatenate([np.zeros(round(LEAD_S * RATE)), pcm, np.zeros(RATE)])
    a = round((LEAD_S + at) * RATE)
    rx[a : a + round(0.3 * RATE)] = 0.0
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    first = res["freezes"][0]
    assert first["how"] == "confirm" and first["confirm_l"] == pytest.approx(L0, abs=1.0)
    assert first["confirm_after_ms"] < dsp.STALL.confirm_from_ms
    late = _replay(*_turn(pcm, rx), replace(dsp.STALL, noshift_tol_ms=0.0))
    assert all(f.get("confirm_after_ms", 1e9) >= dsp.STALL.confirm_from_ms for f in late["freezes"])


def test_the_no_shift_check_accepts_only_near_the_unfrozen_position():
    env, own, l_true = _confirm_case()
    j_res = l_true + 100
    args = (env, own, j_res, j_res + 40, l_true - 30, l_true + 40, dsp.STALL)
    assert dsp.confirm_position(*args, noshift_l=l_true, noshift_only=True) == pytest.approx(
        l_true, abs=0.5
    )
    # the peak is real but 50 ms from the hypothesis: the no-shift test does not take it
    assert dsp.confirm_position(*args, noshift_l=l_true + 10, noshift_only=True) is None


def test_an_unconfirmed_freeze_is_reverted_when_asked():
    # Silence over speech freezes the pointer; the audio that follows matches nothing (noise), so
    # the confirm never accepts: when its time runs out the pointer goes back where it would be
    # without the freeze.
    revert = replace(dsp.STALL, revert_unconfirmed=True)  # evaluated, off by default
    tr, env, ref, count = _locked_tracker(revert)
    own = dsp.dequantise(ref)
    j0 = next(
        j
        for j in range(count, count + 300)
        if (own[j - round(L0) : j - round(L0) + 12] > dsp.STALL.ref_speech_db).all()
    )
    env = env.copy()
    rng = np.random.default_rng(3)
    env[j0 : j0 + 40] = -100.0
    end = j0 + 40 + tr.n_until + 5
    env[j0 + 40 : end] = rng.uniform(-50, -20, end - j0 - 40)
    for j in range(count, end):
        tr.frame(env, j)
    assert len(tr.freezes) == 1 and tr.freezes[0]["how"] == "expired"
    assert tr.freezes[0].get("reverted")
    assert tr.l == pytest.approx(L0, abs=0.5)
    assert tr.events[-1][1]["kind"] == "unfrozen" and tr.events[-1][1]["confirmed"] is False
    kept, _, _, count2 = _locked_tracker(dsp.STALL)
    for j in range(count2, end):
        kept.frame(env, j)
    assert kept.l == pytest.approx(L0 + kept.freezes[0]["frames"], abs=0.5)


def test_a_second_stall_before_the_confirm_keeps_the_first_bracket():
    # 200 ms stall, 150 ms of audio, another 200 ms stall: the second freeze starts while the
    # first one's confirm is pending, keeps its bracket, and the confirm finds both stalls.
    pcm = _speechlike(5.0, seed=8)
    at = _syllable_at(pcm, 1.8)
    a, b = round(at * RATE), round((at + 0.15) * RATE)
    z = np.zeros(round(0.2 * RATE))
    lead, tail = np.zeros(round(LEAD_S * RATE)), np.zeros(RATE)
    rx = np.concatenate([lead, pcm[:a], z, pcm[a:b], z, pcm[b:], tail])
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    assert [f["how"] for f in res["freezes"]][:2] == ["superseded", "confirm"]
    assert res["freezes"][1]["carried"]
    assert res["freezes"][1]["l_pre"] == res["freezes"][0]["l_pre"]
    assert res["freezes"][1]["confirm_l"] == pytest.approx(L0 + 80, abs=1.0)
    # without the carried bracket the second confirm's range starts at the frozen pointer
    dropped = _replay(*_turn(pcm, rx), replace(dsp.STALL, carry_bracket=False))
    assert not any(f.get("carried") for f in dropped["freezes"])


def test_a_quiet_dropout_during_a_pending_confirm_does_not_freeze_again():
    # After a 200 ms stall, while its confirm is pending, 100 ms of the received fall to about
    # -80 dBFS (quiet, not digital silence) over speech, without moving the timeline.
    pcm = _speechlike(5.0, seed=8)
    at = _syllable_at(pcm, 1.8)
    a = round(at * RATE)
    rx = np.concatenate(
        [
            np.zeros(round(LEAD_S * RATE)),
            pcm[:a],
            np.zeros(round(0.2 * RATE)),
            pcm[a:],
            np.zeros(RATE),
        ]
    )
    d = round((LEAD_S + _syllable_at(pcm, at + 0.1) + 0.2) * RATE)
    rng = np.random.default_rng(1)
    rx[d : d + round(0.1 * RATE)] = np.round(rng.standard_normal(round(0.1 * RATE)) * 3)
    res = _replay(*_turn(pcm, rx), dsp.STALL)
    # one freeze, not superseded (its confirm window holds the dropout, so it may expire; the
    # normal rule then carries on)
    assert len(res["freezes"]) == 1 and res["freezes"][0]["how"] != "superseded"
    anything = _replay(*_turn(pcm, rx), replace(dsp.STALL, confirming_silent_db=0.0))
    assert len(anything["freezes"]) >= 2  # any silence would freeze again


# ---------------------------------------------------------------- confirm


def _confirm_case(seed=21):
    pcm = _speechlike(2.0, seed=seed)
    own = dsp.dequantise(dsp.envelope_bytes(pcm, RATE))
    l_true = 300
    env = _envelope(np.concatenate([np.zeros(l_true * HOP), pcm, np.zeros(RATE)]))
    return env, own, l_true


def test_confirm_accepts_the_position_inside_its_range():
    env, own, l_true = _confirm_case()
    j_res = l_true + 100
    est = dsp.confirm_position(env, own, j_res, j_res + 60, l_true - 30, l_true + 40)
    assert est == pytest.approx(l_true, abs=0.5)


def test_confirm_rejects_a_position_outside_its_range():
    env, own, l_true = _confirm_case()
    j_res = l_true + 100
    # the true position lies 5 frames below the range: the best inside is its edge
    assert dsp.confirm_position(env, own, j_res, j_res + 60, l_true + 5, l_true + 60) is None
    assert dsp.confirm_position(env, own, j_res, j_res + 60, l_true - 60, l_true - 5) is None
    # the edge check alone, one frame outside (peak and sharpness relaxed: they reject a slope
    # at the edge too)
    edge_only = replace(dsp.STALL, confirm_min_sharpness=-1.0, confirm_min_peak=0.5)
    for lo, hi in ((l_true + 1, l_true + 60), (l_true - 60, l_true - 1)):
        assert dsp.confirm_position(env, own, j_res, j_res + 60, lo, hi, edge_only) is None
    inside = dsp.confirm_position(env, own, j_res, j_res + 60, l_true - 30, l_true + 40, edge_only)
    assert inside == pytest.approx(l_true, abs=0.5)


def test_confirm_rejects_an_ambiguous_peak():
    # A reference that repeats every 100 ms: every period matches equally well (ratio ~1).
    rng = np.random.default_rng(4)
    unit = np.round(rng.standard_normal(round(0.1 * RATE)) * np.hanning(round(0.1 * RATE)) * 8000)
    pcm = np.tile(unit, 20)
    own = dsp.dequantise(dsp.envelope_bytes(pcm, RATE))
    l_true = 300
    env = _envelope(np.concatenate([np.zeros(l_true * HOP), pcm, np.zeros(RATE)]))
    j_res = l_true + 100
    args = (env, own, j_res, j_res + 60, l_true - 50, l_true + 50)
    assert dsp.confirm_position(*args) is None
    assert dsp.confirm_position(*args, stall=replace(dsp.STALL, confirm_min_ratio=0.0)) is not None


# ---------------------------------------------------------------- newer half


def _deletion_case(cut_s=0.15):
    """A turn whose received audio loses ``cut_s`` of content mid-turn (a jitter-buffer
    deletion, no silence): the full window straddles it, the newer half does not."""
    pcm = _speechlike(5.0, seed=13)
    a = round(2.2 * RATE)
    rx_body = np.concatenate([pcm[:a], pcm[a + round(cut_s * RATE) :]])
    rx = np.concatenate([np.zeros(round(LEAD_S * RATE)), rx_body, np.zeros(RATE)])
    return _turn(pcm, rx)


def test_newer_half_relocks_only_on_the_second_agreeing_tick():
    res = _replay(*_deletion_case(), dsp.STALL)
    ticks = res["ticks"]
    first = next(
        i
        for i, k in enumerate(ticks)
        if k["half_l"] is not None and i > 0 and ticks[i - 1]["accepted"] is False
    )
    assert not ticks[first]["accepted"]  # one candidate is not enough
    halves = [i for i, k in enumerate(ticks) if k["via"] == "half"]
    assert halves
    i = halves[0]
    assert ticks[i - 1]["half_l"] is not None
    assert abs(ticks[i]["half_l"] - ticks[i - 1]["half_l"]) * 5 <= 10
    assert ticks[i]["l"] == pytest.approx(L0 - 30, abs=1.0)  # 150 ms earlier
    today = _replay(*_deletion_case(), dsp.TODAY)
    assert not any(k.get("via") == "half" for k in today["ticks"])


def test_newer_half_refuses_a_half_with_digital_silence():
    # 100 ms of quiet noise (not silence) inserted at 2.1 s moves the rest 100 ms later; at two
    # ticks 200 ms apart the full window straddles the move, the newer halves agree on the moved
    # position, and both hold a 100 ms digital pause (where the reference pauses too).
    pcm = _speechlike(4.0, seed=17, pauses=[(2.35, 0.1)])
    ref = dsp.envelope_bytes(pcm, RATE)
    rng = np.random.default_rng(2)
    a, shift = round(2.1 * RATE), 20  # frames: 100 ms
    quiet = np.round(rng.standard_normal(shift * HOP) * 30)  # about -60 dBFS
    lead = np.zeros(round(LEAD_S * RATE))
    env = _envelope(np.concatenate([lead, pcm[:a], quiet, pcm[a:], np.zeros(RATE)]))
    first = round(L0 + a / HOP + shift + 100)  # the newer half starts right after the insertion
    counts = [first, first + 40]

    def run(stall):
        tr = dsp.TurnTracker("t", stall=stall)
        tr.set_reference(ref)
        tr.locked, tr.l = True, L0  # locked at the old position
        return [tr.tick(env, c, ref, 0, L0, 300.0, 200.0) for c in counts]

    refused = run(dsp.STALL)
    assert all(not r["accepted"] for r in refused)
    assert all(r["half_refused"] for r in refused)
    allowed = run(replace(dsp.STALL, half_silence_ms=1e9))
    assert not allowed[0]["accepted"] and allowed[0]["half_l"] is not None
    assert allowed[1]["accepted"] and allowed[1]["via"] == "half"
    assert allowed[1]["l"] == pytest.approx(L0 + shift, abs=1.0)


# ---------------------------------------------------------------- loss clock


def test_frozen_time_does_not_count_toward_the_loss_clock():
    def run(stall):
        tr, env, ref, count = _locked_tracker(stall)
        own = dsp.dequantise(ref)
        j0 = next(
            j
            for j in range(count, count + 300)
            if (own[j - round(L0) : j - round(L0) + 12] > dsp.STALL.ref_speech_db).all()
        )
        env = env.copy()
        env[j0:] = -100.0  # the track stalls for good
        clock = []
        last = count
        for j in range(count, j0 + 300):
            tr.frame(env, j)
            if (j + 1 - count) % 40 == 0:
                rec = tr.tick(env, j + 1, ref, 0, L0, 300.0, (j + 1 - last) * 5.0)
                last = j + 1
                clock.append(rec["unlocked_ms"])
        return clock, tr

    held, tr = run(dsp.STALL)
    assert tr.frozen or tr.freezes
    # only the time before the freeze counts (less than a tick)
    assert max(held) <= 200.0
    counted, _ = run(replace(dsp.STALL, hold_clock=False))
    assert max(counted) >= 1200.0


# ---------------------------------------------------------------- rule 5: the chained prior


def test_chained_prior_takes_an_onset_only_within_200_ms():
    assert dsp.chained_prior(1000.0, 1030.0) == (1030.0, "onset")  # 150 ms
    assert dsp.chained_prior(1000.0, 960.0) == (960.0, "onset")  # -200 ms: inside
    assert dsp.chained_prior(1000.0, 1041.0) == (1000.0, "chain")  # 205 ms
    assert dsp.chained_prior(1000.0, 1320.0) == (1000.0, "chain")  # 1.6 s (run2's turn 44)
    assert dsp.chained_prior(1000.0, None) == (1000.0, "chain")


def test_chain_prior_follows_the_locked_predecessor_plus_its_pcm_ms():
    placed = {"prev": {"history": [(4800, 500.0), (9600, 520.0)]}}
    t = {"prev_turn_id": "prev", "prev_turn_pcm_ms": 2000}
    at = rc.chain_prior_for(t, placed, RATE, HOP)
    assert at(4799) is None  # the predecessor has not locked yet
    assert at(5000) == pytest.approx(500.0 + 400.0)  # 2000 ms = 400 frames
    assert at(20000) == pytest.approx(520.0 + 400.0)  # moved with its freeze or re-lock
    assert (
        rc.chain_prior_for({"prev_turn_id": "", "prev_turn_pcm_ms": 0}, placed, RATE, HOP) is None
    )


@pytest.mark.parametrize(("onset_off", "expect"), [(30, "onset"), (60, "chain")])
def test_the_tick_loop_uses_the_chain_unless_the_onset_is_near_it(onset_off, expect):
    pcm = _speechlike(3.0, seed=9)
    rx = np.concatenate([np.zeros(round(LEAD_S * RATE)), pcm, np.zeros(RATE)])
    chain_l = L0 + 4  # 20 ms off the truth
    onset_pos = (chain_l + onset_off) * HOP
    tick = {"pos": 0, "prior_pos": onset_pos, "prior_kind": "onset", "accepted": False}
    t, arrivals, env, n = _turn(pcm, rx, ticks=[tick], onset_prior_pos=onset_pos)
    res = _replay(t, arrivals, env, n, dsp.STALL, chain_prior=lambda pos: chain_l)
    first = res["ticks"][0]["prior_l"]
    assert first == pytest.approx(chain_l + onset_off if expect == "onset" else chain_l)
    # without rule 5 the onset prior is taken wherever it is
    off = _replay(
        t, arrivals, env, n, replace(dsp.STALL, chain_prior=False), chain_prior=lambda pos: chain_l
    )
    assert off["ticks"][0]["prior_l"] == pytest.approx(chain_l + onset_off)


# ---------------------------------------------------------------- the committed captures


@pytest.fixture(scope="module", params=_FIXTURES or [None], ids=lambda p: p.name if p else "none")
def capture(request):
    if request.param is None:
        pytest.skip("tests/fixtures/sync/* is not in the sdist")
    x, fs, sync, _ = rc.load(str(request.param))
    env, hop = rc.received_envelope(x, fs, True)
    return request.param, x, fs, sync, env, hop


def test_today_in_the_tracker_is_the_page_tick_for_tick(capture):
    _, x, fs, sync, env, hop = capture
    by_id = {t["turn_id"]: t for t in sync["turns"]}
    n = 0
    for t in sync["turns"]:
        if not len(rc.b64(t["reference"])):
            continue
        a = rc.replay_turn(t, env, hop, fs, len(x), dsp.DEFAULTS, by_id, sync["arrivals"])
        b = rc.replay_turn_tracked(
            t, env, hop, fs, len(x), dsp.DEFAULTS, by_id, sync["arrivals"], None, dsp.TODAY
        )
        keys = (
            "pos",
            "speech",
            "accepted",
            "l",
            "candidate_l",
            "prior_l",
            "wide",
            "spliced",
            "fails",
        )
        assert [{k: v[k] for k in keys} for v in a] == [{k: v[k] for k in keys} for v in b["ticks"]]
        n += len(a)
    assert n > 20


def test_the_committed_captures_never_freeze(capture):
    _, x, fs, sync, env, hop = capture
    out = rc.replay_all_tracked(env, hop, fs, len(x), sync, dsp.DEFAULTS, dsp.STALL)
    freezes = {
        tid: r["res"]["freezes"] for tid, r in out.items() if r["res"] and r["res"]["freezes"]
    }
    lost = {tid: r["res"]["lost_at"] for tid, r in out.items() if r["res"] and r["res"]["lost_at"]}
    assert not freezes and not lost


@pytest.mark.parametrize("causal", [False, True], ids=["zero-phase", "causal"])
def test_the_committed_captures_have_zero_false_locks_under_the_stall_rule(capture, causal):
    path = capture[0]
    with contextlib.redirect_stdout(io.StringIO()):
        results = rc.main([str(path), "--stall-rule"] + (["--causal"] if causal else []))
    errors = [e for r in results for e in r["window_errors_ms"]]
    assert len(errors) == sum(r["accepted"] for r in results)
    assert not [e for e in errors if e > 20.0]
