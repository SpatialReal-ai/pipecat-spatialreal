import base64
import json

import pytest
from google.protobuf.json_format import ParseDict

from pipecat_spatialreal.envelope import (
    FPS,
    MAX_CHUNK_BYTES,
    TAG,
    frames,
    state,
    to_dict,
    turn_abort,
    turn_end,
    turn_envelope,
    turn_onsets,
    turn_start,
)
from pipecat_spatialreal.proto.generated import avatar_pb2


def test_frames_chunks_fit_daily_cap():
    wire = bytes(range(256)) * 63  # 16128 B, one real first segment is ~16 KB
    msgs = frames("r1", 0, wire, end=False)
    assert [m["frames"]["chunk_index"] for m in msgs] == list(range(7))
    assert all(m["frames"]["chunk_count"] == 7 and m["tag"] == "spatialreal-avatar" for m in msgs)
    assert all(len(json.dumps(m)) <= 3584 for m in msgs)
    assert b"".join(base64.b64decode(m["frames"]["data"]) for m in msgs) == wire


def test_turn_start_shape():
    d = turn_start("r1", sample_rate=24000, onsets_ms=[400.0])
    assert d == {
        "tag": "spatialreal-avatar",
        "turn_start": {
            "turn_id": "r1",
            "sample_rate": 24000,
            "fps": 25,
            "onsets_ms": [400.0],
            "prev_turn_id": "",
            "prev_turn_pcm_ms": 0.0,
            "degraded": False,
            "gap_ms": 0.0,
        },
    }


def test_dicts_parse_back_into_proto():
    for d in (
        turn_onsets("r1", [1300.0]),
        turn_end("r1", pcm_ms=2400.0, segment_count=2),
        turn_abort("r1"),
        state(avatar_pb2.State.STATUS_DEGRADED, "driveningress down"),
    ):
        ParseDict(d, avatar_pb2.AvatarMessage())  # raises on a wrong shape


# --- Contract rules the tests above leave unpinned. ---


def test_constants():
    assert (TAG, MAX_CHUNK_BYTES, FPS) == ("spatialreal-avatar", 2400, 25)


def test_to_dict_uses_proto_field_names_and_prints_defaults():
    msg = avatar_pb2.AvatarMessage(tag=TAG, turn_abort=avatar_pb2.TurnAbort())
    # An all-default member is still present, and its keys are the proto field names.
    assert to_dict(msg) == {"tag": TAG, "turn_abort": {"turn_id": ""}}


def test_turn_start_chains_previous_turn_and_keeps_empty_onsets():
    d = turn_start(
        "r2",
        sample_rate=16000,
        onsets_ms=[],
        prev_turn_id="r1",
        prev_turn_pcm_ms=2400.0,
        degraded=True,
    )
    assert d == {
        "tag": TAG,
        "turn_start": {
            "turn_id": "r2",
            "sample_rate": 16000,
            "fps": FPS,
            "onsets_ms": [],  # present, never omitted
            "prev_turn_id": "r1",
            "prev_turn_pcm_ms": 2400.0,
            "degraded": True,
            "gap_ms": 0.0,
        },
    }


def test_turn_start_carries_a_non_zero_gap_ms():
    d = turn_start("r2", sample_rate=16000, onsets_ms=[], gap_ms=312.5)
    assert d["turn_start"]["gap_ms"] == 312.5
    assert ParseDict(d, avatar_pb2.AvatarMessage()).turn_start.gap_ms == 312.5


def test_turn_onsets_shape():
    assert turn_onsets("r1", [1300.0, 2900.5]) == {
        "tag": TAG,
        "turn_onsets": {"turn_id": "r1", "onsets_ms": [1300.0, 2900.5]},
    }


def test_turn_end_shape():
    assert turn_end("r1", pcm_ms=2400.0, segment_count=2) == {
        "tag": TAG,
        "turn_end": {"turn_id": "r1", "pcm_ms": 2400.0, "segment_count": 2},
    }


def test_turn_abort_shape():
    assert turn_abort("r1") == {"tag": TAG, "turn_abort": {"turn_id": "r1"}}


def test_state_carries_the_enum_name_and_keeps_an_empty_reason():
    assert state(avatar_pb2.State.STATUS_DEGRADED, "driveningress down") == {
        "tag": TAG,
        "state": {"status": "STATUS_DEGRADED", "reason": "driveningress down"},
    }
    assert state(avatar_pb2.State.STATUS_OK) == {
        "tag": TAG,
        "state": {"status": "STATUS_OK", "reason": ""},
    }


@pytest.mark.parametrize(
    ("length", "sizes"),
    [
        pytest.param(0, [0], id="empty-wire-is-one-empty-chunk"),
        pytest.param(1, [1], id="one-byte"),
        pytest.param(4, [4], id="exactly-one-chunk"),
        pytest.param(5, [4, 1], id="one-over"),
        pytest.param(8, [4, 4], id="exact-multiple-has-no-trailing-empty-chunk"),
        pytest.param(9, [4, 4, 1], id="one-over-a-multiple"),
        pytest.param(10, [4, 4, 2], id="short-tail"),
    ],
)
def test_frames_slices_by_max_chunk_bytes_exactly(length, sizes):
    wire = bytes(i % 251 for i in range(length))  # every position distinguishable
    msgs = frames("r1", 2, wire, end=False, max_chunk_bytes=4)
    chunks = [base64.b64decode(m["frames"]["data"]) for m in msgs]
    assert [len(c) for c in chunks] == sizes
    assert [m["frames"]["chunk_index"] for m in msgs] == list(range(len(sizes)))
    assert {m["frames"]["chunk_count"] for m in msgs} == {len(sizes)}
    assert b"".join(chunks) == wire


@pytest.mark.parametrize("end", [True, False])
def test_frames_carry_turn_segment_and_end_on_every_chunk(end):
    msgs = frames("r9", 3, b"x" * 10, end=end, max_chunk_bytes=4)
    assert len(msgs) == 3
    for m in msgs:
        assert set(m) == {"tag", "frames"}
        assert set(m["frames"]) == {
            "turn_id",
            "segment_seq",
            "chunk_index",
            "chunk_count",
            "data",
            "end",
        }
        assert m["frames"]["turn_id"] == "r9"
        assert m["frames"]["segment_seq"] == 3
        assert m["frames"]["end"] is end


def test_frames_empty_wire_keeps_every_key_with_empty_data():
    (msg,) = frames("r1", 0, b"", end=True)
    assert msg == {
        "tag": TAG,
        "frames": {
            "turn_id": "r1",
            "segment_seq": 0,
            "chunk_index": 0,
            "chunk_count": 1,
            "data": "",
            "end": True,
        },
    }


@pytest.mark.parametrize("bad", [0, -1])
def test_frames_rejects_a_chunk_size_below_one(bad):
    # Without the guard a negative size would silently return one empty chunk and drop the wire.
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        frames("r1", 0, b"abc", end=False, max_chunk_bytes=bad)


def test_turn_envelope_keys_and_base64():
    values = bytes([0, 1, 2, 250, 255])
    msgs = turn_envelope("r1", 7, values)
    assert msgs == [
        {
            "tag": TAG,
            "turn_envelope": {
                "turn_id": "r1",
                "first_index": 7,
                "hop_ms": 5,
                "window_ms": 10,
                "values": base64.b64encode(values).decode(),
            },
        }
    ]


def test_turn_envelope_carries_custom_hop_and_window():
    (msg,) = turn_envelope("r1", 0, b"\x01", hop_ms=10, window_ms=20)
    assert (msg["turn_envelope"]["hop_ms"], msg["turn_envelope"]["window_ms"]) == (10, 20)


def test_turn_envelope_chunks_at_2400_bytes():
    values = bytes(i % 256 for i in range(5000))
    msgs = turn_envelope("r1", 0, values)
    assert [m["turn_envelope"]["first_index"] for m in msgs] == [0, 2400, 4800]
    chunks = [base64.b64decode(m["turn_envelope"]["values"]) for m in msgs]
    assert [len(c) for c in chunks] == [2400, 2400, 200]
    assert b"".join(chunks) == values


def test_turn_envelope_first_index_offsets_every_chunk():
    msgs = turn_envelope("r1", 100, bytes(2401))
    assert [m["turn_envelope"]["first_index"] for m in msgs] == [100, 2500]


def test_turn_envelope_empty_values_gives_no_messages():
    assert turn_envelope("r1", 0, b"") == []


def test_turn_envelope_round_trips_through_the_generated_message():
    values = bytes(i % 256 for i in range(2500))
    msgs = turn_envelope("r1", 40, values, hop_ms=5, window_ms=10)
    parsed = [ParseDict(m, avatar_pb2.AvatarMessage()) for m in msgs]
    assert all(p.WhichOneof("payload") == "turn_envelope" for p in parsed)
    assert [p.turn_envelope.first_index for p in parsed] == [40, 2440]
    assert all(p.turn_envelope.turn_id == "r1" for p in parsed)
    assert all((p.turn_envelope.hop_ms, p.turn_envelope.window_ms) == (5, 10) for p in parsed)
    assert b"".join(p.turn_envelope.values for p in parsed) == values
