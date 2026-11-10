"""Builders for the JSON envelope the processor sends to the SpatialReal client SDKs.

Every builder returns the dict to pass as ``RTVIServerMessageFrame(data=...)``. The dict is an
``AvatarMessage`` from ``pipecat/v1/avatar.proto`` rendered under the contract's JSON rules: wire
keys are the proto field names, exactly one payload member appears beside ``tag``, and every key
inside that member is present ("unset" is ``""``, ``0``, ``[]``, ``false`` or the
``_UNSPECIFIED`` enum name). Bytes are standard base64, milliseconds are doubles and enums travel
as their names.

The builders fill a generated ``AvatarMessage`` and convert it with :func:`to_dict`, so the schema
alone decides the shape and no builder spells a wire key by hand. The builders only serialize: when
to send a message (never an empty ``turn_onsets``, never ``STATUS_UNSPECIFIED``) is the caller's
rule.

:func:`client_message` reads the other direction: an ``AvatarClientMessage`` a client sends in an
RTVI client message.
"""

from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict, ParseError

from pipecat_spatialreal.proto.generated import avatar_pb2

# Value of AvatarMessage.tag; lets a client skip other server messages.
TAG = "spatialreal-avatar"

# Most bytes that one frames chunk or one turn_envelope message carries. Base64 turns 2400
# bytes into 3200 characters (2400 is a multiple of 3, so a full chunk has no padding); with the
# keys, a UUID turn id and the RTVI wrapper that is about 3.4 KB, inside Daily's 4 KB app-message
# cap.
MAX_CHUNK_BYTES = 2400

# Keyframes per second of the animation the web SDK renders.
FPS = 25


def to_dict(msg: avatar_pb2.AvatarMessage) -> dict[str, Any]:
    """Render an ``AvatarMessage`` as the JSON-ready dict the contract describes.

    Args:
        msg: The message to render.

    Returns:
        A dict keyed by proto field names in which every field of the set payload member is
        present, bytes are base64 strings and enums are their names.
    """
    return MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


def turn_start(
    turn_id: str,
    *,
    sample_rate: int,
    onsets_ms: list[float],
    prev_turn_id: str = "",
    prev_turn_pcm_ms: float = 0.0,
    degraded: bool = False,
    gap_ms: float = 0.0,
) -> dict[str, Any]:
    """Build the message that announces a bot turn.

    Args:
        turn_id: The driveningress req_id of the turn.
        sample_rate: Hz of the audio sent to inference, which is also the room track's rate.
        onsets_ms: Offsets in ms from the turn's first sample of the audible onsets found so
            far, ascending; empty when none was found yet. A prior only: the browser places the
            frames by them until the turn's envelope locks.
        prev_turn_id: The previous turn when this one follows it without a silence gap, or an
            empty string.
        prev_turn_pcm_ms: Exact PCM length of ``prev_turn_id`` in ms; 0 when that is empty.
        degraded: True when no animation arrived in time, so the client plays audio only.
        gap_ms: Silence in ms the processor itself released between the previous turn's last
            sample and this turn's first, so the browser can join the two envelopes; 0 for a
            chained turn, the first turn, after an interruption and when the room ran dry.

    Returns:
        The ``turn_start`` envelope, with ``fps`` set to :data:`FPS`.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(
            tag=TAG,
            turn_start=avatar_pb2.TurnStart(
                turn_id=turn_id,
                sample_rate=sample_rate,
                fps=FPS,
                onsets_ms=onsets_ms,
                prev_turn_id=prev_turn_id,
                prev_turn_pcm_ms=prev_turn_pcm_ms,
                degraded=degraded,
                gap_ms=gap_ms,
            ),
        )
    )


def turn_onsets(turn_id: str, onsets_ms: list[float]) -> dict[str, Any]:
    """Build the message that reports onsets found after ``turn_start``.

    Args:
        turn_id: The driveningress req_id of the turn.
        onsets_ms: The new onsets only, ascending, in ms from the turn's first sample; none
            repeats an earlier entry.

    Returns:
        The ``turn_onsets`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(
            tag=TAG,
            turn_onsets=avatar_pb2.TurnOnsets(turn_id=turn_id, onsets_ms=onsets_ms),
        )
    )


def turn_envelope(
    turn_id: str,
    first_index: int,
    values: bytes,
    *,
    hop_ms: int = 5,
    window_ms: int = 10,
) -> list[dict[str, Any]]:
    """Split a run of envelope values into ``turn_envelope`` messages that each fit an app message.

    The values are one byte each, so a chunk of ``n`` bytes covers ``n`` consecutive indices.

    Args:
        turn_id: The driveningress req_id of the turn.
        first_index: 0-based index, counted from the turn's first released sample, of
            ``values[0]``.
        values: The envelope values, one byte per index, contiguous and in order.
        hop_ms: Spacing in ms between consecutive values.
        window_ms: Length in ms of the window each value is measured over.

    Returns:
        One envelope per :data:`MAX_CHUNK_BYTES` of ``values`` in index order, each with its own
        ``first_index``; an empty list when ``values`` is empty.
    """
    return [
        to_dict(
            avatar_pb2.AvatarMessage(
                tag=TAG,
                turn_envelope=avatar_pb2.TurnEnvelope(
                    turn_id=turn_id,
                    first_index=first_index + offset,
                    hop_ms=hop_ms,
                    window_ms=window_ms,
                    values=values[offset : offset + MAX_CHUNK_BYTES],
                ),
            )
        )
        for offset in range(0, len(values), MAX_CHUNK_BYTES)
    ]


def frames(
    turn_id: str,
    segment_seq: int,
    wire: bytes,
    *,
    end: bool,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
) -> list[dict[str, Any]]:
    """Split one driveningress message into ``frames`` envelopes that each fit one app message.

    Args:
        turn_id: The driveningress req_id of the turn.
        segment_seq: 0-based position of this segment within the turn.
        wire: The raw ``driveningress.v2.Message`` bytes, exactly as driveningress sent them.
        end: True when this is the turn's last segment; it is set on every chunk.
        max_chunk_bytes: Most bytes of ``wire`` that one chunk carries.

    Returns:
        ``ceil(len(wire) / max_chunk_bytes)`` envelopes in chunk order, all with the same
        ``chunk_count``. An empty ``wire`` still gives one chunk, with empty data.

    Raises:
        ValueError: If ``max_chunk_bytes`` is less than 1.
    """
    if max_chunk_bytes < 1:
        raise ValueError(f"max_chunk_bytes must be at least 1, got {max_chunk_bytes}")
    pieces = [wire[i : i + max_chunk_bytes] for i in range(0, len(wire), max_chunk_bytes)]
    if not pieces:
        pieces = [b""]  # chunk_count is at least 1: an empty wire travels as one empty chunk
    return [
        to_dict(
            avatar_pb2.AvatarMessage(
                tag=TAG,
                frames=avatar_pb2.Frames(
                    turn_id=turn_id,
                    segment_seq=segment_seq,
                    chunk_index=index,
                    chunk_count=len(pieces),
                    data=piece,
                    end=end,
                ),
            )
        )
        for index, piece in enumerate(pieces)
    ]


def turn_end(turn_id: str, *, pcm_ms: float, segment_count: int) -> dict[str, Any]:
    """Build the message that says a turn's audio has finished playing out.

    Args:
        turn_id: The driveningress req_id of the turn.
        pcm_ms: Exact length of the turn's audio in ms, stall silence included.
        segment_count: Segments delivered for this turn; may be 0 for a degraded turn.

    Returns:
        The ``turn_end`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(
            tag=TAG,
            turn_end=avatar_pb2.TurnEnd(
                turn_id=turn_id,
                pcm_ms=pcm_ms,
                segment_count=segment_count,
            ),
        )
    )


def turn_abort(turn_id: str) -> dict[str, Any]:
    """Build the message that tells the client a turn was interrupted.

    The client drops the turn's remaining frames and blends to idle at once.

    Args:
        turn_id: The driveningress req_id of the interrupted turn.

    Returns:
        The ``turn_abort`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(tag=TAG, turn_abort=avatar_pb2.TurnAbort(turn_id=turn_id))
    )


def state(status: int, reason: str = "") -> dict[str, Any]:
    """Build the message that reports a change in the processor's health.

    Args:
        status: An ``avatar_pb2.State.Status`` value, ``STATUS_OK`` or ``STATUS_DEGRADED``; the
            JSON carries its name.
        reason: Free text for logs.

    Returns:
        The ``state`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(tag=TAG, state=avatar_pb2.State(status=status, reason=reason))
    )


# RTVI client-message type (``data.t``) under which a client sends an ``AvatarClientMessage``.
CLIENT_MESSAGE_TYPE = "spatialreal-avatar"


def audio(turn_id: str, seq: int, data: bytes, *, end: bool) -> dict[str, Any]:
    """Build one piece of a turn's audio, for clients that cannot tap the bot's audio track.

    Args:
        turn_id: The turn the audio belongs to.
        seq: 0-based position of this piece within the turn.
        data: Whole Ogg pages of the turn's Ogg Opus stream, at most ``MAX_CHUNK_BYTES``.
        end: True on the turn's last piece.

    Returns:
        The ``audio`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(
            tag=TAG, audio=avatar_pb2.Audio(turn_id=turn_id, seq=seq, data=data, end=end)
        )
    )


def audio_mode(*, active: bool, first_turn_id: str = "", sample_rate: int = 0) -> dict[str, Any]:
    """Build the answer to an audio request, or the notice that audio mode stopped.

    Args:
        active: True while ``audio`` messages are sent; False once no more follow.
        first_turn_id: The first turn whose audio comes as ``audio`` messages; ``""`` when
            ``active`` is False.
        sample_rate: Opus input rate of the streams in Hz; 0 when ``active`` is False.

    Returns:
        The ``audio_mode`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(
            tag=TAG,
            audio_mode=avatar_pb2.AudioMode(
                active=active, first_turn_id=first_turn_id, sample_rate=sample_rate
            ),
        )
    )


def turn_audio_only(turn_id: str) -> dict[str, Any]:
    """Build the message that says a turn lost its animation after its ``turn_start``.

    Args:
        turn_id: The turn that gets no more frames.

    Returns:
        The ``turn_audio_only`` envelope.
    """
    return to_dict(
        avatar_pb2.AvatarMessage(tag=TAG, turn_audio_only=avatar_pb2.TurnAudioOnly(turn_id=turn_id))
    )


def client_message(type_: str, data: object) -> avatar_pb2.AvatarClientMessage | None:
    """Read an ``AvatarClientMessage`` from an RTVI client message, if it is one.

    Args:
        type_: The client message's type (``data.t`` on the wire).
        data: Its payload (``data.d``), as Pipecat hands it over.

    Returns:
        The message, or None when the type is not ``CLIENT_MESSAGE_TYPE``, the payload does not
        parse, its tag is not ``TAG``, or it carries no payload this version knows.
    """
    if type_ != CLIENT_MESSAGE_TYPE or not isinstance(data, dict):
        return None
    try:
        msg = ParseDict(data, avatar_pb2.AvatarClientMessage(), ignore_unknown_fields=True)
    except (ParseError, TypeError, ValueError):
        return None
    member = msg.WhichOneof("payload")
    if msg.tag != TAG or member is None or not isinstance(data.get(member), dict):
        return None  # protobuf's JSON parser would also take a non-object for an empty message
    return msg
