from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AvatarMessage(_message.Message):
    __slots__ = ("tag", "turn_start", "turn_onsets", "frames", "turn_end", "turn_abort", "state")
    TAG_FIELD_NUMBER: _ClassVar[int]
    TURN_START_FIELD_NUMBER: _ClassVar[int]
    TURN_ONSETS_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    TURN_END_FIELD_NUMBER: _ClassVar[int]
    TURN_ABORT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    tag: str
    turn_start: TurnStart
    turn_onsets: TurnOnsets
    frames: Frames
    turn_end: TurnEnd
    turn_abort: TurnAbort
    state: State
    def __init__(self, tag: _Optional[str] = ..., turn_start: _Optional[_Union[TurnStart, _Mapping]] = ..., turn_onsets: _Optional[_Union[TurnOnsets, _Mapping]] = ..., frames: _Optional[_Union[Frames, _Mapping]] = ..., turn_end: _Optional[_Union[TurnEnd, _Mapping]] = ..., turn_abort: _Optional[_Union[TurnAbort, _Mapping]] = ..., state: _Optional[_Union[State, _Mapping]] = ...) -> None: ...

class TurnStart(_message.Message):
    __slots__ = ("turn_id", "sample_rate", "fps", "onsets_ms", "prev_turn_id", "prev_turn_pcm_ms", "degraded")
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    ONSETS_MS_FIELD_NUMBER: _ClassVar[int]
    PREV_TURN_ID_FIELD_NUMBER: _ClassVar[int]
    PREV_TURN_PCM_MS_FIELD_NUMBER: _ClassVar[int]
    DEGRADED_FIELD_NUMBER: _ClassVar[int]
    turn_id: str
    sample_rate: int
    fps: int
    onsets_ms: _containers.RepeatedScalarFieldContainer[float]
    prev_turn_id: str
    prev_turn_pcm_ms: float
    degraded: bool
    def __init__(self, turn_id: _Optional[str] = ..., sample_rate: _Optional[int] = ..., fps: _Optional[int] = ..., onsets_ms: _Optional[_Iterable[float]] = ..., prev_turn_id: _Optional[str] = ..., prev_turn_pcm_ms: _Optional[float] = ..., degraded: bool = ...) -> None: ...

class TurnOnsets(_message.Message):
    __slots__ = ("turn_id", "onsets_ms")
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    ONSETS_MS_FIELD_NUMBER: _ClassVar[int]
    turn_id: str
    onsets_ms: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, turn_id: _Optional[str] = ..., onsets_ms: _Optional[_Iterable[float]] = ...) -> None: ...

class Frames(_message.Message):
    __slots__ = ("turn_id", "segment_seq", "chunk_index", "chunk_count", "data", "end")
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_SEQ_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    CHUNK_COUNT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    turn_id: str
    segment_seq: int
    chunk_index: int
    chunk_count: int
    data: bytes
    end: bool
    def __init__(self, turn_id: _Optional[str] = ..., segment_seq: _Optional[int] = ..., chunk_index: _Optional[int] = ..., chunk_count: _Optional[int] = ..., data: _Optional[bytes] = ..., end: bool = ...) -> None: ...

class TurnEnd(_message.Message):
    __slots__ = ("turn_id", "pcm_ms", "segment_count")
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    PCM_MS_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_COUNT_FIELD_NUMBER: _ClassVar[int]
    turn_id: str
    pcm_ms: float
    segment_count: int
    def __init__(self, turn_id: _Optional[str] = ..., pcm_ms: _Optional[float] = ..., segment_count: _Optional[int] = ...) -> None: ...

class TurnAbort(_message.Message):
    __slots__ = ("turn_id",)
    TURN_ID_FIELD_NUMBER: _ClassVar[int]
    turn_id: str
    def __init__(self, turn_id: _Optional[str] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ("status", "reason")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[State.Status]
        STATUS_OK: _ClassVar[State.Status]
        STATUS_DEGRADED: _ClassVar[State.Status]
    STATUS_UNSPECIFIED: State.Status
    STATUS_OK: State.Status
    STATUS_DEGRADED: State.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    status: State.Status
    reason: str
    def __init__(self, status: _Optional[_Union[State.Status, str]] = ..., reason: _Optional[str] = ...) -> None: ...
