"""The Ogg Opus stream that carries a turn's audio in audio mode (pipecat/v1 Audio)."""

import io
import math
import struct

import av
import numpy as np
import pytest

from pipecat_spatialreal.oggopus import MAX_PAGE_BYTES, OPUS_RATES, OggOpusStream


def tone(seconds, rate, freq=440.0):
    n = round(seconds * rate)
    return struct.pack(
        f"<{n}h", *(round(9000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))
    )


def split_pages(stream: bytes):
    """Cut an Ogg byte stream into pages, checking each page's framing on the way."""
    pages, pos = [], 0
    while pos < len(stream):
        assert stream[pos : pos + 4] == b"OggS"
        nseg = stream[pos + 26]
        size = 27 + nseg + sum(stream[pos + 27 : pos + 27 + nseg])
        pages.append(stream[pos : pos + size])
        pos += size
    return pages


def ogg_crc(page: bytes) -> int:
    crc = 0
    for byte in page[:22] + b"\0\0\0\0" + page[26:]:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
            crc &= 0xFFFFFFFF
    return crc


def decode(pages):
    """Decode an Ogg Opus stream with PyAV; return its 48 kHz PCM16 samples, pre-skip trimmed."""
    container = av.open(io.BytesIO(b"".join(pages)), format="ogg")
    out = []
    for frame in container.decode(audio=0):
        out.append(frame.to_ndarray().reshape(-1))
    pcm = np.concatenate(out)
    if pcm.dtype.kind == "f":
        pcm = np.round(pcm * 32767)
    return pcm.astype(np.int16)


def correlation(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n].astype(float), b[:n].astype(float))[0, 1])


def best_correlation(decoded, original):
    """Correlation at the best small lag: the codec may shift the signal by a few samples."""
    return max(correlation(decoded[lag:], original) for lag in range(0, 400, 4))


def test_a_stream_decodes_back_to_its_audio():
    rate = 24000
    pcm = tone(1.0, rate)
    stream = OggOpusStream(rate)
    pages = stream.feed(pcm) + stream.close()
    decoded = decode(pages)
    # Decoders output 48 kHz; the trimmed length is the input's length at 48 kHz.
    assert len(decoded) == 48000
    original_48k = np.frombuffer(tone(1.0, 48000), dtype="<i2")
    assert best_correlation(decoded, original_48k) > 0.9


def test_pages_are_framed_as_rfc_7845_says():
    stream = OggOpusStream(24000)
    pages = stream.feed(tone(0.5, 24000)) + stream.close()
    whole = split_pages(b"".join(pages))
    assert whole == pages  # every returned item is exactly one page
    assert all(len(p) <= MAX_PAGE_BYTES for p in pages)
    assert all(struct.unpack_from("<I", p, 22)[0] == ogg_crc(p) for p in pages)
    header_type = [p[5] for p in pages]
    assert header_type[0] & 0x02 and not any(t & 0x02 for t in header_type[1:])  # BOS first only
    assert header_type[-1] & 0x04 and not any(t & 0x04 for t in header_type[:-1])  # EOS last only
    assert pages[0][28:36] == b"OpusHead" and pages[1][28:36] == b"OpusTags"
    serials = {struct.unpack_from("<I", p, 14)[0] for p in pages}
    assert len(serials) == 1
    seqs = [struct.unpack_from("<I", p, 18)[0] for p in pages]
    assert seqs == list(range(len(pages)))
    granules = [struct.unpack_from("<q", p, 6)[0] for p in pages]
    assert granules[:2] == [0, 0] and granules == sorted(granules)


def test_pages_come_out_as_soon_as_a_packet_is_complete():
    rate = 24000
    stream = OggOpusStream(rate)
    ten_ms = tone(0.01, rate)
    got = [stream.feed(ten_ms) for _ in range(40)]  # 400 ms in 10 ms pieces
    audio_pages = [len(p) for p in got]
    # Nothing before the first 40 ms packet; from then on audio flows, at most one packet behind.
    assert sum(audio_pages[:3]) == 0
    packets_out = sum(1 for pages in got for p in pages if p[28:32] != b"Opus")
    assert packets_out >= 40 // 4 - 1


def test_a_stream_that_got_no_audio_sends_nothing():
    stream = OggOpusStream(24000)
    assert stream.feed(b"") == []
    assert stream.close() == []


def test_a_partial_last_packet_is_kept_and_trimmed():
    rate = 16000
    pcm = tone(0.13, rate)  # 130 ms: three whole packets and a 10 ms remainder
    stream = OggOpusStream(rate)
    pages = stream.feed(pcm) + stream.close()
    decoded = decode(pages)
    assert len(decoded) == round(0.13 * 48000)


@pytest.mark.parametrize("rate", OPUS_RATES)
def test_every_opus_rate_round_trips(rate):
    stream = OggOpusStream(rate)
    pages = stream.feed(tone(0.2, rate)) + stream.close()
    decoded = decode(pages)
    assert len(decoded) == round(0.2 * 48000)
    assert struct.unpack_from("<I", pages[0], 27 + pages[0][26] + 12)[0] == rate  # input rate


def test_a_rate_opus_does_not_take_is_refused():
    with pytest.raises(ValueError):
        OggOpusStream(22050)


def test_feeding_after_close_is_an_error():
    stream = OggOpusStream(24000)
    stream.feed(tone(0.1, 24000))
    stream.close()
    with pytest.raises(RuntimeError):
        stream.feed(tone(0.1, 24000))


def test_a_page_never_has_more_than_255_segment_entries():
    rate = 24000
    stream = OggOpusStream(rate)
    # 254 packets of digital silence encode to tiny one-entry packets; the speech after them to
    # packets of 255 bytes or more, which take two entries: one feed that would put 256 entries on
    # a page if the limit counted packets.
    silence_s = 254 * 0.04
    pcm = bytes(2 * round(rate * silence_s)) + tone(1.0, rate)
    pages = stream.feed(pcm) + stream.close()
    assert all(p[26] <= 255 for p in pages)
    assert split_pages(b"".join(pages)) == pages
    assert len(decode(pages)) == round((silence_s + 1.0) * 48000)


def test_speech_is_encoded_at_a_voice_bitrate():
    rate = 24000
    stream = OggOpusStream(rate)
    pages = stream.feed(tone(4.0, rate)) + stream.close()
    audio_bytes = sum(len(p) for p in pages[2:])
    assert audio_bytes / 4.0 < 4000  # about 3 KB/s for a voice, not FFmpeg's 64 kbps default
