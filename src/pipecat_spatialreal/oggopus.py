"""An Ogg Opus stream built page by page, for the ``audio`` messages of audio mode.

A client that cannot tap the bot's audio track (a phone on Daily) gets each turn's audio as one Ogg
Opus stream (RFC 7845) split over ``audio`` messages. The stream has to go out while the turn is
still being spoken, a few packets at a time, in pages small enough for one message, so it is framed
here rather than by a container library: PyAV encodes the Opus packets (its wheels bundle libopus),
and this module wraps them in Ogg pages.
"""

import os
import struct

import av
import numpy as np

# Input rates Opus accepts.
OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

# Largest page this stream emits, header included: one page always fits one ``audio`` message.
MAX_PAGE_BYTES = 2400

# Length of each Opus packet.
PACKET_MS = 40

# Opus bitrate: a voice at this rate is about 3 KB/s on the wire (FFmpeg's default is 64 kbps).
BITRATE = 24000

# Ogg Opus counts granule positions in 48 kHz samples whatever the input rate (RFC 7845, 4).
_GRANULE_RATE = 48000

_BOS, _EOS = 0x02, 0x04

# Pre-skip to declare if the encoder does not report its own (RFC 7845 recommends 3840 at most;
# libopus reports 312 at 48 kHz).
_DEFAULT_PRE_SKIP = 312


def _crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
        table.append(crc & 0xFFFFFFFF)
    return table


_CRC_TABLE = _crc_table()


def _crc(data: bytes) -> int:
    """The Ogg page checksum: CRC-32 with polynomial 0x04C11DB7, no reflection, no final xor."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[(crc >> 24) ^ byte]
    return crc


def _lacing(size: int) -> bytes:
    """The segment table entries of one packet: 255 for every full segment, then the remainder."""
    return bytes([255] * (size // 255) + [size % 255])


class OggOpusStream:
    """One Ogg Opus logical stream, fed PCM16 mono and read back as complete pages.

    ``feed`` returns the pages for the packets that its audio completed, ``close`` the rest of the
    stream. The first pages returned are the ``OpusHead`` and ``OpusTags`` header pages; a stream
    that never got any audio returns no pages at all.
    """

    def __init__(self, rate: int):
        """Create the stream.

        Args:
            rate: Sample rate of the PCM that will be fed, one of ``OPUS_RATES``.

        Raises:
            ValueError: If Opus does not take ``rate``.
        """
        if rate not in OPUS_RATES:
            raise ValueError(f"Opus takes {OPUS_RATES} Hz, not {rate}")
        self._rate = rate
        self._packet_bytes = 2 * rate * PACKET_MS // 1000
        self._codec: av.CodecContext | None = None
        self._pcm = bytearray()
        self._fed_samples = 0  # input samples fed in all, before any padding
        self._encoded = 0  # input samples handed to the encoder
        self._serial = struct.unpack("<I", os.urandom(4))[0]
        self._page_seq = 0
        self._packets = 0  # audio packets written to pages
        self._pre_skip = _DEFAULT_PRE_SKIP
        self._started = False  # the header pages are out
        self._closed = False

    def feed(self, pcm: bytes) -> list[bytes]:
        """Add PCM16 mono audio; return the pages its completed packets fill.

        Args:
            pcm: Audio at the stream's rate, little-endian 16-bit samples.

        Returns:
            Zero or more complete pages, in order.

        Raises:
            RuntimeError: If the stream is closed.
        """
        if self._closed:
            raise RuntimeError("the stream is closed")
        self._pcm.extend(pcm)
        self._fed_samples += len(pcm) // 2
        packets = []
        while len(self._pcm) >= self._packet_bytes:
            chunk = bytes(self._pcm[: self._packet_bytes])
            del self._pcm[: self._packet_bytes]
            packets.extend(self._encode(chunk))
        return self._pages(packets, last=False)

    def close(self) -> list[bytes]:
        """End the stream: pad the last packet, flush the encoder, return the remaining pages.

        The last page carries the end-of-stream flag and a granule position that trims the
        padding, so a decoder plays exactly the samples that were fed.

        Returns:
            The remaining pages, ending with the end-of-stream page; none when nothing was fed.
        """
        if self._closed:
            return []
        self._closed = True
        if self._fed_samples == 0:
            return []
        packets = []
        if self._pcm:
            chunk = bytes(self._pcm) + bytes(self._packet_bytes - len(self._pcm))
            self._pcm.clear()
            packets.extend(self._encode(chunk))
        packets.extend(self._flush())
        return self._pages(packets, last=True)

    # --- encoding

    def _open(self) -> av.CodecContext:
        codec = av.CodecContext.create("libopus", "w")
        codec.sample_rate = self._rate
        codec.layout = "mono"
        codec.format = "s16"
        codec.bit_rate = BITRATE
        codec.options = {"frame_duration": str(PACKET_MS), "application": "voip"}
        codec.open()
        head = bytes(codec.extradata or b"")
        if head[:8] == b"OpusHead" and len(head) >= 12:
            self._pre_skip = struct.unpack_from("<H", head, 10)[0]
        return codec

    def _encode(self, chunk: bytes) -> list[bytes]:
        if self._codec is None:
            self._codec = self._open()
        samples = np.frombuffer(chunk, dtype="<i2").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = self._rate
        frame.pts = self._encoded
        self._encoded += samples.shape[1]
        return [bytes(packet) for packet in self._codec.encode(frame)]

    def _flush(self) -> list[bytes]:
        if self._codec is None:
            return []
        return [bytes(packet) for packet in self._codec.encode(None)]

    # --- Ogg framing

    def _head(self) -> bytes:
        """The ``OpusHead`` packet (RFC 7845, 5.1): version 1, mono, channel mapping family 0."""
        return struct.pack("<8sBBHIhB", b"OpusHead", 1, 1, self._pre_skip, self._rate, 0, 0)

    @staticmethod
    def _tags() -> bytes:
        vendor = b"pipecat-spatialreal"
        return b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)

    def _page(self, packets: list[bytes], granule: int, flags: int) -> bytes:
        table = b"".join(_lacing(len(packet)) for packet in packets)
        header = struct.pack(
            "<4sBBqIIIB",
            b"OggS",
            0,
            flags,
            granule,
            self._serial,
            self._page_seq,
            0,
            len(table),
        )
        page = bytearray(header + table + b"".join(packets))
        struct.pack_into("<I", page, 22, _crc(bytes(page)))
        self._page_seq += 1
        return bytes(page)

    def _pages(self, packets: list[bytes], *, last: bool) -> list[bytes]:
        if not packets and not last:
            return []
        pages = []
        if not self._started:
            self._started = True
            pages.append(self._page([self._head()], 0, _BOS))
            pages.append(self._page([self._tags()], 0, 0))
        per_packet = _GRANULE_RATE * PACKET_MS // 1000
        # The exact end: the samples fed, at 48 kHz, after the pre-skip. Every page's granule is
        # capped at it, so the padding of the last packet and the encoder's flush are trimmed.
        end = self._pre_skip + self._fed_samples * _GRANULE_RATE // self._rate
        group: list[bytes] = []
        size = 27
        entries = 0  # segment table entries of the page being filled; a page holds 255 at most
        for index, packet in enumerate(packets):
            lacing = len(_lacing(len(packet)))
            cost = lacing + len(packet)
            if group and (size + cost > MAX_PAGE_BYTES or entries + lacing > 255):
                pages.append(self._page(group, min(self._packets * per_packet, end), 0))
                group, size, entries = [], 27, 0
            group.append(packet)
            size += cost
            entries += lacing
            self._packets += 1
            if index == len(packets) - 1:
                flags = _EOS if last else 0
                granule = end if last else min(self._packets * per_packet, end)
                pages.append(self._page(group, granule, flags))
                group = []
        if last and not packets:
            pages.append(self._page([], end, _EOS))  # nothing left to carry: an empty last page
        return pages
