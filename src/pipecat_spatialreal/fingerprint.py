"""Streaming log-RMS envelope of PCM audio, quantised to one byte per value.

The envelope is the audio half of a content fingerprint: the plugin computes it from the audio it
releases to the output transport and sends it to the browser, which computes the same curve on the
track it receives, so the two sides can be lined up on content rather than on arrival time.

The definition is a contract and must not drift:

* A value covers a window of ``window_ms`` (10 ms) and a new one starts every ``hop_ms`` (5 ms).
  Value ``k`` covers the released samples ``[k * hop, k * hop + window)``. A window counts only once
  it lies fully inside the audio fed so far, so the values do not depend on how the audio is cut
  into chunks. A 40 ms keyframe ``j`` is at index ``8 * j``.
* The value is ``10 * log10(mean_square / 32768**2 + 1e-10)`` in dB, clamped to ``[-96, 0]`` and
  mapped linearly to ``[0, 255]`` with round-half-up: ``floor((db + 96) / 96 * 255 + 0.5)``.
* The sum of squares is exact integer arithmetic over int16 little-endian mono samples; the only
  floating-point step is one ``math.log10`` per value. There is no numpy.

Hop and window lengths in samples are ``ms * sample_rate // 1000``. At 22 050 Hz the 5 ms hop is
110.25 samples and the 10 ms window 220.5; at 44 100 Hz the hop is 220.5 and the window 441. Where
a length is not whole it is truncated: the hops are 110 and 220 samples, the 22 050 Hz window 220
samples (the 44 100 Hz window, 441, is exact), and the envelope runs fractionally fast against a
true 5 ms clock. The processor only ever uses the transport's sample rate, which is 16, 24 or 48 kHz in
practice, where every length is exact. ``hop_ms`` may not exceed ``window_ms``.
"""

import math
import struct

# Full-scale int16 magnitude squared, the reference for 0 dB.
_FULL_SCALE_SQUARED = 32768**2

# Added to the normalised mean square so silence is finite (-100 dB) before the clamp.
_FLOOR = 1e-10

# dB range that maps to the byte range.
_DB_MIN = -96.0
_DB_MAX = 0.0


class Envelope:
    """Streaming log-RMS envelope with a 5 ms hop and one byte per value.

    Feed it int16 little-endian mono PCM in chunks of any size (a chunk may even end between the
    two bytes of a sample); it returns the values that the chunk completed.
    """

    def __init__(self, sample_rate: int, *, hop_ms: int = 5, window_ms: int = 10) -> None:
        """Create an envelope for audio at ``sample_rate``.

        Args:
            sample_rate: Samples per second of the audio that will be fed.
            hop_ms: Milliseconds between the starts of consecutive windows.
            window_ms: Length of one window in milliseconds.

        Raises:
            ValueError: If ``hop_ms`` exceeds ``window_ms`` (samples would go unmeasured) or a
                length in samples would be zero.
        """
        if hop_ms > window_ms:
            raise ValueError(f"hop_ms={hop_ms} exceeds window_ms={window_ms}")
        self._hop = hop_ms * sample_rate // 1000
        self._window = window_ms * sample_rate // 1000
        if self._hop < 1 or self._window < 1:
            raise ValueError(
                f"hop_ms={hop_ms}, window_ms={window_ms} are empty at {sample_rate} Hz"
            )
        self.reset()

    def reset(self) -> None:
        """Forget everything fed so far; the next value produced is index 0 again."""
        self._squares: list[int] = []  # squares of the samples not yet past the next window start
        self._index = 0
        self._carry = b""  # a trailing odd byte of the last chunk

    @property
    def index(self) -> int:
        """Index of the next value :meth:`feed` will produce."""
        return self._index

    def feed(self, pcm: bytes) -> bytes:
        """Add audio and return the values it completed.

        Args:
            pcm: int16 little-endian mono samples. A trailing odd byte is kept for the next call.

        Returns:
            One byte per newly completed window, in order; empty when none completed.
        """
        data = self._carry + pcm
        count = len(data) // 2
        self._carry = data[count * 2 :]
        if count:
            self._squares.extend(s * s for s in struct.unpack(f"<{count}h", data[: count * 2]))
        out = bytearray()
        squares = self._squares
        window, hop = self._window, self._hop
        start = 0
        while start + window <= len(squares):
            mean_square = sum(squares[start : start + window]) / window
            out.append(_quantise(mean_square))
            start += hop
            self._index += 1
        # Samples before the next window start are never read again (hop <= window, so they are
        # all in the buffer).
        del squares[:start]
        return bytes(out)


def _quantise(mean_square: float) -> int:
    """Map a mean square (int16 units) to its envelope byte."""
    return _to_byte(10 * math.log10(mean_square / _FULL_SCALE_SQUARED + _FLOOR))


def _to_byte(db: float) -> int:
    """Clamp a level in dB to ``[-96, 0]`` and map it to ``[0, 255]``, rounding half up."""
    db = max(_DB_MIN, min(_DB_MAX, db))
    return math.floor((db - _DB_MIN) / (_DB_MAX - _DB_MIN) * 255 + 0.5)
