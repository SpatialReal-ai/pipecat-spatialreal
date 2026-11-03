"""Streaming detector for the audible onsets of a turn's audio.

The processor reports the onsets it finds in the PCM it sends to inference (``onsets_ms`` in
``pipecat/v1/avatar.proto``), and the web SDK applies the same rule to the same audio arriving as
a room track. The rule is therefore part of the contract between the two and must not drift.
"""

import sys
from array import array


class OnsetTracker:
    """Finds the audible onsets in PCM audio that arrives in chunks of any size.

    The audio is PCM16 little-endian mono. A sample is audible when its absolute value is greater
    than ``threshold`` and silent otherwise, so a sample at exactly ``threshold`` is silent. An
    onset is an audible sample that follows at least ``min_silence_ms`` of consecutive silent
    samples; the rest of the same sound is not an onset. The start of the stream counts as such a
    silence whatever its length, so audio that begins with sound has an onset at 0 ms and audio
    that begins with 400 ms of silence has its first onset at 400 ms.

    Offsets are in ms from the first sample ever fed. Where the audio is cut into chunks does not
    change the result.
    """

    def __init__(self, sample_rate: int, *, threshold: int = 328, min_silence_ms: int = 150):
        """Start tracking at sample 0.

        Args:
            sample_rate: Hz of the audio that will be fed.
            threshold: Largest absolute sample value that still counts as silence; the default,
                328, is 1 % of int16 full scale.
            min_silence_ms: Shortest run of silence, in ms, that makes the next audible sample an
                onset.

        Raises:
            ValueError: If ``sample_rate`` or ``min_silence_ms`` is less than 1, or ``threshold``
                is negative.
        """
        if sample_rate < 1:
            raise ValueError(f"sample_rate must be at least 1, got {sample_rate}")
        if threshold < 0:
            raise ValueError(f"threshold must not be negative, got {threshold}")
        if min_silence_ms < 1:
            raise ValueError(f"min_silence_ms must be at least 1, got {min_silence_ms}")
        self._sample_rate = sample_rate
        self._threshold = threshold
        # Whole samples that last at least min_silence_ms: ceil(min_silence_ms * rate / 1000).
        self._min_silence_samples = (min_silence_ms * sample_rate + 999) // 1000
        self._fed = 0  # whole samples scanned so far
        self._silent_run = 0  # silent samples in a row, ending at the last one scanned
        # True when the next audible sample is an onset. It starts true: the stream's leading
        # run of silence counts however short, and none at all is a run of length 0.
        self._armed = True
        self._carry = b""  # the odd byte that ended the last feed, the low half of a sample

    @property
    def fed_ms(self) -> float:
        """Length in ms of the whole samples fed so far; a held-back odd byte is not counted."""
        return self._ms(self._fed)

    def feed(self, pcm: bytes) -> list[float]:
        """Scan the next stretch of audio for onsets.

        Args:
            pcm: The next PCM16 little-endian mono samples, in any number. A trailing byte that
                does not complete a sample is held back and completed by the next call.

        Returns:
            The onsets found in the samples this call completed, ascending, as offsets in ms from
            the first sample ever fed; empty when there are none. An onset is returned once, by
            the call that scans its sample.
        """
        data = self._carry + pcm
        whole = len(data) - len(data) % 2
        self._carry = data[whole:]
        samples = array("h")
        samples.frombytes(data[:whole])
        if sys.byteorder == "big":
            samples.byteswap()  # array reads the machine's byte order, the audio is little-endian

        onsets: list[float] = []
        for index, sample in enumerate(samples, start=self._fed):
            if abs(sample) > self._threshold:
                if self._armed:
                    onsets.append(self._ms(index))
                    self._armed = False
                self._silent_run = 0
            else:
                self._silent_run += 1
                if self._silent_run >= self._min_silence_samples:
                    self._armed = True
        self._fed += len(samples)
        return onsets

    def _ms(self, samples: int) -> float:
        """Convert a count of samples to ms.

        One division of two integers gives the double nearest the true value, so a whole number of
        40 ms keyframes comes out as exactly ``k * 40.0``. ``samples / rate * 1000`` rounds twice
        and sometimes lands just below (8039.999999999999 for 192960 samples at 24 kHz), which
        the client's ``floor(onset_ms * 25 / 1000)`` would turn into the previous keyframe.

        Args:
            samples: Number of samples.

        Returns:
            The duration of ``samples`` at the tracker's sample rate, in ms.
        """
        return samples * 1000 / self._sample_rate
