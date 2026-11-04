"""Opus round trip: ffmpeg/libopus encodes (bitrate, 20 ms frames, voip); libopus (ctypes)
decodes packet by packet so packets can be dropped and replaced by the decoder's real PLC."""

import ctypes
import ctypes.util
import os
import shutil
import subprocess

import numpy as np


def _load_opus():
    """libopus: ``$OPUS_LIB``, else the system's, else the Homebrew location."""
    for path in (
        os.environ.get("OPUS_LIB"),
        ctypes.util.find_library("opus"),
        "/opt/homebrew/lib/libopus.dylib",
    ):
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise OSError("libopus not found: install it or set OPUS_LIB to its path")


_lib = _load_opus()
_lib.opus_decoder_create.restype = ctypes.c_void_p
_lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_lib.opus_decode.restype = ctypes.c_int
_lib.opus_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.c_int,
    ctypes.c_int,
]
_lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]

FS = 48000
FRAME = 960  # 20 ms at 48 kHz


def encode(pcm48_int16, kbps):
    """Encode mono 48 kHz int16 -> (list of opus packets, pre_skip)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg (with libopus) is needed to encode")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-b:a",
        f"{kbps}k",
        "-frame_duration",
        "20",
        "-application",
        "voip",
        "-f",
        "ogg",
        "pipe:1",
    ]
    ogg = subprocess.run(
        cmd, input=pcm48_int16.astype("<i2").tobytes(), capture_output=True, check=True
    ).stdout
    packets, cur, pos = [], b"", 0
    while pos < len(ogg):
        assert ogg[pos : pos + 4] == b"OggS"
        nseg = ogg[pos + 26]
        lac = ogg[pos + 27 : pos + 27 + nseg]
        body = pos + 27 + nseg
        for l in lac:
            cur += ogg[body : body + l]
            body += l
            if l < 255:
                packets.append(cur)
                cur = b""
        pos = body
    head = packets[0]
    assert head[:8] == b"OpusHead"
    pre_skip = int.from_bytes(head[10:12], "little")
    return packets[2:], pre_skip


def decode(packets, pre_skip, n_out, plc_drop=(), plc_insert=None):
    """Decode to float32 48 kHz (int16 scale).
    plc_drop: packet indices replaced by decoder PLC (timeline unchanged).
    plc_insert: (packet_index, n_frames): before that packet, n_frames of PLC are inserted
                (late packets / jitter-buffer expand: timeline shifts later).
    Returns (pcm, rtime) where rtime[i] is the source sample index (48 kHz) or -1 for
    inserted audio."""
    err = ctypes.c_int()
    dec = _lib.opus_decoder_create(FS, 1, ctypes.byref(err))
    assert err.value == 0
    buf = (ctypes.c_int16 * 5760)()
    out, rt = [], []
    src = 0
    drop = set(plc_drop)
    for i, p in enumerate(packets):
        if plc_insert is not None and i == plc_insert[0]:
            for _ in range(plc_insert[1]):
                n = _lib.opus_decode(dec, None, 0, buf, FRAME, 0)
                out.append(np.frombuffer(buf, np.int16, n).copy())
                rt.append(np.full(n, -1, np.int64))
        if i in drop:
            n = _lib.opus_decode(dec, None, 0, buf, FRAME, 0)
        else:
            n = _lib.opus_decode(dec, p, len(p), buf, 5760, 0)
        assert n > 0, n
        out.append(np.frombuffer(buf, np.int16, n).copy())
        rt.append(np.arange(src, src + n))
        src += n
    _lib.opus_decoder_destroy(dec)
    pcm = np.concatenate(out).astype(np.float32)
    rtime = np.concatenate(rt)
    # pre-skip is in source time: drop the first pre_skip source samples
    keep = ~((rtime >= 0) & (rtime < pre_skip))
    pcm, rtime = pcm[keep], rtime[keep]
    rtime = np.where(rtime >= 0, rtime - pre_skip, -1)
    # trim to the input length (in source time)
    last = np.searchsorted(np.maximum.accumulate(rtime), n_out)
    return pcm[:last], rtime[:last]
