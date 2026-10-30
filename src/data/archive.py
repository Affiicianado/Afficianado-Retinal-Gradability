"""Read the split EyePACS archive in place, without ever joining it on disk.

Copied from retina-01b-extend-eyeq (week-3-retina has the same class).

The Kaggle competition ships train.zip as five byte-split parts (~35 GB) and test.zip
as seven (~53 GB). The notebook working disk is capped at 20 GB, so the usual
"cat the parts together, then unzip" cannot work. zipfile only needs a seekable byte
stream: it reads the central directory from the end of the archive, then jumps to each
member's offset. SplitZipStream presents the parts as one such stream.
"""
import bisect
import glob
import io
import os
import zipfile
from itertools import accumulate


class SplitZipStream(io.RawIOBase):
    """Present train.zip.001..00N as a single seekable byte stream."""

    def __init__(self, parts):
        self.parts = sorted(parts)
        self.fhs = [open(p, "rb") for p in self.parts]
        self.sizes = [os.path.getsize(p) for p in self.parts]
        self.starts = [0] + list(accumulate(self.sizes))
        self.total = self.starts[-1]
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total + offset
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def readinto(self, buf):
        want, got = len(buf), 0
        while got < want and self.pos < self.total:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            local = self.pos - self.starts[i]
            self.fhs[i].seek(local)
            chunk = self.fhs[i].read(min(want - got, self.sizes[i] - local))
            if not chunk:
                break
            buf[got:got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self):
        for f in self.fhs:
            f.close()
        super().close()


def open_split(pattern, buffer_size=1 << 22):
    """Open a split archive. A BufferedReader on top is about 10x faster than raw."""
    parts = sorted(glob.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"no parts matched {pattern}")
    return zipfile.ZipFile(io.BufferedReader(SplitZipStream(parts),
                                             buffer_size=buffer_size))
