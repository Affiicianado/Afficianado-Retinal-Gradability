import os, io, glob, bisect, zipfile
from itertools import accumulate

class SplitZipStream(io.RawIOBase):
    """Present train.zip.001..005 as one seekable stream."""

    def __init__(self, parts):
        self.parts  = sorted(parts)
        self.fhs    = [open(p, "rb") for p in self.parts]
        self.sizes  = [os.path.getsize(p) for p in self.parts]
        self.starts = [0] + list(accumulate(self.sizes))
        self.total  = self.starts[-1]
        self.pos    = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self):     return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if   whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.total + offset
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def readinto(self, buf):
        want, got = len(buf), 0
        while got < want and self.pos < self.total:
            i     = bisect.bisect_right(self.starts, self.pos) - 1
            local = self.pos - self.starts[i]
            self.fhs[i].seek(local)
            chunk = self.fhs[i].read(min(want - got, self.sizes[i] - local))
            if not chunk:
                break
            buf[got:got + len(chunk)] = chunk
            got      += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self):
        for f in self.fhs:
            f.close()
        super().close()


def open_split(pattern):
    parts = sorted(glob.glob(pattern))
    assert parts, f"no parts matched {pattern}"
    stream = io.BufferedReader(SplitZipStream(parts), buffer_size=1 << 22)
    return zipfile.ZipFile(stream)