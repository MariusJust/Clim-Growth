"""Minimal pure-Python reader for Keras old-style (superblock v0, symbol-table)
.weights.h5 files. Extracts contiguous/compact datasets as numpy arrays keyed
by their full group path. No h5py / HDF5 C library needed.

Scope: enough of the HDF5 1.x format to walk symbol-table groups and read
uncompressed Dense-layer weight datasets (float32/float64, contiguous or
compact). Not a general HDF5 reader.
"""
from __future__ import annotations
import struct
import numpy as np


class _Reader:
    def __init__(self, path):
        with open(path, "rb") as fh:
            self.b = fh.read()
        self.datasets = {}
        self._parse_superblock()

    def u(self, off, n):
        return int.from_bytes(self.b[off:off + n], "little")

    # ---- superblock v0 ----
    def _parse_superblock(self):
        assert self.b[:8] == b"\x89HDF\r\n\x1a\n", "not an HDF5 file"
        assert self.b[8] == 0, f"only superblock v0 supported (got {self.b[8]})"
        self.size_off = self.b[13]
        self.size_len = self.b[14]
        # base address sits after the 4-byte consistency flags at offset 24
        self.base = self.u(24, self.size_off)
        # root group symbol table entry begins at offset 56; object header
        # address is the 2nd field (after link name offset)
        root_oh = self.u(56 + self.size_off, self.size_off)
        self._walk_group(root_oh, prefix="")

    def addr(self, a):
        return a + (self.base if self.base not in (0, 2 ** (8 * self.size_off) - 1) else 0)

    # ---- object header v1 ----
    def _messages(self, oh_addr):
        """Yield (msg_type, body_offset, body_size), following continuations.

        Object header v1 stores the size of the messages in the first block in
        the header-size field; messages beyond that live in continuation
        blocks pointed to by type-0x10 messages. Respecting the block bounds is
        essential -- otherwise the scan runs off the end into B-tree bytes.
        """
        ver = self.b[oh_addr]
        assert ver == 1, f"object header v{ver} unsupported"
        nmsg = self.u(oh_addr + 2, 2)
        hdr_size = self.u(oh_addr + 8, 4)
        start = (oh_addr + 12 + 7) & ~7  # messages begin after 12-byte prefix
        blocks = [(start, start + hdr_size)]
        count = 0
        bi = 0
        while bi < len(blocks) and count < nmsg:
            pos, end = blocks[bi]
            bi += 1
            while pos + 8 <= end and count < nmsg:
                mtype = self.u(pos, 2)
                msize = self.u(pos + 2, 2)
                body = pos + 8
                count += 1
                if mtype == 0x10:  # continuation block
                    coff = self.u(body, self.size_off)
                    clen = self.u(body + self.size_off, self.size_len)
                    blocks.append((coff, coff + clen))
                else:
                    yield mtype, body, msize
                pos = body + ((msize + 7) & ~7)

    # ---- group walking via symbol table ----
    def _walk_group(self, oh_addr, prefix):
        btree = heap = None
        for mtype, body, msize in self._messages(oh_addr):
            if mtype == 0x11:  # symbol table message
                btree = self.u(body, self.size_off)
                heap = self.u(body + self.size_off, self.size_off)
        if btree is None:
            return
        heap_data = self._heap_data(heap)
        for name_off, child_oh in self._btree_entries(btree):
            name = self._cstr(heap_data, name_off)
            path = f"{prefix}/{name}" if prefix else name
            if self._is_group(child_oh):
                self._walk_group(child_oh, path)
            else:
                arr = self._read_dataset(child_oh)
                if arr is not None:
                    self.datasets[path] = arr

    def _heap_data(self, heap_addr):
        assert self.b[heap_addr:heap_addr + 4] == b"HEAP"
        data_seg = self.u(heap_addr + 8 + self.size_len + self.size_len, self.size_off)
        return data_seg

    def _cstr(self, data_seg, off):
        p = data_seg + off
        e = self.b.index(b"\x00", p)
        return self.b[p:e].decode("ascii", "replace")

    def _btree_entries(self, btree_addr):
        out = []
        self._btree_recurse(btree_addr, out)
        return out

    def _btree_recurse(self, addr, out):
        assert self.b[addr:addr + 4] == b"TREE", "expected TREE node"
        level = self.b[addr + 5]
        nused = self.u(addr + 6, 2)
        p = addr + 8 + 2 * self.size_off  # skip siblings
        # keys (nused+1) and children (nused) interleaved: key,child,...,key
        children = []
        p += self.size_len  # first key
        for _ in range(nused):
            child = self.u(p, self.size_off); p += self.size_off
            p += self.size_len  # following key
            children.append(child)
        if level > 0:
            for c in children:
                self._btree_recurse(c, out)
        else:
            for c in children:  # each child is an SNOD
                out.extend(self._snod_entries(c))

    def _snod_entries(self, addr):
        assert self.b[addr:addr + 4] == b"SNOD"
        n = self.u(addr + 6, 2)
        p = addr + 8
        ent = []
        esize = 2 * self.size_off + 4 + 4 + 16
        for _ in range(n):
            name_off = self.u(p, self.size_off)
            oh = self.u(p + self.size_off, self.size_off)
            ent.append((name_off, oh))
            p += esize
        return ent

    def _is_group(self, oh_addr):
        for mtype, body, msize in self._messages(oh_addr):
            if mtype == 0x11:
                return True
        return False

    # ---- dataset ----
    def _read_dataset(self, oh_addr):
        dims = None; dtype = None; layout = None
        for mtype, body, msize in self._messages(oh_addr):
            if mtype == 0x01:   # dataspace
                dims = self._dataspace(body)
            elif mtype == 0x03: # datatype
                dtype = self._datatype(body)
            elif mtype == 0x08: # data layout
                layout = self._layout(body)
        if dims is None or dtype is None or layout is None:
            return None
        kind, a, b = layout
        if kind == "contiguous":
            raw = self.b[a:a + b]
        elif kind == "compact":
            raw = self.b[a:a + b]
        else:
            return None  # chunked not supported
        arr = np.frombuffer(raw, dtype=dtype)
        n = int(np.prod(dims)) if dims else 1
        arr = arr[:n].reshape(dims if dims else (1,))
        return arr

    def _dataspace(self, body):
        ver = self.b[body]
        rank = self.b[body + 1]
        if ver == 1:
            p = body + 8
        else:  # v2
            p = body + 4
        dims = [self.u(p + i * self.size_len, self.size_len) for i in range(rank)]
        return tuple(dims)

    def _datatype(self, body):
        cls = self.b[body] & 0x0F
        size = self.u(body + 4, 4)
        if cls == 1:  # float
            return np.dtype("<f4") if size == 4 else np.dtype("<f8")
        if cls == 0:  # fixed point int
            return np.dtype(f"<i{size}") if size in (1,2,4,8) else np.dtype("<f4")
        return np.dtype("<f4")

    def _layout(self, body):
        ver = self.b[body]
        if ver == 3:
            cls = self.b[body + 1]
            if cls == 0:  # compact
                size = self.u(body + 2, 2)
                return ("compact", body + 4, size)
            if cls == 1:  # contiguous
                addr = self.u(body + 2, self.size_off)
                size = self.u(body + 2 + self.size_off, self.size_len)
                return ("contiguous", addr, size)
            return ("chunked", None, None)
        if ver in (1, 2):
            dim = self.b[body + 1]
            cls = self.b[body + 2]
            p = body + 8
            if cls == 1:  # contiguous: address then sizes
                addr = self.u(p, self.size_off)
                return ("contiguous", addr, None)  # size inferred by caller? fallback
            return ("chunked", None, None)
        return ("chunked", None, None)


def read_weights(path):
    """Return {dataset_path: ndarray} for a Keras old-style .weights.h5."""
    return _Reader(path).datasets
