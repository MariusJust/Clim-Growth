"""Minimal NetCDF-3 (classic / 64-bit offset) reader - pure numpy, no netCDF4.

The CRU TS and GPW rasters used by this project are NetCDF-3 ("CDF\\x01"/"CDF\\x02"),
not HDF5, so h5py/netCDF4 are not required. This parses the header and gives
random access to individual time slices of a record variable, which matters
because the CRU files are ~6 GB and we only ever need a few months at a time.

Usage:
    nc = NC3("cru_ts4.08.1901.2023.tmp.dat.nc")
    nc.dims          -> {'lon': 720, 'lat': 360, 'time': 1476}
    nc.vars.keys()   -> variable names
    arr = nc.read_record("tmp", 123)      # one time slice, shape (lat, lon)
"""
import numpy as np

_TYPE = {1: ("i1", 1), 2: ("S1", 1), 3: (">i2", 2), 4: (">i4", 4),
         5: (">f4", 4), 6: (">f8", 8)}


class NC3:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        magic = self.f.read(4)
        if magic[:3] != b"CDF":
            raise ValueError(f"not NetCDF-3: {magic!r}")
        self.version = magic[3]                      # 1 = classic, 2 = 64-bit offset
        self.numrecs = self._i4()
        self.dims = {}
        self._dim_order = []
        self._read_dims()
        self.gattrs = self._read_atts()
        self.vars = {}
        self._read_vars()

    # ---- primitive readers ------------------------------------------------
    def _i4(self):
        return int(np.frombuffer(self.f.read(4), ">i4")[0])

    def _i8(self):
        return int(np.frombuffer(self.f.read(8), ">i8")[0])

    def _string(self):
        n = self._i4()
        s = self.f.read(n)
        pad = (4 - n % 4) % 4
        self.f.read(pad)
        return s.decode("utf-8", "replace")

    def _values(self, nc_type, nelems):
        dt, size = _TYPE[nc_type]
        raw = self.f.read(size * nelems)
        pad = (4 - (size * nelems) % 4) % 4
        self.f.read(pad)
        if nc_type == 2:
            return raw.decode("utf-8", "replace")
        return np.frombuffer(raw, dt)

    # ---- header sections --------------------------------------------------
    def _read_dims(self):
        tag = self._i4(); n = self._i4()
        if tag == 0:
            return
        for _ in range(n):
            name = self._string(); size = self._i4()
            self.dims[name] = size
            self._dim_order.append(name)

    def _read_atts(self):
        tag = self._i4(); n = self._i4()
        atts = {}
        if tag == 0:
            return atts
        for _ in range(n):
            name = self._string(); t = self._i4(); nel = self._i4()
            atts[name] = self._values(t, nel)
        return atts

    def _read_vars(self):
        tag = self._i4(); n = self._i4()
        if tag == 0:
            return
        for _ in range(n):
            name = self._string()
            ndims = self._i4()
            dimids = [self._i4() for _ in range(ndims)]
            atts = self._read_atts()
            nc_type = self._i4()
            vsize = self._i4()
            begin = self._i8() if self.version == 2 else self._i4()
            dnames = [self._dim_order[d] for d in dimids]
            shape = [self.dims[n] for n in dnames]
            self.vars[name] = dict(dimids=dimids, shape=shape, type=nc_type,
                                   vsize=vsize, begin=begin, atts=atts,
                                   dnames=dnames)
        # The record (unlimited) dimension is declared with size 0 in the header;
        # its true length is numrecs. Record variables are interleaved: each
        # record holds one slab of every record variable, so the stride between
        # consecutive time slices is the SUM of their vsizes.
        self.rec_dim = next((n for n in self._dim_order if self.dims[n] == 0), None)
        if self.rec_dim is not None:
            self.dims[self.rec_dim] = self.numrecs
            for v in self.vars.values():
                v["shape"] = [self.dims[n] for n in v["dnames"]]
        self.recsize = sum(v["vsize"] for v in self.vars.values()
                           if v["dnames"] and v["dnames"][0] == self.rec_dim)

    # ---- data access ------------------------------------------------------
    def read_record(self, name, index):
        """Read one slice along the record (unlimited) dimension."""
        v = self.vars[name]
        dt, size = _TYPE[v["type"]]
        inner = v["shape"][1:]
        count = int(np.prod(inner))
        off = v["begin"] + index * self.recsize
        self.f.seek(off)
        a = np.frombuffer(self.f.read(count * size), dt, count=count)
        return a.reshape(inner)

    def read_all(self, name):
        """Read a NON-record variable in full (e.g. lon, lat)."""
        v = self.vars[name]
        dt, size = _TYPE[v["type"]]
        count = int(np.prod(v["shape"])) if v["shape"] else 1
        self.f.seek(v["begin"])
        a = np.frombuffer(self.f.read(count * size), dt, count=count)
        return a.reshape(v["shape"]) if v["shape"] else a[0]

    def fill_value(self, name):
        a = self.vars[name]["atts"]
        for k in ("_FillValue", "missing_value"):
            if k in a:
                x = a[k]
                return float(np.asarray(x).ravel()[0])
        return None

    def close(self):
        self.f.close()
