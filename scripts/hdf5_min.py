"""read_weights(path) -> {dataset_path: ndarray}

Reads a Keras ``.weights.h5`` into a flat dict keyed by full dataset path,
e.g. ``"layers/dense/vars/0"`` (kernel) and ``"layers/dense/vars/1"`` (bias).

Uses h5py when available (it ships with TensorFlow, so: devcontainer, server).
Falls back to the bundled pure-Python HDF5 reader in ``_hdf5_pure.py`` so the
figure scripts also run in environments without h5py.
"""
from __future__ import annotations

import numpy as np


def _read_h5py(path):
    import h5py
    out = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = np.asarray(obj)

    with h5py.File(str(path), "r") as f:
        f.visititems(visit)
    return out


def read_weights(path):
    """Return {dataset_path: ndarray} for a Keras old-style .weights.h5."""
    try:
        return _read_h5py(path)
    except ImportError:
        from _hdf5_pure import read_weights as _pure   # no h5py available
        return _pure(str(path))
