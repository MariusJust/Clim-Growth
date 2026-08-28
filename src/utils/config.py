"""Flatten the grouped ``instance`` config into a single namespace.

The config file groups estimation keys under ``instance.optim``,
``instance.model`` and ``instance.server`` for readability. The pipeline,
however, consumes a flat ``instance`` namespace (``inst.lr``,
``inst.formulation``, ...) and the model unpacks ``dict(cfg).items()`` directly.

``flatten_instance`` merges the optim/model/server sub-blocks up to the top
level, leaving ``bootstrap`` and ``nic`` (and any already-flat keys) untouched.
It is idempotent: a config that is already flat is returned unchanged.

Note: Hydra command-line overrides still target the *nested* paths, e.g.
``instance.model.formulation=income`` or ``instance.optim.lr=1e-4``. Flattening
happens in code, after Hydra has resolved the config.
"""

from omegaconf import OmegaConf

_GROUPS = ("optim", "model", "server")


def flatten_instance(inst):
    """Return a flat ``instance`` DictConfig from the grouped config."""
    if inst is None or not any(g in inst for g in _GROUPS):
        return inst
    flat = OmegaConf.create({k: v for k, v in inst.items() if k not in _GROUPS})
    for g in _GROUPS:
        sub = inst.get(g)
        if sub is not None:
            flat = OmegaConf.merge(flat, sub)
    return flat
