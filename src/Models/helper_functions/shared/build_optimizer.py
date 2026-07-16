"""Build a Keras optimizer selected by name.

Used by the global and regional model factories:

    optimizer = build_optimizer(getattr(self, "optimizer", "adam"), lr, cfg=self)

``name`` selects the optimizer, ``lr`` is the initial learning rate, and ``cfg``
(the model / config object) may carry optional optimizer hyperparameters, which
are applied only when present and only to optimizers that accept them.
"""

import inspect

from tensorflow.keras import optimizers as _opt


_OPTIMIZERS = {
    "adam": _opt.Adam,
    "nadam": _opt.Nadam,
    "rmsprop": _opt.RMSprop,
    "sgd": _opt.SGD,
}
# AdamW is only available in newer Keras; register it when present.
try:
    _OPTIMIZERS["adamw"] = _opt.AdamW
except AttributeError:  # pragma: no cover
    pass

_OPTIONAL = ("beta_1", "beta_2", "epsilon", "weight_decay",
             "momentum", "clipnorm", "clipvalue")


def build_optimizer(name="adam", lr=1e-3, cfg=None):
    """Return a configured Keras optimizer.

    Parameters
    ----------
    name : str
        Optimizer key, one of ``adam``, ``nadam``, ``rmsprop``, ``sgd``
        (and ``adamw`` on newer Keras). Case-insensitive; ``None`` -> ``adam``.
    lr : float
        Initial learning rate.
    cfg : object, optional
        Config/model object; any of ``beta_1, beta_2, epsilon, weight_decay,
        momentum, clipnorm, clipvalue`` found on it are forwarded to optimizers
        that accept them.
    """
    key = str(name or "adam").lower()
    if key not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}."
        )
    opt_cls = _OPTIMIZERS[key]

    kwargs = {"learning_rate": lr}
    if cfg is not None:
        for attr in _OPTIONAL:
            val = getattr(cfg, attr, None)
            if val is not None:
                kwargs[attr] = val

    # Keep only kwargs the chosen optimizer actually accepts (learning_rate always kept).
    try:
        allowed = set(inspect.signature(opt_cls).parameters)
        kwargs = {k: v for k, v in kwargs.items() if k == "learning_rate" or k in allowed}
    except (TypeError, ValueError):  # pragma: no cover
        kwargs = {"learning_rate": lr}

    return opt_cls(**kwargs)
