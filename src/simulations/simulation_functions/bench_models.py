"""Benchmark econometric models fit on real or simulated panel data.

Two specifications are supported:

- **Burke (2015)** quadratic:
  ``delta_logGDP ~ T + T^2 + P + P^2 + alpha_i + gamma_t
                   + theta_{1i}*t + theta_{2i}*t^2``
- **Leirvik (2024)** full-interaction (also called the "interactive" spec
  in this codebase): Burke quadratic plus ``T*P, T^2*P, T*P^2, T^2*P^2``.
  The paper chapter ``102_Leirvik.tex`` does not document an equation;
  this matches the Leirvik DGP in ``Simulate_data.py`` and the legacy
  ``interaction_model`` formula.

The high-level entry points :func:`fit_burke` and :func:`fit_leirvik` return
a :class:`FittedBenchmark` exposing the raw statsmodels result plus helpers
for R^2, AIC, BIC, and the climate-only surface.

Legacy entry points :func:`quadratic_model` and :func:`interaction_model`
are preserved (without country trends) for Monte Carlo callers in
``multiprocess_mc.py`` and ``monte_carlo.py``, whose synthetic DGPs do not
include per-country trends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper


_BURKE_CLIMATE_TERMS = "temp + temp_sq + precip + precip_sq"
_LEIRVIK_EXTRA_TERMS = (
    "temp:precip + temp_sq:precip + temp:precip_sq + temp_sq:precip_sq"
)


@dataclass
class FittedBenchmark:
    """A fitted Burke- or Leirvik-style benchmark model.

    Attributes
    ----------
    fit
        The underlying ``statsmodels`` regression result. Use this for
        residuals, full coefficient vector, predictions on training rows,
        clustered standard errors, etc.
    spec
        ``"burke"`` or ``"leirvik"``.
    has_country_trends, has_quadratic_trends
        Whether the fit included per-country linear / quadratic trends.
    """

    fit: RegressionResultsWrapper
    spec: str
    has_country_trends: bool
    has_quadratic_trends: bool

    @property
    def r_squared(self) -> float:
        return float(self.fit.rsquared)

    @property
    def aic(self) -> float:
        return float(self.fit.aic)

    @property
    def bic(self) -> float:
        return float(self.fit.bic)

    @property
    def n_obs(self) -> int:
        return int(self.fit.nobs)

    def predict_surface(self, T, P):
        """Evaluate the climate-only surface on a (T, P) grid.

        Returns the contribution of the climate terms alone; fixed effects
        and country trends are excluded so the result can be plotted as a
        marginal response surface. ``T`` and ``P`` broadcast naturally.
        """
        return _predict_climate_surface(self.fit.params, T, P, self.spec)


def fit_burke(
    data: pd.DataFrame,
    *,
    country_trends: bool = True,
    quadratic_trends: bool = True,
    temp_col: str = "temp",
    precip_col: str = "precip",
    growth_col: str = "growth",
    country_col: str = "ISO",
    year_col: str = "Year",
) -> FittedBenchmark:
    """Fit the Burke (2015) panel specification.

    Defaults match the real-data column names used in ``comparison.ipynb``.
    For Monte Carlo data with different column names, override the
    ``*_col`` arguments.
    """
    return _fit_benchmark(
        spec="burke",
        data=data,
        country_trends=country_trends,
        quadratic_trends=quadratic_trends,
        temp_col=temp_col,
        precip_col=precip_col,
        growth_col=growth_col,
        country_col=country_col,
        year_col=year_col,
    )


def fit_leirvik(
    data: pd.DataFrame,
    *,
    country_trends: bool = True,
    quadratic_trends: bool = True,
    temp_col: str = "temp",
    precip_col: str = "precip",
    growth_col: str = "growth",
    country_col: str = "ISO",
    year_col: str = "Year",
) -> FittedBenchmark:
    """Fit the Leirvik (2024) full-interaction panel specification."""
    return _fit_benchmark(
        spec="leirvik",
        data=data,
        country_trends=country_trends,
        quadratic_trends=quadratic_trends,
        temp_col=temp_col,
        precip_col=precip_col,
        growth_col=growth_col,
        country_col=country_col,
        year_col=year_col,
    )


def _fit_benchmark(
    *,
    spec: str,
    data: pd.DataFrame,
    country_trends: bool,
    quadratic_trends: bool,
    temp_col: str,
    precip_col: str,
    growth_col: str,
    country_col: str,
    year_col: str,
) -> FittedBenchmark:
    rename_map = {
        temp_col: "temp",
        precip_col: "precip",
        growth_col: "growth",
        country_col: "ISO",
        year_col: "Year",
    }
    rename_map = {src: dst for src, dst in rename_map.items() if src != dst}
    df = data.rename(columns=rename_map).copy()

    df["temp_sq"] = df["temp"] ** 2
    df["precip_sq"] = df["precip"] ** 2

    climate_terms = _BURKE_CLIMATE_TERMS
    if spec == "leirvik":
        climate_terms = f"{climate_terms} + {_LEIRVIK_EXTRA_TERMS}"

    trend_part = ""
    if country_trends:
        yi_vars = [c for c in df.columns if "_yi_" in c]
        y2_vars = [c for c in df.columns if "_y2_" in c]
        if yi_vars:
            trend_part = " + " + " + ".join(yi_vars)
            if quadratic_trends and y2_vars:
                trend_part += " + " + " + ".join(y2_vars)
        else:
            df["t_idx"] = df.groupby("ISO")["Year"].transform(
                lambda x: (x - x.min()).astype(float)
            )
            trend_part = " + C(ISO):t_idx"
            if quadratic_trends:
                df["t_idx2"] = df["t_idx"] ** 2
                trend_part += " + C(ISO):t_idx2"

    formula = f"growth ~ {climate_terms} + C(ISO) + C(Year){trend_part}"
    df_fit = df.dropna(subset=["growth", "temp", "precip"]).copy()
    fit = smf.ols(formula=formula, data=df_fit).fit()

    return FittedBenchmark(
        fit=fit,
        spec=spec,
        has_country_trends=country_trends,
        has_quadratic_trends=country_trends and quadratic_trends,
    )


def _predict_climate_surface(coefs, T, P, spec: str):
    base = (
        coefs["temp"] * T
        + coefs["precip"] * P
        + coefs["temp_sq"] * T ** 2
        + coefs["precip_sq"] * P ** 2
    )
    if spec == "burke":
        return base
    return base + (
        coefs["temp:precip"] * T * P
        + coefs["temp_sq:precip"] * T ** 2 * P
        + coefs["temp:precip_sq"] * T * P ** 2
        + coefs["temp_sq:precip_sq"] * T ** 2 * P ** 2
    )


# ---------------------------------------------------------------------------
# Legacy entry points: surface-only, no country trends.
# Retained for Monte Carlo callers whose DGPs lack per-country trends.
# ---------------------------------------------------------------------------


def quadratic_model(data, P, T):
    """Fit Burke without country trends; return the climate surface array.

    Backward-compatible shim for ``multiprocess_mc.py`` and
    ``monte_carlo.py``. New code should call :func:`fit_burke` and use
    ``FittedBenchmark.predict_surface``.
    """
    fitted = fit_burke(
        data,
        country_trends=False,
        temp_col="temperature",
        precip_col="precipitation",
        growth_col="delta_logGDP",
        country_col="CountryCode",
    )
    return fitted.predict_surface(T, P)


def interaction_model(data, P, T):
    """Fit Leirvik without country trends; return the climate surface array.

    Backward-compatible shim for ``multiprocess_mc.py`` and
    ``monte_carlo.py``. New code should call :func:`fit_leirvik` and use
    ``FittedBenchmark.predict_surface``.
    """
    fitted = fit_leirvik(
        data,
        country_trends=False,
        temp_col="temperature",
        precip_col="precipitation",
        growth_col="delta_logGDP",
        country_col="CountryCode",
    )
    return fitted.predict_surface(T, P)

