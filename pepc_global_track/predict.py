from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "CONGG/pepc-global-track"
VALID_BASINS = ("AS", "BoB", "WNP", "ENP", "NA", "SI", "SP")
VALID_TYPES = ("lon", "lat")


@lru_cache(maxsize=None)
def _load_model(basin: str, type_: str):
    filename = f"track_{type_}_RF-lat_{basin}.joblib"
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    return joblib.load(path)


def predict_track(
    basin: str,
    type: str,
    X: np.ndarray,
) -> np.ndarray:
    """Predict tropical cyclone track displacement.

    Parameters
    ----------
    basin : str
        Basin name: "AS", "BoB", "WNP", "ENP", "NA", "SI", or "SP".
    type : str
        "lon" for zonal displacement or "lat" for meridional displacement.
    X : np.ndarray
        2D array of shape (N, 5) with columns [u250, v250, u850, v850, lat].
        u250, v250, u850, v850 are wind predictors (m s^−1).
        lat is storm latitude (degrees).

    Returns
    -------
    np.ndarray
        1D array of predicted displacements (no noise).
    """
    if basin not in VALID_BASINS:
        raise ValueError(f"basin must be one of {VALID_BASINS}, got {basin!r}")
    if type not in VALID_TYPES:
        raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError(f"X must have shape (N, 5), got {X.shape}")

    model = _load_model(basin, type)
    return model.predict(X)
