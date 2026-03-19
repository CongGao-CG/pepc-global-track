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
    u250: np.ndarray,
    v250: np.ndarray,
    u850: np.ndarray,
    v850: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Predict tropical cyclone track displacement.

    Parameters
    ----------
    basin : str
        Basin name: "AS", "BoB", "WNP", "ENP", "NA", "SI", or "SP".
    type : str
        "lon" for zonal displacement or "lat" for meridional displacement.
    u250, v250, u850, v850 : np.ndarray
        1D arrays of wind predictors (m/s).
    lat : np.ndarray
        1D array of storm latitudes (degrees).

    Returns
    -------
    np.ndarray
        1D array of predicted displacements (no noise).
    """
    if basin not in VALID_BASINS:
        raise ValueError(f"basin must be one of {VALID_BASINS}, got {basin!r}")
    if type not in VALID_TYPES:
        raise ValueError(f"type must be one of {VALID_TYPES}, got {type!r}")

    arrays = [np.asarray(a, dtype=float).ravel() for a in [u250, v250, u850, v850, lat]]
    n = arrays[0].size
    for i, a in enumerate(arrays):
        if a.size != n:
            raise ValueError(
                f"All input arrays must have the same length; got {n} and {a.size}"
            )

    X = np.column_stack(arrays)
    model = _load_model(basin, type)
    return model.predict(X)
