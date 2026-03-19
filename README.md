# pepc-global-track

Predict tropical cyclone track displacements (longitude or latitude) using pre-trained Random Forest models.

## Installation

```bash
pip install pepc-global-track
```

## Usage

```python
import numpy as np
from pepc_global_track import predict_track

# (N, 5) array with columns [u250, v250, u850, v850, lat]
# u250, v250, u850, v850: wind predictors (m s^−1)
# lat: storm latitude (degrees)
X = np.array([
    [5.0, 1.0, 3.0, -1.0, 15.0],
    [6.0, 1.5, 3.5, -0.5, 16.0],
    [7.0, 2.0, 4.0,  0.0, 17.0],
])

# Predict longitude displacement
delta_lon = predict_track("WNP", "lon", X)

# Predict latitude displacement
delta_lat = predict_track("WNP", "lat", X)
```

## Parameters

- **basin**: `str` — one of `"AS"`, `"BoB"`, `"WNP"`, `"ENP"`, `"NA"`, `"SI"`, `"SP"`
- **type**: `str` — `"lon"` (zonal displacement) or `"lat"` (meridional displacement)
- **X**: `numpy.ndarray` — 2D array of shape `(N, 5)` with columns `[u250, v250, u850, v850, lat]`

## Returns

- `numpy.ndarray` — 1D array of predicted displacements (no noise added)

## Model Weights

Model weights are automatically downloaded from [HuggingFace](https://huggingface.co/CONGG/pepc-global-track) on first use and cached locally.
