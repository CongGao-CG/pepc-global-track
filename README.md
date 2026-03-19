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

# 1D arrays of environmental predictors at storm locations
u250 = np.array([5.0, 6.0, 7.0])   # 250 hPa zonal wind (m s^−1)
v250 = np.array([1.0, 1.5, 2.0])   # 250 hPa meridional wind (m s^−1)
u850 = np.array([3.0, 3.5, 4.0])   # 850 hPa zonal wind (m s^−1)
v850 = np.array([-1.0, -0.5, 0.0]) # 850 hPa meridional wind (m s^−1)
lat  = np.array([15.0, 16.0, 17.0]) # storm latitude (degrees)

# Predict longitude displacement
delta_lon = predict_track("WNP", "lon", u250, v250, u850, v850, lat)

# Predict latitude displacement
delta_lat = predict_track("WNP", "lat", u250, v250, u850, v850, lat)
```

## Parameters

- **basin**: `str` — one of `"AS"`, `"BoB"`, `"WNP"`, `"ENP"`, `"NA"`, `"SI"`, `"SP"`
- **type**: `str` — `"lon"` (zonal displacement) or `"lat"` (meridional displacement)
- **u250, v250, u850, v850**: `numpy.ndarray` — 1D arrays of wind predictors
- **lat**: `numpy.ndarray` — 1D array of storm latitudes

## Returns

- `numpy.ndarray` — 1D array of predicted displacements (no noise added)

## Model Weights

Model weights are automatically downloaded from [HuggingFace](https://huggingface.co/CONGG/pepc-global-track) on first use and cached locally.
