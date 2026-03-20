# PepC-Global Track

A Python package for predicting tropical cyclone track displacements using pre-trained Random Forest models.

## Installation

```bash
pip install --upgrade pepc-global-track
```

Or from source:

```bash
git clone https://github.com/CongGao-CG/pepc-global-track.git
cd pepc-global-track
pip install .
```

## Quick Start

```python
import numpy as np
from pepc_global_track import predict_track

# Create predictor array with shape (N, 5)
# Columns: [u250, v250, u850, v850, lat]
# u250, v250: zonal and meridional wind at 250 hPa (m s^-1)
# u850, v850: zonal and meridional wind at 850 hPa (m s^-1)
# lat: storm latitude (degrees)
X = np.array([
    [5.0, 1.0, 3.0, -1.0, 15.0],
    [6.0, 1.5, 3.5, -0.5, 16.0],
    [7.0, 2.0, 4.0,  0.0, 17.0],
])

# Predict longitude displacement for a specific basin
basin = 'NA'  # North Atlantic
delta_lon = predict_track(basin, 'lon', X)

# Predict latitude displacement
delta_lat = predict_track(basin, 'lat', X)

print(f"Predicted longitude displacement: {delta_lon}")
print(f"Predicted latitude displacement: {delta_lat}")
```

## Available Basins

| Basin Code | Name                  | Latitude Range    | Longitude Range     |
|------------|-----------------------|-------------------|---------------------|
| `AS`       | Arabian Sea           | 5° to 22.5°N      | 50° to 77.5°E       |
| `BoB`      | Bay of Bengal         | 5° to 22.5°N      | 80° to 100°E        |
| `WNP`      | Western North Pacific | 5° to 30°N        | 102.5°E to 180°     |
| `ENP`      | Eastern North Pacific | 5° to 25°N        | 177.5° to 75°W      |
| `NA`       | North Atlantic        | 5° to 30°N        | 97.5° to 2.5°W      |
| `SI`       | South Indian          | 30° to 5°S        | 20° to 145°E        |
| `SP`       | South Pacific         | 30° to 5°S        | 147.5°E to 100°W    |

## API Reference

### `predict_track(basin, disp_type, X)`

Predict tropical cyclone track displacement (longitude or latitude).

**Parameters:**
- `basin` (str): Basin name (one of: 'AS', 'BoB', 'WNP', 'ENP', 'NA', 'SI', 'SP')
- `disp_type` (str): Displacement type — 'lon' for zonal (longitude) displacement or 'lat' for meridional (latitude) displacement
- `X` (np.ndarray): 2D array of shape (N, 5) with columns [u250, v250, u850, v850, lat]

**Returns:**
- `np.ndarray`: 1D array of predicted displacements (degrees)

**Raises:**
- `ValueError`: If basin is invalid, disp_type is not 'lon' or 'lat', or X does not have shape (N, 5)

### `get_basin_names()`

Get the list of valid basin names.

**Returns:**
- `list[str]`: List of 7 basin names

### `BASINS`

List of valid basin names.

## Input Variables

| Variable | Description                          | Typical Units |
|----------|--------------------------------------|---------------|
| u250     | Zonal wind at 250 hPa                | m s^−1        |
| v250     | Meridional wind at 250 hPa           | m s^−1        |
| u850     | Zonal wind at 850 hPa                | m s^−1        |
| v850     | Meridional wind at 850 hPa           | m s^−1        |
| lat      | Storm latitude                       | degrees       |

## Model Details

This package contains pre-trained Random Forest models for 7 tropical cyclone basins. The models were pre-trained on ERA5 monthly data using five environmental predictors known to influence tropical cyclone track:

1. **Zonal wind at 250 hPa (u250)**: Upper-level zonal flow
2. **Meridional wind at 250 hPa (v250)**: Upper-level meridional flow
3. **Zonal wind at 850 hPa (u850)**: Lower-level zonal flow
4. **Meridional wind at 850 hPa (v850)**: Lower-level meridional flow
5. **Storm latitude (lat)**: Current storm position

Each basin has its own pre-trained models for both longitude and latitude displacement predictions.

## License

MIT License

## Citation

If you use this package in your research, please cite:

Gao, Cong, Ning Lin. "PepC-Global: A Basin-Tuned Probabilistic Tropical Cyclone Model with Enhanced Out-of-Sample Skill and Climate-Sensitive Over-Land Decay". Journal of Advances in Modeling Earth Systems (JAMES), under review.
