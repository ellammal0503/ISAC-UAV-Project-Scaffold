# Datasets

## Used Datasets
- [3GPP TR 38.901 Channel Models](https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/)
- No external radar/ISAC datasets are currently used; only synthetic data is generated.

## Generated Datasets
- Synthetic UAV sensing dataset
  - Inputs: simulated RAI cubes from RDM + DoA processing
  - Labels: {range, velocity, DoA}
  - LOS/NLOS flag (optional)

## Data Format
- HDF5/NPZ files for efficient storage
- Each sample contains:
  - `X`: extracted features from RAI cubes or detections
  - `y`: corresponding labels (range, velocity, DoA, LOS/NLOS)

## Licensing
- Generated datasets released under **CC BY 4.0**
- corresponding labels (range, velocity, DoA, LOS/NLOS)
