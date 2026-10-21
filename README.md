# nh3pred

Predict cumulative ammonia emissions after field fertilizer application using a recurrent neural network coupled with a physically constrained TAN reservoir. A detailed description of the model is available in the Model and interpretation vignette (see Documentation).

## Installation

```bash
pip install nh3pred==0.2.0
```

Version 0.2.0 changes the model and input interface. See the migration guide
below before upgrading from 0.1.3. Python >=3.12.3 is currently declared;
runtime dependencies are NumPy, pandas and PyTorch.

## Quick start

```python
import pandas as pd
from nh3pred import predict

pdat = pd.DataFrame({"id": [1], "tan_app": [70.0], "app_method": ["bc"]})
result = predict(pdat)
print(result[["id", "ct", "prediction_ecum", "prediction_erel"]])
```

Each row of `pdat` describes one application. Required columns are `id`,
`tan_app` (kg N/ha) and `app_method`. Optional `idat` supplies `id`, `ct`
(hours since application), `air_temp`, `wind_2m` and `rain_rate`.
Times must be positive and strictly increasing within each application.
If idat is omitted, `predict()` uses the default time and weather profiles bundled with the package, covering the first 240 h after application.

```python
idat = pd.DataFrame({
    "id": [1, 1, 1], 
    "ct": [2.0, 12.0, 48.0],
    "air_temp": [15.0, 18.0, 12.0], 
    "wind_2m": [2.0, 3.0, 1.0],
    "rain_rate": [0.0, 0.0, 0.0],
})
result = predict(pdat, idat, batch_size=128)
```

`prediction_ecum` is cumulative emission in kg N-NH3/ha; `prediction_erel` is its fraction of applied TAN. 

## Migration from 0.1.3

The former `predict(df)` input combined application and time rows. Split it
into one application row per ID (`pdat`) and a weather/time table (`idat`).
Rename `pmid -> id`, `app_mthd -> app_method`, `man_source -> fer_origin`,
`man_dm -> fer_dm`, `man_ph -> fer_ph`, and `t_incorp -> time_incorp`.
The new model, defaults and accepted categories differ: reproducing 0.1.3
results requires installing `nh3pred==0.1.3`.

## Documentation

The documentation includes API reference material, a migration guide, and four vignettes illustrating the main features of the package:

- **Getting started** provides simple examples showing how to prepare inputs and generate predictions.
- **Input reference** describes the model inputs and shows their distributions in the training data.
- **Model and interpretation** describes the model architecture, including the physically constrained TAN reservoir, and provides guidance for interpreting its predictions.
- **Scenario comparison** illustrates how the package can be used to compare ammonia emission predictions across contrasting application scenarios.

The documentation also provides the executable Quarto sources used to generate the vignettes.

