# nh3pred

Predict cumulative ammonia emissions after field fertilizer application using
an embedded physical reservoir recurrent model.

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
If omitted, weather and time profiles use the bundled defaults, through 240 h.

```python
idat = pd.DataFrame({
    "id": [1, 1, 1], "ct": [2.0, 12.0, 48.0],
    "air_temp": [15.0, 18.0, 12.0], "wind_2m": [2.0, 3.0, 1.0],
    "rain_rate": [0.0, 0.0, 0.0],
})
result = predict(pdat, idat, batch_size=128)
```

`prediction_ecum` is cumulative emission in kg N-NH3/ha;
`prediction_erel` is its fraction of applied TAN. Public numeric outputs
are rounded to two decimals. Calculations use full precision internally.
The decoder produces nondecreasing emissions bounded by applied TAN for
valid positive time intervals and nonnegative TAN.

## Migration from 0.1.3

The former `predict(df)` input combined application and time rows. Split it
into one application row per ID (`pdat`) and a weather/time table (`idat`).
Rename `pmid -> id`, `app_mthd -> app_method`, `man_source -> fer_origin`,
`man_dm -> fer_dm`, `man_ph -> fer_ph`, and `t_incorp -> time_incorp`.
The new model, defaults and accepted categories differ: reproducing 0.1.3
results requires installing `nh3pred==0.1.3`.

## Documentation and model

The documentation includes API help, a migration guide and the rendered
vignettes and their executable Quarto sources. Model weights and all inference parameters are bundled;
no training dataset or network download is needed for prediction.
The public interface represents micrometeorological applications.
See `src/nh3pred/data/model_manifest.json` for model identity and checksums.

## Development

```bash
pip install -e '.[dev,build,docs]'
python -m pytest
python -m build
python -m twine check dist/*
python docs/render_vignettes.py  # requires Quarto and pdflatex
python -m sphinx -W -b html docs/source docs/_build/html
```

The wheel contains only inference code and model resources. Tests and
documentation are included in the source distribution. Training and exploratory
analysis remain in the research repository.

The vignette renderer uses the current Python interpreter and a temporary
Jupyter kernel. It executes synthetic prediction examples with the installed
package and writes four self-contained HTML files and one combined PDF to
`docs/source/_static/vignettes/`. Interpretation graphics are precomputed;
their source research data are not included or needed for rendering.



