# nh3pred

This package provides a single function, `predict`, which estimates ammonia emissions following field fertilization under given environmental conditions.  
The underlying model is a recurrent neural network described in [ref] under the name *"rnn 9 – data a."*.  
The `predict` function works similarly to the `alfam2` function from the R package [ALFAM2](https://cran.r-project.org/web/packages/ALFAM2/index.html).

## Install

```bash
pip install nh3pred 
```

## Documentation

A complete documentation for the `predict` function is available [here](https://nh3pred.readthedocs.io/en/latest/api.html#nh3pred.api.predict).

## Usage

You can use the package in Python as follows:

```python
import pandas as pd
from nh3pred import predict

df = pd.DataFrame ({
    "pmid": [1, 1, 1, 1, 1, 1],
    "ct": [3, 6, 10, 24, 48, 72],
    "tan_app": [42, 42, 42, 42, 42, 42],
    "air_temp": [18, 23, 24, 15, 21, 20],
    "wind_2m": [2, 2, 1, 1, 2, 2],
    "rain_rate": 0,
    "app_rate": [20, 20, 20, 20, 20, 20],
    "man_dm": [8.3, 8.3, 8.3, 8.3, 8.3, 8.3],
    "man_ph": [7.1, 7.1, 7.1, 7.1, 7.1, 7.1],
    "app_mthd": ["ts", "ts", "ts", "ts", "ts", "ts"],
    "man_source": ["cat", "cat", "cat", "cat", "cat", "cat"]
})

pred = predict(df)
print(pred)
``` 


## Notes

- The trained weights are included in the package under `ammonia_predict/data/final_model.pth`.
- The package requires **Python ≥3.12**, **PyTorch ≥2.5.0**, and **pandas ≥2.2.3**.



