import pandas as pd
import torch

from nh3pred.utils import generate_tensors_predictors

def test_generate_tensors_predictors_shapes_and_types():
    df = pd.DataFrame({
        "pmid": [1, 1, 2],
        "ct": [1, 2, 1],
        "dt": [1, 1, 1],
        "air_temp": [10, 12, 14],
        "wind_2m": [3, 4, 2],
        "rain_rate": [0, 1, 0],
        "tan_app": [30, 30, 40],
        "app_rate": [10, 10, 20],
        "man_dm": [0.1, 0.2, 0.3],
        "man_ph": [7, 7, 8],
        "t_incorp": [0, 0, 1000],
        "app_mthd": [1, 2, 3],
        "incorp": [0, 1, 2],
        "man_source": [0, 0, 1],
    })
    
    device = "cpu"
    out = generate_tensors_predictors(df, pmid = 1, device=device)
    
    assert isinstance(out, list)
    assert len(out) == 2

    x_cont, x_cat = out

    assert x_cont.shape == (2, 10)      
    assert len(x_cat) == 3              
    for tensor in x_cat:
        assert tensor.shape == (2,)    

    assert device in str (x_cont.device)
    assert all(device in str (t.device) for t in x_cat)

    assert x_cont.dtype == torch.float32
    assert all(t.dtype == torch.int64 for t in x_cat)
