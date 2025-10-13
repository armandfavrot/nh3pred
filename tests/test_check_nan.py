import pytest
import pandas as pd

from nh3pred.utils import check_nan

def test_check_nan_with_nan_air_temp():
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "t_incorp" : [24, 31, 9, 20, 5, 8, 15],
         "air_temp" : float("nan")}
    )

    with pytest.raises(ValueError) as excinfo:
        check_nan (df)

    assert "NaN values in air_temp" in str (excinfo.value)

def test_check_nan_no_nans():
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15]}
    )
    check_nan (df)
