import pytest
import pandas as pd

from nh3pred.utils import check_t_incorp_values

def test_1():
    # df with t_incorp values different to NaN when incorp = none
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep"],
         "t_incorp" : [1, 1, 0, 0],
         "man_ph": [1, 1, 3, 3]}
    )

    with pytest.raises (ValueError) as excinfo:
        check_t_incorp_values (df)

    assert "t_incorp should be set to NaN when incorp = none." in str (excinfo)


def test_2():
    # df with t_incorp values equal to NaN when incorp != none
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep"],
         "t_incorp" : [float ("nan"), float ("nan"), 0, float ("nan")],
         "man_ph": [1, 1, 3, 3]}
    )

    with pytest.raises (ValueError) as excinfo:
        check_t_incorp_values (df)

    assert "NaN values in t_incorp for incorp != none." in str (excinfo)

def test_3():
    # df with correct values of t_incorp
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep"],
         "t_incorp" : [float ("nan"), float ("nan"), 0, 0],
         "man_ph": [1, 1, 3, 3]}
    )

    check_t_incorp_values (df)

