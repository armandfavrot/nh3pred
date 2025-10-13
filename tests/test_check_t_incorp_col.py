import pytest
import pandas as pd

from nh3pred.utils import check_t_incorp_col

def test_1 ():
    # df with incorp column with some values different from 'none', but no t_incorp column
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep"],
         "man_ph": [1, 1, 3, 3]}
    )

    with pytest.raises (ValueError) as excinfo:
        check_t_incorp_col (df)

    assert "Column t_incorp is missing." in str (excinfo)

def test_2():
    # df with column t_incorp but no incorp column
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "t_incorp": 0,
         "man_ph": [1, 1, 3, 3]}
    )

    with pytest.raises (ValueError) as excinfo:
        check_t_incorp_col (df)
   
    assert "Column t_incorp is not allowed since there is no incorp column." in str (excinfo)

def test_3():
    # df with incorp and t_incorp 
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

    check_t_incorp_col (df) 

def test_4():
    # df with only incorp but all incorp values = none
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42],
         "ct":[2, 5, 1, 8],
         "air_temp" : [24, 31, 9, 20],
         "app_mthd": ["bc", "bc", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat"],
         "incorp": ["none", "none", "none", "none"],
         "t_incorp" : [float ("nan"), float ("nan"), 0, 0],
         "man_ph": [1, 1, 3, 3]}
    )

    check_t_incorp_col (df)
