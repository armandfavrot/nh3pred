import pytest
import pandas as pd

from nh3pred.utils import check_correct_values_for_categorical_variables

def test_1():
    # with correct values 
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "app_mthd": ["bc", "bc", "os", "os", "os", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat", "cat", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"],
         "man_ph": [1, 1, 3, 5, 3, 3, 3]}
    )

    check_correct_values_for_categorical_variables (df)

def test_2():
    # with wrong values
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "app_mthd": ["bc", "bc", "os", "os", "os", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat", "cat", "rat", "cap"],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"],
         "man_ph": [1, 1, 3, 5, 3, 3, 3]}
    )

    with pytest.raises (ValueError) as excinfo:
        check_correct_values_for_categorical_variables (df)

    assert "The following values are not allowed in man_source" in str (excinfo)
    assert "rat" in str (excinfo)
    assert "cap" in str (excinfo)
