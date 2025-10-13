import pytest 
import pandas as pd

from nh3pred.utils import check_needed_cols

def test_check_needed_cols_missing_columns ():
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air.temp" : [24, 31, 9, 20, 5, 8, 15]}
    )

    # Check that the function raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        check_needed_cols (df)

    # Check the exact error message
    assert "The following column(s) is(are) missing: ['tan_app']" in str(excinfo.value)


def test_check_needed_cols_valid_columns():
    # DataFrame containing only allowed columns
    df = pd.DataFrame({
        "pmid": [1, 2],
        "ct": [3, 4],
        "tan_app": [10, 15],
        "man_source": ["cat", "pig"]
    })
    
    # Should execute without raising any error
    check_needed_cols(df)
