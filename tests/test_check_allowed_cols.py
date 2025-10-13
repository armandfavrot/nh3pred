import pytest
import pandas as pd

from nh3pred.utils import check_allowed_cols  

def test_check_allowed_cols_invalid_column():
    # DataFrame containing a non-allowed column
    df = pd.DataFrame({
        "pmid": [11, 11, 42],
        "ct": [2, 5, 1],
        "power": [3.9, 3.9, 3.9],  # non-allowed column
        "air_temp": [24, 31, 9]
    })

    # Check that the function raises a ValueError
    with pytest.raises(ValueError) as excinfo:
        check_allowed_cols(df)

    # Check the exact error message
    assert "The following columns are not allowed: ['power']" in str(excinfo.value)

def test_check_allowed_cols_valid_columns():
    # DataFrame containing only allowed columns
    df = pd.DataFrame({
        "pmid": [1, 2],
        "ct": [3, 4],
        "air_temp": [10, 15],
        "man_source": ["cat", "pig"]
    })
    
    # Should execute without raising any error
    check_allowed_cols(df)
