import pytest
import pandas as pd 

from nh3pred.utils import check_unique_values_for_plot_level_variables 

def test_1(): 
    # df with unique values for plot level variables
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "app_mthd": ["bc", "bc", "os", "os", "os", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat", "cat", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"],
         "man_ph": [1, 1, 3, 3, 3, 3, 3]}
    )

    check_unique_values_for_plot_level_variables (df)

def test_2():
    # df with multiple values for pH in pmid 42
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "app_mthd": ["bc", "bc", "os", "os", "os", "os", "os"],
         "man_source": ["pig", "pig", "cat", "cat", "cat", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"],
         "man_ph": [1, 1, 3, 5, 3, 3, 3]}
    )

    
    with pytest.raises(ValueError) as excinfo:
        check_unique_values_for_plot_level_variables (df)

    # Check the exact error message
    assert "Not unique value for man_ph in pmid 42" in str(excinfo.value)
