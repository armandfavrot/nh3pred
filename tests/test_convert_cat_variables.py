import pandas as pd

from nh3pred.utils import convert_cat_variables

def test_1():
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "app_mthd": ["bc", "bsth", "os", "bc", "ts", "cs", "cs"],
         "man_source": ["pig", "pig", "cat", "cat", "cat", "cat", "cat"],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"]}
    )
    
    df_tmp = convert_cat_variables (df)
    
    assert df_tmp.iloc[3]["app_mthd"] == 0
