import pandas as pd

from nh3pred.utils import complete_other

def test_1():
    # df without app_rate
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15]}
    )

    df_tmp = complete_other (df)
    
    assert ('app_rate' in df_tmp.columns)
    assert (df_tmp ['app_rate'].unique() == 29.38)
