import pandas as pd

from nh3pred.utils import add_dt

def test_1():

    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air.temp" : [24, 31, 9, 20, 5, 8, 15]}
    )
    
    df_tmp = add_dt (df)
    
    assert "dt" in df_tmp.columns
    
    pmids = df_tmp['pmid'].unique()
    
    for x in pmids:
        
        tmp = df_tmp [df_tmp ['pmid'] == x]
        assert (tmp.iloc [0]['ct'] == tmp.iloc[0]['dt'])
        
        for h in range (len (tmp) - 1):
            assert ((tmp.iloc [h + 1]["ct"] - tmp.iloc [h]["ct"]) == tmp.iloc [h + 1]["dt"])
