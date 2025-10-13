import pytest
import pandas as pd

from nh3pred.utils import complete_incorp

def test_1():
    # no incorp column in df
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15]}
    )

    df_tmp = complete_incorp (df)
    
    for x in ["incorp", "t_incorp"]: assert x in df_tmp.columns 
    
    assert df["incorp"].unique() == "none"
    assert df["t_incorp"].unique() == 1000

def test_2():
    # df with incorp column but no t_incorp column
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"]}
    )

    df_tmp = complete_incorp (df)

    assert "t_incorp" in df_tmp.columns
    assert (df_tmp.loc [df_tmp ['incorp'] == "none", 't_incorp'] == 1000).all()
    assert (df_tmp.loc [df_tmp ['incorp'] != "none", 't_incorp'] == 0).all()

def test_3():
    # df with both incorp and t_incorp columns
    df = pd.DataFrame (
        {"pmid": [11, 11, 42, 42, 42, 42, 42],
         "ct":[2, 5, 1, 8, 11, 19, 22],
         "air_temp" : [24, 31, 9, 20, 5, 8, 15],
         "incorp": ["none", "none", "deep", "deep", "deep", "deep", "deep"],
         "t_incorp": [float ("nan"), float ("nan"), 10, 10, 10, 10, 10]}
    )

    df_tmp = complete_incorp (df)
    assert (df_tmp.loc [df_tmp ['incorp'] == "none", 't_incorp'] == 1000).all()

