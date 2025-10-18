import pandas as pd
from nh3pred import predict

def test_predict_columns():
    df = pd.DataFrame({
        "pmid": [1, 1, 2, 2],
        "ct": [2, 4, 2, 4],
        "tan_app": [36.7, 36.7, 36.7, 36.7],
        "air_temp": [12, 15, 11, 10],
        "wind_2m": [3, 3, 4, 2],
        "rain_rate": [0, 0, 1, 0],
        "app_rate": [10, 10, 12, 12],
        "man_dm": [0.1, 0.1, 0.1, 0.1],
        "man_ph": [7, 7, 7, 7],
    })

    out = predict(df)
    assert isinstance(out, pd.DataFrame)
    assert "prediction_ecum" in out.columns
    assert len(out) == len(df)
    assert not out["prediction_ecum"].isna().any()
    assert out["app_mthd"].isin(["bc", "bsth", "os", "ts", "cs"]).all()
