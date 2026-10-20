"""Reference outputs captured from scripts/package before repackaging."""

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nh3pred import predict
from nh3pred.prediction import _load_model, _predict_emissions
from nh3pred.preprocessing import _prepare_data

REFERENCE_PATH = Path(
    os.environ.get(
        "NH3PRED_REFERENCE_FILE",
        Path(__file__).parent / "fixtures/reference_predictions.json",
    )
)
CASES = json.loads(REFERENCE_PATH.read_text())


def decode(value):
    if value is None:
        return None
    data = [[np.nan if item is None else item for item in row] for row in value["data"]]
    return pd.DataFrame(data, columns=value["columns"], index=value["index"])


def test_bundled_model_artifacts_match_manifest():
    data = files("nh3pred").joinpath("data")
    manifest = json.loads((data / "model_manifest.json").read_text())
    for name, expected_hash in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected_hash


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_migration_matches_original_package(case):
    pdat, idat = decode(case["pdat"]), decode(case["idat"])
    actual = predict(pdat, idat)
    pd.testing.assert_frame_equal(
        actual, decode(case["expected"]), check_dtype=False, atol=1e-10, rtol=0
    )
    prepared, original = _prepare_data(pdat, idat)
    model, cont, cat, norm, device, response = _load_model()
    raw = _predict_emissions(
        prepared,
        original,
        model=model,
        cont_vars=cont,
        cat_vars=cat,
        normalisation_values=norm,
        device=device,
        response=response,
        batch_size=128,
    )
    np.testing.assert_allclose(
        raw.prediction_ecum, case["raw_ecum"], rtol=1e-6, atol=1e-7
    )
    for _, group in raw.groupby("id", sort=False):
        values = group.prediction_ecum.to_numpy()
        assert np.isfinite(values).all()
        assert (np.diff(np.r_[0.0, values]) >= -1e-7).all()
        assert (values <= group.tan_app.to_numpy() + 1e-6).all()


@pytest.fixture(scope="session", autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
