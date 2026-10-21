import numpy as np
import pandas as pd
import pytest

from nh3pred.prediction import (
    _load_model,
    _predict_emissions,
    _to_cumulative_emissions,
    predict,
)
from nh3pred.preprocessing import (
    _complete_idat,
    _complete_pdat,
    _prepare_data,
    _split_wind_predictors,
    default_values,
)


def _example_data():
    pdat = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "tan_app": [60, 80, 70],
            "app_method": ["bc", "os", "ts"],
            "soil_oc": [3.0, 3.5, 2.5],
        }
    )

    # IDs are deliberately interleaved to check that predictions are
    # assigned back to the correct rows.
    idat = pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 3, 1, 3],
            "ct": [2, 2, 2, 8, 10, 6, 14, 12],
            "air_temp": [13, 9, 15, 23, 34, 20, 11, 18],
        }
    )

    return _prepare_data(pdat, idat)


def test_batched_predictions_match_single_sequence_predictions():
    prepared, original = _example_data()
    model, cont_vars, cat_vars, normalisation_values, device, response = _load_model()

    reference = _predict_emissions(
        prepared,
        original,
        model,
        cont_vars,
        cat_vars,
        normalisation_values,
        device,
        response=response,
        batch_size=1,
    )
    batched = _predict_emissions(
        prepared,
        original,
        model,
        cont_vars,
        cat_vars,
        normalisation_values,
        device,
        response=response,
        batch_size=3,
    )

    np.testing.assert_allclose(
        reference["prediction_ecum"],
        batched["prediction_ecum"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert batched["id"].tolist() == original["id"].tolist()
    assert batched["prediction_ecum"].notna().all()


def test_relative_emissions_are_prediction_divided_by_applied_tan():
    prepared, original = _example_data()
    model, cont_vars, cat_vars, normalisation_values, device, response = _load_model()

    predictions = _predict_emissions(
        prepared,
        original,
        model,
        cont_vars,
        cat_vars,
        normalisation_values,
        device,
        response=response,
        batch_size=3,
    )

    np.testing.assert_allclose(
        predictions["prediction_erel"],
        predictions["prediction_ecum"] / predictions["tan_app"],
    )


def test_public_prediction_rounds_all_numeric_output_to_two_decimals():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [60.1234],
            "app_method": ["bc"],
            "crop": ["grass"],
            "till": ["no"],
            "soil_type": ["clay"],
        }
    )

    predictions = predict(pdat)
    numeric = predictions.select_dtypes(include=np.number)

    pd.testing.assert_frame_equal(numeric, numeric.round(2))
    assert predictions["tan_app"].iat[0] == 60.12
    assert predictions["air_temp"].equals(predictions["air_temp"].round(2))
    assert predictions["wind_2m"].equals(predictions["wind_2m"].round(2))
    assert predictions["prediction_ecum"].equals(
        predictions["prediction_ecum"].round(2)
    )
    assert predictions["prediction_erel"].equals(
        predictions["prediction_erel"].round(2)
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("e_cum", [2.0, 4.0, 6.0]),
        ("e_cum_origin", [1.0, 2.0, 3.0]),
        ("delta_e_cum", [2.0, 6.0, 12.0]),
        ("delta_e_cum_origin", [1.0, 3.0, 6.0]),
        ("e_rel", [10.0, 40.0, 90.0]),
    ],
)
def test_response_is_converted_to_original_cumulative_emissions(response, expected):
    result = _to_cumulative_emissions(
        np.array([1.0, 2.0, 3.0]),
        response,
        {"e_cum": 2.0, "e_rel": 0.1},
        tan_app=np.array([100.0, 200.0, 300.0]),
    )
    np.testing.assert_allclose(result, expected)


def test_delta_draws_are_accumulated_independently():
    result = _to_cumulative_emissions(
        np.array([[1.0, 2.0], [10.0, 20.0]]),
        "delta_e_cum_origin",
        {"e_cum": 2.0},
    )
    np.testing.assert_allclose(result, [[1.0, 3.0], [10.0, 30.0]])


def test_e_rel_conversion_requires_applied_tan():
    with pytest.raises(ValueError, match="tan_app is required"):
        _to_cumulative_emissions(
            np.array([1.0]),
            "e_rel",
            {"e_rel": 0.1},
        )


def test_crop_z_default_depends_on_crop_and_preserves_explicit_values():
    pdat = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "tan_app": [50.0, 60.0, 70.0, 80.0],
            "app_method": ["bc", "bc", "bc", "bc"],
            "crop": ["bare soil", "grass", "maize", None],
            "crop_z": [np.nan, np.nan, 25.0, np.nan],
        }
    )

    completed, _ = _complete_pdat(pdat)

    assert completed["crop"].tolist() == [
        "bare soil",
        "grass",
        "maize",
        default_values["crop"],
    ]
    assert completed["crop_z"].tolist() == [
        -1000.0,
        default_values["crop_z"],
        25.0,
        default_values["crop_z"],
    ]


def test_crop_and_crop_z_defaults_when_both_columns_are_absent():
    completed, _ = _complete_pdat(
        pd.DataFrame({"id": [1], "tan_app": [50.0], "app_method": ["bc"]})
    )

    assert completed.loc[0, "crop"] == default_values["crop"]
    assert completed.loc[0, "crop_z"] == default_values["crop_z"]


def test_unknown_crop_is_not_exposed_by_the_public_preprocessing():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "crop": ["unknown"],
        }
    )

    with pytest.raises(ValueError, match="is not allowed for crop"):
        _complete_pdat(pdat)


def test_positive_crop_height_is_rejected_for_bare_soil():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "crop": ["bare soil"],
            "crop_z": [5.0],
        }
    )

    with pytest.raises(ValueError, match="crop_z must be missing or zero"):
        _complete_pdat(pdat)


def test_zero_crop_height_is_encoded_as_not_applicable_for_bare_soil():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "crop": ["bare soil"],
            "crop_z": [0.0],
        }
    )

    completed, original = _complete_pdat(pdat)
    assert completed.loc[0, "crop_z"] == -1000
    assert np.isnan(original.loc[0, "crop_z"])


def test_app_rate_default_is_computed_from_tan_app():
    pdat = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "tan_app": [50.0, 100.0, 150.0],
            "app_method": ["bc", "bc", "bc"],
            "app_rate": [np.nan, 42.0, np.nan],
        }
    )

    completed, _ = _complete_pdat(pdat)

    parameters = default_values["app_rate"]
    expected = [
        parameters["intercept"] + parameters["tan_app_coefficient"] * 50.0,
        42.0,
        parameters["intercept"] + parameters["tan_app_coefficient"] * 150.0,
    ]
    np.testing.assert_allclose(completed["app_rate"], expected)


def test_app_method_is_required_and_cannot_be_missing():
    with pytest.raises(ValueError, match="app_method is required"):
        _complete_pdat(pd.DataFrame({"id": [1], "tan_app": [50.0]}))

    with pytest.raises(ValueError, match="app_method must not contain missing values"):
        _complete_pdat(
            pd.DataFrame(
                {
                    "id": [1],
                    "tan_app": [50.0],
                    "app_method": [np.nan],
                }
            )
        )


def test_absent_and_missing_soil_values_receive_the_same_defaults():
    pdat = pd.DataFrame(
        {
            "id": [1, 2],
            "tan_app": [50.0, 50.0],
            "app_method": ["bc", "bc"],
            "soil_type": [np.nan, "clay"],
            "soil_water": [np.nan, 0.4],
            "soil_ph": [np.nan, 7.0],
            "soil_dens": [np.nan, 1.3],
            "soil_oc": [np.nan, 4.0],
        }
    )
    completed_missing, _ = _complete_pdat(pdat)
    completed_absent, _ = _complete_pdat(
        pdat.drop(
            columns=["soil_type", "soil_water", "soil_ph", "soil_dens", "soil_oc"]
        ).iloc[[0]]
    )

    for variable in ["soil_type", "soil_water", "soil_ph", "soil_dens", "soil_oc"]:
        assert completed_missing.loc[0, variable] == default_values[variable]
        assert completed_absent.iloc[0][variable] == default_values[variable]

    assert completed_missing.loc[1, "soil_type"] == "clay"
    assert completed_missing.loc[1, "soil_water"] == 0.4


def test_till_defaults_to_no_when_absent_or_missing():
    pdat = pd.DataFrame(
        {
            "id": [1, 2],
            "tan_app": [50.0, 50.0],
            "app_method": ["bc", "bc"],
            "till": [np.nan, "yes"],
        }
    )
    completed_missing, _ = _complete_pdat(pdat)
    completed_absent, _ = _complete_pdat(pdat.drop(columns="till"))

    assert completed_missing["till"].tolist() == ["no", "yes"]
    assert completed_absent["till"].eq("no").all()


def test_fertiliser_defaults_are_used_for_absent_and_missing_values():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "fer_origin": [np.nan],
            "fer_form": [np.nan],
            "fer_trt1": [np.nan],
            "fer_trt2": [np.nan],
            "fer_dm": [np.nan],
            "fer_ph": [np.nan],
        }
    )
    completed, original = _complete_pdat(pdat)

    for variable in [
        "fer_origin",
        "fer_form",
        "fer_trt1",
        "fer_trt2",
        "fer_dm",
        "fer_ph",
    ]:
        assert completed.loc[0, variable] == default_values[variable]
        assert original.loc[0, variable] == default_values[variable]

    assert completed.loc[0, "fer_trt3"] == "unknown"
    assert "fer_trt3" not in original.columns


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("fer_origin", "mink"),
        ("fer_form", "unknown"),
        ("fer_trt1", "Low DM"),
        ("fer_trt2", "Disruption"),
    ],
)
def test_untrained_fertiliser_categories_are_rejected(variable, value):
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            variable: [value],
        }
    )

    with pytest.raises(ValueError, match=f"for {variable}"):
        _complete_pdat(pdat)


def test_fer_trt3_is_not_a_public_input():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "fer_trt3": ["none"],
        }
    )

    with pytest.raises(ValueError, match="fer_trt3 is not"):
        _complete_pdat(pdat)


def test_meas_tech_is_not_a_public_input():
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
            "meas_tech": ["micro met"],
        }
    )

    with pytest.raises(ValueError, match="meas_tech is not"):
        _complete_pdat(pdat)


@pytest.mark.parametrize("variable", ["meas_tech", "fer_trt3"])
def test_internal_predictors_are_rejected_from_idat(variable):
    pdat = pd.DataFrame(
        {
            "id": [1],
            "tan_app": [50.0],
            "app_method": ["bc"],
        }
    )
    idat = pd.DataFrame(
        {
            "id": [1],
            "ct": [2.0],
            variable: ["micro met" if variable == "meas_tech" else "unknown"],
        }
    )

    with pytest.raises(ValueError, match="internal predictors"):
        _prepare_data(pdat, idat)


def test_rain_cum_is_derived_from_interval_rain_rate():
    pdat = pd.DataFrame({"id": [1, 2]})
    idat = pd.DataFrame(
        {
            "id": [1, 2, 1, 2, 1],
            "ct": [2.0, 1.0, 5.0, 4.0, 9.0],
            "rain_rate": [0.5, 0.0, 2.0, 1.0, 0.25],
        }
    )

    completed = _complete_idat(pdat, idat)

    np.testing.assert_allclose(completed["dt"], [2.0, 1.0, 3.0, 3.0, 4.0])
    np.testing.assert_allclose(completed["rain_cum"], [1.0, 0.0, 7.0, 3.0, 8.0])


def test_public_wind_is_routed_to_micrometeorological_branch():
    table = pd.DataFrame(
        {
            "wind_2m": [2.5, np.nan],
            "wind_2m_ind": [0, 2],
            "meas_tech": ["micro met", "micro met"],
        }
    )

    split = _split_wind_predictors(table)

    assert split["wind_micro_met"].iat[0] == 2.5
    assert split["wind_micro_met_ind"].tolist() == [0, 2]
    assert split["wind_chamber"].isna().all()
    assert split["wind_chamber_ind"].eq(-2).all()


def test_default_time_sequence():
    pdat = pd.DataFrame({"id": [1, 2]})

    completed = _complete_idat(pdat, idat=None)

    expected_ct = np.concatenate(
        (
            np.array([2, 4, 8]),
            np.arange(12, 49, 4),
            np.arange(60, 241, 12),
        )
    )
    for identifier in pdat["id"]:
        np.testing.assert_array_equal(
            completed.loc[completed["id"].eq(identifier), "ct"],
            expected_ct,
        )
    np.testing.assert_array_equal(
        completed.loc[completed["id"].eq(1), "dt"],
        np.diff(np.r_[0, expected_ct]),
    )


def test_user_supplied_rain_cum_is_recomputed():
    pdat = pd.DataFrame({"id": [1]})
    idat = pd.DataFrame(
        {
            "id": [1, 1],
            "ct": [2.0, 6.0],
            "rain_rate": [1.0, 0.5],
            "rain_cum": [999.0, 999.0],
        }
    )

    completed = _complete_idat(pdat, idat)

    np.testing.assert_allclose(completed["rain_cum"], [2.0, 4.0])


def test_time_incorp_default_depends_on_incorporation():
    pdat = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "tan_app": [50.0, 50.0, 50.0, 50.0],
            "app_method": ["bc", "bc", "bc", "bc"],
            "incorp": ["none", "shallow", "deep", None],
            "time_incorp": [np.nan, np.nan, 2.0, np.nan],
        }
    )

    completed, original = _complete_pdat(pdat)
    median = default_values["time_incorp"]["incorp_other"]

    assert completed.loc[0, "time_incorp"] == -1000
    assert np.isnan(original.loc[0, "time_incorp"])
    assert completed.loc[1, "time_incorp"] == median
    assert completed.loc[2, "time_incorp"] == 2.0
    assert completed.loc[3, "incorp"] == "none"
    assert completed.loc[3, "time_incorp"] == -1000
    assert np.isnan(original.loc[3, "time_incorp"])


def test_time_incorp_not_applicable_and_missing_codes_are_distinct():
    pdat = pd.DataFrame(
        {
            "id": [1, 2],
            "tan_app": [50.0, 50.0],
            "app_method": ["bc", "bc"],
            "incorp": ["none", "deep"],
            "time_incorp": [np.nan, 1.0],
        }
    )
    prepared, original = _prepare_data(pdat, idat=None)

    no_incorp = prepared["id"].eq(1)
    incorporated = prepared["id"].eq(2)
    assert prepared.loc[no_incorp, "time_incorp"].eq(-2).all()
    assert prepared.loc[no_incorp, "time_incorp_ind"].eq(-2).all()
    assert original.loc[no_incorp, "time_incorp"].isna().all()
    assert prepared.loc[incorporated, "time_incorp"].eq(1 / 48).all()
    assert prepared.loc[incorporated, "time_incorp_ind"].eq(0).all()


def test_time_varying_incorporation_features():
    pdat = pd.DataFrame(
        {
            "id": [1, 2],
            "tan_app": [50.0, 50.0],
            "app_method": ["bc", "bc"],
            "incorp": ["deep", "none"],
            "time_incorp": [5.0, np.nan],
        }
    )
    idat = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "ct": [2.0, 5.0, 9.0, 2.0, 5.0, 9.0],
        }
    )

    prepared, original = _prepare_data(pdat, idat)

    np.testing.assert_array_equal(original["incorporated_yet"], [0, 1, 1, 0, 0, 0])
    np.testing.assert_allclose(original["time_since_incorp"], [0, 0, 4, 0, 0, 0])
    np.testing.assert_array_equal(prepared["incorporated_yet"], [0, 1, 1, 0, 0, 0])
