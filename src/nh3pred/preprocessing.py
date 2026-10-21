import pickle
from importlib.resources import files

import numpy as np
import pandas as pd

PARAMETERS_DIR = files("nh3pred").joinpath("data")

with (PARAMETERS_DIR / "normalisation_values.pkl").open("rb") as file:
    normalisation_values = pickle.load(file)

with (PARAMETERS_DIR / "categorial_correspondances.pkl").open("rb") as file:
    categorial_correspondances = pickle.load(file)

with (PARAMETERS_DIR / "default_values.pkl").open("rb") as file:
    default_values = pickle.load(file)


### Plot level complete variable set
plot_level_columns_set = [
    "meas_tech",
    "incorp",
    "till",
    "fer_origin",
    "fer_form",
    "fer_trt1",
    "fer_trt2",
    "fer_trt3",
    "crop",
    "crop_z",
    "soil_type",
    "app_rate",
    "fer_dm",
    "fer_ph",
    "time_incorp",
    "furrow_z",
    "soil_water",
    "soil_ph",
    "soil_dens",
    "soil_oc",
]


### Interval level complete variable set
interval_level_columns_set = ["air_temp", "wind_2m", "rain_rate"]


# Categories sufficiently represented in the final training data.
# The encoding dictionaries retain historical levels for compatibility
# with the saved architecture; those levels are not exposed by the API.
public_fertiliser_categories = {
    "fer_origin": {"pig", "cat", "other", "conc", "mix"},
    "fer_form": {"liquid", "solid"},
    "fer_trt1": {
        "none",
        "Acidification",
        "Diluted",
        "Anaerobic digestion",
        "Irrigated",
    },
    "fer_trt2": {"none", "Separation"},
}


def _interpolate_default_series(variable, ct):
    """Interpolate a default profile at the requested times."""
    default_profile = default_values[variable]

    # Support older parameter files containing scalar defaults.
    if np.isscalar(default_profile):
        return np.full(len(ct), default_profile)

    return np.interp(
        np.asarray(ct, dtype=float),
        default_profile["ct"].to_numpy(dtype=float),
        default_profile[variable].to_numpy(dtype=float),
    )


def _complete_pdat(pdat):

    pdat = pdat.copy()

    if "app_method" not in pdat.columns:
        raise ValueError("app_method is required in pdat.")
    if pdat["app_method"].isna().any():
        raise ValueError("app_method must not contain missing values.")

    if "meas_tech" in pdat.columns:
        raise ValueError("meas_tech is not a public input variable.")
    if "fer_trt3" in pdat.columns:
        raise ValueError("fer_trt3 is not a public input variable.")

    # crop_z has a conditional default. Missing crop values use the saved
    # default. The "unknown" category exists in the internal artifacts but
    # is not exposed by the public API.
    if "crop" not in pdat.columns:
        pdat["crop"] = default_values["crop"]
    else:
        pdat["crop"] = pdat["crop"].fillna(default_values["crop"])

    if pdat["crop"].eq("unknown").any():
        raise ValueError("The 'unknown' category is not allowed for crop.")

    bare_soil = pdat["crop"].eq("bare soil")
    if "crop_z" in pdat.columns:
        positive_bare_soil_height = bare_soil & pdat["crop_z"].gt(0)
        if positive_bare_soil_height.any():
            raise ValueError(
                "crop_z must be missing or zero when crop = 'bare soil'."
            )

    default_crop_z = np.where(bare_soil, -1000, default_values["crop_z"])
    if "crop_z" not in pdat.columns:
        pdat["crop_z"] = default_crop_z
    else:
        pdat["crop_z"] = pdat["crop_z"].where(
            pdat["crop_z"].notna(),
            default_crop_z,
        )
        # A zero height on bare soil is accepted as input, then converted
        # to the internal "not applicable" code.
        pdat.loc[bare_soil, "crop_z"] = -1000

    # time_incorp remains missing without incorporation. Otherwise, its
    # default is the median calculated at the plot level.
    if "incorp" not in pdat.columns:
        pdat["incorp"] = default_values["incorp"]
    else:
        pdat["incorp"] = pdat["incorp"].fillna(default_values["incorp"])

    time_incorp_parameters = default_values["time_incorp"]
    if np.isscalar(time_incorp_parameters):
        default_time_incorp = np.full(len(pdat), time_incorp_parameters)
    else:
        default_time_incorp = np.where(
            pdat["incorp"].eq("none"),
            time_incorp_parameters["incorp_none"],
            time_incorp_parameters["incorp_other"],
        )

    if "time_incorp" not in pdat.columns:
        pdat["time_incorp"] = default_time_incorp
    else:
        pdat["time_incorp"] = pdat["time_incorp"].where(
            pdat["time_incorp"].notna(),
            default_time_incorp,
        )

    # Estimate app_rate from tan_app when it is not supplied.
    # Support older artifacts where the default was a scalar.
    app_rate_parameters = default_values["app_rate"]
    if np.isscalar(app_rate_parameters):
        default_app_rate = np.full(len(pdat), app_rate_parameters)
    else:
        if "tan_app" not in pdat.columns:
            raise ValueError(
                "tan_app is required to calculate app_rate when it is missing."
            )
        default_app_rate = (
            app_rate_parameters["intercept"]
            + app_rate_parameters["tan_app_coefficient"] * pdat["tan_app"]
        )

    if "app_rate" not in pdat.columns:
        pdat["app_rate"] = default_app_rate
    else:
        missing_app_rate = pdat["app_rate"].isna()
        if (
            missing_app_rate.any()
            and pdat.loc[missing_app_rate, "tan_app"].isna().any()
        ):
            raise ValueError(
                "tan_app is required to calculate missing app_rate values."
            )
        pdat["app_rate"] = pdat["app_rate"].where(
            ~missing_app_rate,
            default_app_rate,
        )

    # For the other application-level variables, an absent column and an NA
    # value both receive the default. The conditional cases above (crop_z,
    # time_incorp and app_rate) are handled separately.
    specially_completed = {"crop", "crop_z", "incorp", "time_incorp", "app_rate"}
    for col in plot_level_columns_set:
        if col == "app_method" or col in specially_completed:
            continue
        if col not in pdat.columns:
            pdat[col] = default_values[col]
        else:
            pdat[col] = pdat[col].fillna(default_values[col])

    for variable, allowed_values in public_fertiliser_categories.items():
        invalid_values = set(pdat[variable].dropna()) - allowed_values
        if invalid_values:
            raise ValueError(
                f"Unsupported category or categories for {variable}: "
                f"{sorted(invalid_values)}"
            )

    # Retain fer_trt3 internally for compatibility with older artifacts.
    # Newer models exclude it because it has no variation in the final data.
    # It is omitted from the public output.
    plot_level_origin = pdat.drop(columns="fer_trt3").copy()

    # Do not expose the internal sentinel in the public output.
    plot_level_origin.loc[bare_soil, "crop_z"] = np.nan

    # Without incorporation, the delay is structurally not applicable:
    # NA in the public output, sentinel -1000 before internal recoding
    # to -2 for the model.
    no_incorporation = pdat["incorp"].eq("none")
    plot_level_origin.loc[no_incorporation, "time_incorp"] = np.nan
    pdat.loc[no_incorporation, "time_incorp"] = -1000

    return pdat, plot_level_origin


def _complete_idat(pdat, idat):

    if idat is None:
        ids = pdat[["id"]]

        default_ct = np.concatenate(
            (
                np.array([2, 4, 8]),
                np.arange(12, 49, 4),
                np.arange(60, 241, 12),
            )
        )
        time_seq = pd.DataFrame({"ct": default_ct})

        idat = ids.merge(time_seq, how="cross")
        idat["air_temp"] = _interpolate_default_series("air_temp", idat["ct"])
        idat["wind_2m"] = _interpolate_default_series("wind_2m", idat["ct"])
        idat["rain_rate"] = default_values["rain_rate"]
        idat["dt"] = idat.groupby("id")["ct"].diff().fillna(idat["ct"])

    else:
        idat = idat.copy()

        interval_level_provided_cols = idat.columns.tolist()
        interval_level_missing_cols = sorted(
            set(interval_level_columns_set) - set(interval_level_provided_cols)
        )

        for s in interval_level_missing_cols:
            if s in {"air_temp", "wind_2m"}:
                idat[s] = _interpolate_default_series(s, idat["ct"])
            else:
                idat[s] = default_values[s]

        idat["dt"] = idat.groupby("id")["ct"].diff().fillna(idat["ct"])

    # rain_rate is the mean intensity over each interval. Derive cumulative
    # rainfall internally so that the two model predictors remain consistent.
    # A user-supplied rain_cum column, if present, is deliberately replaced.
    idat["rain_cum"] = (idat["rain_rate"] * idat["dt"]).groupby(idat["id"]).cumsum()

    return idat


def _add_indicator_variables(complete_table):

    complete_table = complete_table.copy()

    indicator_variables = [
        "air_temp",
        "wind_2m",
        "fer_dm",
        "fer_ph",
        "soil_water",
        "soil_ph",
        "soil_dens",
        "soil_oc",
        "crop_z",
        "time_incorp",
        "furrow_z",
    ]

    # Add indicator variables to idat.
    for col in indicator_variables:
        complete_table[f"{col}_ind"] = np.select(
            [
                complete_table[col].isna(),
                complete_table[col].eq(-1000),
            ],
            [2, -2],
            default=0,
        )

        complete_table[col] = complete_table[col].replace(-1000, np.nan)

    return complete_table


def _split_wind_predictors(complete_table):
    """Route the public wind_2m input to technique-specific model branches."""

    complete_table = complete_table.copy()
    source_known = complete_table["wind_2m_ind"].eq(0)
    branches = {
        "wind_micro_met": complete_table["meas_tech"].eq("micro met"),
        "wind_chamber": complete_table["meas_tech"].isin(["chamber", "cps", "wt"]),
    }
    if not np.logical_xor(*branches.values()).all():
        raise ValueError("Each row must belong to exactly one wind branch.")

    for variable, applicable in branches.items():
        known = applicable & source_known
        missing = applicable & ~source_known
        complete_table[variable] = complete_table["wind_2m"].where(known)
        complete_table[f"{variable}_ind"] = np.select(
            [known, missing], [0, 2], default=-2
        ).astype("int8")

    return complete_table


def _renormalize_tables(complete_table):

    complete_table = complete_table.copy()

    complete_table.insert(
        complete_table.columns.get_loc("ct") + 1,
        "ct_origin",
        complete_table["ct"],
    )

    for s in [
        "ct",
        "dt",
        "air_temp",
        "rain_cum",
        "rain_rate",
        "wind_2m",
        "wind_micro_met",
        "wind_chamber",
        "tan_app",
        "app_rate",
        "fer_dm",
        "fer_ph",
        "crop_z",
        "time_incorp",
        "time_since_incorp",
        "furrow_z",
        "soil_water",
        "soil_ph",
        "soil_dens",
        "soil_oc",
    ]:
        # Compatibility with model artefacts created before the derived
        # incorporation-time predictors were introduced.
        if s not in normalisation_values:
            continue

        complete_table[s] = complete_table[s] / normalisation_values[s]
        indicator = f"{s}_ind"
        if indicator in complete_table.columns:
            # -2 = not applicable; +2 = missing value.
            complete_table[s] = complete_table[s].fillna(complete_table[indicator])
        else:
            complete_table[s] = complete_table[s].fillna(2)

    return complete_table


def _recode_categorial_variables(complete_table):

    complete_table = complete_table.copy()

    categorial_vars = list(categorial_correspondances.keys())

    # Check that all supplied categories are known.
    for var in categorial_vars:
        unknown_values = set(complete_table[var].dropna()) - set(
            categorial_correspondances[var].keys()
        )

        if unknown_values:
            raise ValueError(
                f"Unknown category or categories for {var}: {sorted(unknown_values)}"
            )

    # Use the nested dictionary to recode all categorical columns at once.
    complete_table[categorial_vars] = (
        complete_table[categorial_vars]
        .replace(categorial_correspondances)
        .astype("int64")
    )

    return complete_table


def _prepare_data(pdat, idat):

    pdat = pdat.copy()
    if idat is not None:
        idat = idat.copy()
        internal_inputs = {"meas_tech", "fer_trt3"} & set(idat.columns)
        if internal_inputs:
            variables = ", ".join(sorted(internal_inputs))
            raise ValueError(
                f"{variables} must not be supplied in idat: "
                "these are internal predictors."
            )

    pdat, pdat_origin = _complete_pdat(pdat)
    idat = _complete_idat(pdat, idat)

    idat_origin = idat.merge(pdat_origin, how="left", on="id")
    idat = idat.merge(pdat, how="left", on="id")

    has_incorporation = idat["incorp"].ne("none") & idat["time_incorp"].ne(-1000)
    idat["incorporated_yet"] = (
        has_incorporation & idat["ct"].ge(idat["time_incorp"])
    ).astype("int8")
    idat["time_since_incorp"] = (idat["ct"] - idat["time_incorp"]).where(
        idat["incorporated_yet"].eq(1), 0.0
    )

    origin_has_incorporation = (
        idat_origin["incorp"].ne("none") & idat_origin["time_incorp"].notna()
    )
    idat_origin["incorporated_yet"] = (
        origin_has_incorporation & idat_origin["ct"].ge(idat_origin["time_incorp"])
    ).astype("int8")
    idat_origin["time_since_incorp"] = (
        idat_origin["ct"] - idat_origin["time_incorp"]
    ).where(idat_origin["incorporated_yet"].eq(1), 0.0)

    idat = _add_indicator_variables(idat)
    idat = _split_wind_predictors(idat)

    idat = _renormalize_tables(idat)
    idat = _recode_categorial_variables(idat)

    return idat, idat_origin
