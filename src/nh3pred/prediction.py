import pickle
from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch

from .model import AmmoniaRNN
from .physical_model import decode_physical_predictions
from .preprocessing import _prepare_data

DATA_DIR = files("nh3pred").joinpath("data")


@lru_cache(maxsize=1)
def _load_model():
    """Load the model once, when the first prediction is requested."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with (DATA_DIR / "normalisation_values.pkl").open("rb") as f:
        normalisation_values = pickle.load(f)

    with (DATA_DIR / "model_parameters.pkl").open("rb") as f:
        model_parameters = pickle.load(f)

    model_parameters = model_parameters.copy()
    cat_vars = model_parameters.pop("cat_vars")
    cont_vars = model_parameters.pop("cont_vars")
    # Support older artifacts trained on normalized cumulative emissions
    # that do not contain this metadata.
    response = model_parameters.pop("response", "e_cum")
    if response == ["e_rel", "j_nh3_rel_log"]:
        response = "physical"
    model_parameters.pop("loss", None)
    kinetic_rate_scale = model_parameters.pop("kinetic_rate_scale", 1e-2)
    flux_floor = model_parameters.pop("flux_floor", 1e-6)
    model_parameters.pop("flux_weight", None)

    model = AmmoniaRNN(**model_parameters).to(device)
    model.kinetic_rate_scale = kinetic_rate_scale
    model.flux_floor = flux_floor

    with (DATA_DIR / "model.pth").open("rb") as weights:
        state_dict = torch.load(weights, weights_only=True, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    return (
        model,
        cont_vars,
        cat_vars,
        normalisation_values,
        device,
        response,
    )


# --------------------------------------------------------------------
def _remove_padding(padded_tensor, lengths):

    return torch.cat(
        [padded_tensor[k, : length.item(), :] for k, length in enumerate(lengths)],
        dim=0,
    )


# --------------------------------------------------------------------


# --------------------------------------------------------------------
def _prepare_predictions(model_output):

    if isinstance(model_output, tuple):
        padded_predictions, lengths = model_output
        return _remove_padding(padded_predictions, lengths)

    return model_output


# --------------------------------------------------------------------


# --------------------------------------------------------------------
def _generate_tensors_predictors(df, id, cont_vars, cat_vars, device):

    df_filtered = df[df["id"] == id]

    x_cont = df_filtered[cont_vars]

    x_cont_tensor = torch.tensor(x_cont.values, dtype=torch.float32).view(
        len(x_cont), len(x_cont.columns)
    )
    x_cont_tensor = x_cont_tensor.to(device)

    x_cat = df_filtered[cat_vars]

    x_cat_tensor = torch.tensor(x_cat.values, dtype=torch.long).view(
        len(x_cat), len(x_cat.columns)
    )
    x_cat_tensor = x_cat_tensor.to(device)
    x_cat_tensor = torch.unbind(x_cat_tensor, dim=1)

    output = [x_cont_tensor, x_cat_tensor]

    return output


# --------------------------------------------------------------------


def _generate_tensors_from_group(df_group, cont_vars, cat_vars, device):
    """Build tensors from an already isolated sequence.

    This avoids scanning the entire DataFrame again for each application ID.
    """

    x_cont_tensor = torch.as_tensor(
        df_group[cont_vars].to_numpy(copy=True),
        dtype=torch.float32,
        device=device,
    )

    x_cat_tensor = torch.as_tensor(
        df_group[cat_vars].to_numpy(copy=True),
        dtype=torch.long,
        device=device,
    )

    return [x_cont_tensor, torch.unbind(x_cat_tensor, dim=1)]


def _batched(items, batch_size):
    """Split a list into batches of size ``batch_size``."""

    if batch_size < 1:
        raise ValueError("batch_size must be greater than or equal to 1")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _predict_prepared_groups(
    grouped_data,
    model,
    cont_vars,
    cat_vars,
    device,
    batch_size,
    normalisation_values=None,
    response="e_cum",
):
    """Predict batches of variable-length sequences."""

    predictions = {}

    for batch_groups in _batched(grouped_data, batch_size):
        batch_inputs = [
            _generate_tensors_from_group(
                df_group,
                cont_vars,
                cat_vars,
                device,
            )
            for _, df_group in batch_groups
        ]

        model_output = model(batch_inputs)

        if response == "physical":
            decoded = decode_physical_predictions(
                model_output,
                batch_inputs,
                normalisation_values,
                kinetic_rate_scale=model.kinetic_rate_scale,
                flux_floor=model.flux_floor,
            )
        else:
            padded_predictions, lengths = model_output

        for batch_index, (identifier, _) in enumerate(batch_groups):
            if response == "physical":
                values = decoded[batch_index]["e_rel"]
            else:
                length = lengths[batch_index].item()
                values = padded_predictions[batch_index, :length, 0]
            predictions[identifier] = values.detach().cpu().numpy()

    return predictions


# --------------------------------------------------------------------


def _to_cumulative_emissions(
    predictions,
    response,
    normalisation_values,
    tan_app=None,
):
    """Convert network output to cumulative emissions in the original units."""

    predictions = np.asarray(predictions)

    if response == "e_rel":
        if tan_app is None:
            raise ValueError("tan_app is required to convert the e_rel response.")
        predictions = predictions * normalisation_values["e_rel"] * np.asarray(tan_app)
    elif response == "physical":
        if tan_app is None:
            raise ValueError("tan_app is required for physical decoding.")
        predictions = predictions * np.asarray(tan_app)
    elif response in {"e_cum", "delta_e_cum"}:
        predictions = predictions * normalisation_values["e_cum"]

    if response in {"delta_e_cum", "delta_e_cum_origin"}:
        predictions = np.cumsum(predictions, axis=-1)
    elif response not in {"e_cum", "e_cum_origin", "e_rel", "physical"}:
        raise ValueError(
            f"Unknown response: {response}. Supported values: "
            "e_rel, e_cum, e_cum_origin, delta_e_cum, "
            "delta_e_cum_origin."
        )

    return predictions


# --------------------------------------------------------------------


def _add_relative_emissions(predictions):
    """Add unnormalized relative cumulative emissions to the public output."""

    predictions["prediction_erel"] = (
        predictions["prediction_ecum"] / predictions["tan_app"]
    )
    return predictions


# --------------------------------------------------------------------


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def _predict_emissions(
    idat,
    idat_origin,
    model,
    cont_vars,
    cat_vars,
    normalisation_values,
    device,
    response="e_cum",
    batch_size=128,
):

    # Apply the model to the prepared sequence for each application.

    idat_origin = idat_origin.copy()
    data_predictions = idat.copy()
    grouped_data = list(data_predictions.groupby("id", sort=False))
    origin_groups = {
        identifier: df_group
        for identifier, df_group in idat_origin.groupby("id", sort=False)
    }

    model.eval()

    with torch.inference_mode():
        predictions_by_id = _predict_prepared_groups(
            grouped_data,
            model,
            cont_vars,
            cat_vars,
            device,
            batch_size,
            normalisation_values,
            response,
        )

    idat_origin["prediction_ecum"] = np.nan

    for identifier, prepared_group in grouped_data:
        original_group = origin_groups[identifier]

        idat_origin.loc[original_group.index, "prediction_ecum"] = (
            _to_cumulative_emissions(
                predictions_by_id[identifier],
                response,
                normalisation_values,
                original_group["tan_app"].to_numpy(),
            )
        )

    return _add_relative_emissions(idat_origin)


# --------------------------------------------------------------------


def predict(pdat, idat=None, batch_size=128):
    """
    Predict cumulative ammonia emissions.

    Parameters
    ----------

    pdat : pandas.DataFrame
        Data describing each plot or application. See the ``Input reference``
        vignette for the accepted columns, units and categories.
    
    idat : pandas.DataFrame, optional
        Weather and time data. If omitted, default time and weather profiles
        bundled with the package are used. See the ``Input reference`` vignette
        for the accepted columns and units.

    batch_size : int, default=128
        Number of sequences processed simultaneously by the model.

    Returns
    -------
    pandas.DataFrame
        Original data completed with predictions. The ``prediction_ecum``
        column contains predicted cumulative ammonia emissions in kg N-NH3/ha.
        The ``prediction_erel`` column expresses the same predictions as a
        fraction of applied TAN.
    """

    model, cont_vars, cat_vars, normalisation_values, device, response = _load_model()

    prepared_data, original_data = _prepare_data(pdat, idat=idat)

    predictions = _predict_emissions(
        prepared_data,
        original_data,
        model=model,
        cont_vars=cont_vars,
        cat_vars=cat_vars,
        normalisation_values=normalisation_values,
        device=device,
        response=response,
        batch_size=batch_size,
    )
    return predictions.round(2)
