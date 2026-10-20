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
    """Charge le modèle une seule fois, lors de la première prédiction."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with (DATA_DIR / "normalisation_values.pkl").open("rb") as f:
        normalisation_values = pickle.load(f)

    with (DATA_DIR / "model_parameters.pkl").open("rb") as f:
        model_parameters = pickle.load(f)

    model_parameters = model_parameters.copy()
    cat_vars = model_parameters.pop("cat_vars")
    cont_vars = model_parameters.pop("cont_vars")
    # Compatibilité avec les anciens artefacts, entraînés sur la réponse
    # cumulée normalisée et dépourvus de cette métadonnée.
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
    """Construit les tenseurs d'une séquence déjà isolée.

    Cette variante évite de reparcourir tout le DataFrame pour chaque id.
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
    """Découpe une liste en lots de taille ``batch_size``."""

    if batch_size < 1:
        raise ValueError("batch_size doit être supérieur ou égal à 1")

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
    """Prédit plusieurs séquences de longueurs variables par batch."""

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
    """Convertit une sortie du réseau en émission cumulée d'origine."""

    predictions = np.asarray(predictions)

    if response == "e_rel":
        if tan_app is None:
            raise ValueError("tan_app est requis pour convertir la réponse e_rel.")
        predictions = predictions * normalisation_values["e_rel"] * np.asarray(tan_app)
    elif response == "physical":
        if tan_app is None:
            raise ValueError("tan_app est requis pour le décodage physique.")
        predictions = predictions * np.asarray(tan_app)
    elif response in {"e_cum", "delta_e_cum"}:
        predictions = predictions * normalisation_values["e_cum"]

    if response in {"delta_e_cum", "delta_e_cum_origin"}:
        predictions = np.cumsum(predictions, axis=-1)
    elif response not in {"e_cum", "e_cum_origin", "e_rel", "physical"}:
        raise ValueError(
            f"Réponse inconnue : {response}. Valeurs possibles : "
            "e_rel, e_cum, e_cum_origin, delta_e_cum, "
            "delta_e_cum_origin."
        )

    return predictions


# --------------------------------------------------------------------


def _add_relative_emissions(predictions):
    """Ajoute la réponse relative dénormalisée à la sortie publique."""

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

    # Le modèle est appliqué au pas de temps de référence, puis il y a ensuite une étape d'interpolation pour revenir aux pas de temps fournis par l'utilisateur

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
    Prédit les émissions cumulées d'ammoniac.

    Parameters
    ----------
    pdat : pandas.DataFrame
        Données décrivant chaque parcelle ou application.

    idat : pandas.DataFrame, optional
        Données météorologiques et temporelles.
        Si elles ne sont pas fournies, des valeurs par défaut sont utilisées.

    batch_size : int, default=128
        Nombre de séquences traitées simultanément par le modèle.

    Returns
    -------
    pandas.DataFrame
        Données d'origine complétées avec les prédictions. La colonne
        ``prediction_erel`` exprime l'émission cumulée prédite comme une
        fraction du TAN appliqué. Toutes les colonnes numériques de la sortie
        sont arrondies à deux décimales ; les calculs du modèle conservent leur
        précision complète en interne.
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
