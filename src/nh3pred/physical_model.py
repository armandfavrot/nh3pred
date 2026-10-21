"""Decode the latent kinetic rate into physically consistent emissions."""

import torch
import torch.nn.functional as F


def decode_physical_predictions(
    model_output,
    batch_x,
    normalisation_values,
    kinetic_rate_scale=1e-2,
    flux_floor=1e-6,
):
    """Decode each sequence into bounded cumulative emissions and non-negative flux."""

    padded_latent, lengths = model_output
    decoded = []

    for index, length in enumerate(lengths):
        sequence_length = length.item()
        latent = padded_latent[index, :sequence_length, 0]
        dt_origin = batch_x[index][0][:sequence_length, 1] * normalisation_values["dt"]

        kinetic_rate = F.softplus(latent) * kinetic_rate_scale
        integrated_rate = torch.cumsum(kinetic_rate * dt_origin, dim=0)
        prediction_e_rel = -torch.expm1(-integrated_rate)

        previous_e_rel = torch.cat(
            [
                prediction_e_rel.new_zeros(1),
                prediction_e_rel[:-1],
            ]
        )
        prediction_delta_e_rel = prediction_e_rel - previous_e_rel
        prediction_j_nh3_rel = prediction_delta_e_rel / dt_origin
        prediction_j_nh3_rel_log = torch.log(
            torch.clamp(prediction_j_nh3_rel, min=flux_floor)
        )

        decoded.append(
            {
                "e_rel": prediction_e_rel,
                "delta_e_rel": prediction_delta_e_rel,
                "j_nh3_rel": prediction_j_nh3_rel,
                "j_nh3_rel_log": prediction_j_nh3_rel_log,
                "e_rel_normalized": (prediction_e_rel / normalisation_values["e_rel"]),
                "j_nh3_rel_log_normalized": (
                    prediction_j_nh3_rel_log / normalisation_values["j_nh3_rel_log"]
                ),
            }
        )

    return decoded
