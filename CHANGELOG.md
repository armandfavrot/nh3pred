# Changelog

## 0.2.0

- Replace the original recurrent model with the physical reservoir model.
- Expose `predict(pdat, idat=None, batch_size=128)`, with application-level
  and time-dependent inputs and optional default weather profiles.
- Add relative cumulative predictions (`prediction_erel`).
- Bundle the weights, normalization, category codes and defaults together.
- Include batching, public-input validation and migration regression tests.

This release breaks compatibility with the 0.1.x input schema and changes
the fitted model and predictions. See the migration guide in the documentation.
Use `nh3pred==0.1.3` to reproduce results obtained with that version.
