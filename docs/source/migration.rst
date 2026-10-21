Migration to 0.2.0
==================

Version 0.2.0 introduces a new physical reservoir model and a new interface::

    from nh3pred import predict
    predictions = predict(pdat, idat=None, batch_size=128)

``pdat`` has one row per application, with required ``id``, ``tan_app`` and
``app_method`` columns. ``idat`` optionally supplies the weather and times.
The columns from 0.1.3 map as follows:

================== ==================
0.1.3              0.2.0
================== ==================
pmid               id
app_mthd           app_method
man_source         fer_origin
man_dm             fer_dm
man_ph             fer_ph
t_incorp           time_incorp
================== ==================

Retain a single consistent row of application-level inputs per ID. Supply
time rows in increasing order, starting strictly after zero. ``app_method``
is now required. Defaults and the fitted model have changed; a column rename
does not reproduce the old predictions. Pin ``nh3pred==0.1.3`` for old results.

The output adds ``prediction_erel`` (fraction of applied TAN). 

Current treatment categories are ``none``, ``Acidification``,
``Diluted``, ``Anaerobic digestion``, ``Irrigated`` for ``fer_trt1`` and
``none``, ``Separation`` for ``fer_trt2``. Fertilizer origins accepted by the
current code are ``pig``, ``cat``, ``other``, ``conc``, ``mix``. 
See the input-reference
vignette for all accepted categories and default values.

Pandas 3.0 or later is required to load the bundled serialized parameters.
