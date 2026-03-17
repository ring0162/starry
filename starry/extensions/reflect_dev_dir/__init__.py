# -*- coding: utf-8 -*-
"""
starry.extensions.reflect_dev_dir
==================================
Planetary reflection spectroscopy tools for direct-imaging observations.

Classes
-------
SpectralMap         : assigns reflectance spectra to surface regions on a starry map
Planet              : wraps SpectralMap with orbital/physical parameters
ReflectObservation  : simulates a direct-imaging spectroscopic observation

Telescope / Observation (noise model)
--------------------------------------
Telescope           : ELT/ANDES telescope hardware model (from obs_noise)
Observation         : ETC-based noise model (from obs_noise)

Utility functions
-----------------
load_surface_spectra, load_chlorophyll_spectra, load_purple_bacteria_spectra,
load_other_spectra, build_continent_spectra
gaussian_fit, H, reflectance_model, reflectance_model_short,
bcl_reflectance_model, fit_poly
wav_grid_from_R, resample_to_R
"""

import copy

# ---- reflect module (requires starry, spectres, astropy) ----
try:
    from .reflect import (
        SpectralMap,
        Planet,
        ReflectObservation,
        load_surface_spectra,
        load_chlorophyll_spectra,
        load_purple_bacteria_spectra,
        load_other_spectra,
        build_continent_spectra,
        gaussian_fit,
        H,
        reflectance_model,
        reflectance_model_short,
        bcl_reflectance_model,
        fit_poly,
    )
except Exception as _reflect_error:
    _reflect_err = copy.deepcopy(_reflect_error)

    def _reflect_unavailable(*args, **kwargs):
        raise _reflect_err

    SpectralMap = _reflect_unavailable
    Planet = _reflect_unavailable
    ReflectObservation = _reflect_unavailable
    load_surface_spectra = _reflect_unavailable
    load_chlorophyll_spectra = _reflect_unavailable
    load_purple_bacteria_spectra = _reflect_unavailable
    load_other_spectra = _reflect_unavailable
    build_continent_spectra = _reflect_unavailable
    gaussian_fit = _reflect_unavailable
    H = _reflect_unavailable
    reflectance_model = _reflect_unavailable
    reflectance_model_short = _reflect_unavailable
    bcl_reflectance_model = _reflect_unavailable
    fit_poly = _reflect_unavailable

# ---- obs_noise module (requires numpy, scipy, spectres, matplotlib) ----
try:
    from .obs_noise import (
        Telescope,
        Observation,
        wav_grid_from_R,
        resample_to_R,
    )
except Exception as _obs_error:
    _obs_err = copy.deepcopy(_obs_error)

    def _obs_unavailable(*args, **kwargs):
        raise _obs_err

    Telescope = _obs_unavailable
    Observation = _obs_unavailable
    wav_grid_from_R = _obs_unavailable
    resample_to_R = _obs_unavailable
