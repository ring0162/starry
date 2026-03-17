"""
Planetary reflection spectroscopy toolkit for starry.

Provides three main classes:
  - SpectralMap  : assigns reflectance spectra to surface regions on a starry map
  - Planet       : wraps a SpectralMap with orbital/physical parameters
  - ReflectObservation : simulates a direct-imaging spectroscopic observation

Utility functions:
  - load_surface_spectra()          : load standard ASTER/MODIS surface spectra
  - load_chlorophyll_spectra()      : load natural chlorophyll absorption spectra
  - load_purple_bacteria_spectra()  : load Coelho et al. 2024 bacterial spectra
  - load_other_spectra()            : load water, trees_cont, hydrogenic, Proxima b spectra
  - build_continent_spectra()       : build composite continental spectra from surface types
  - gaussian_fit()                  : single Gaussian absorption feature
  - H()                             : smooth Heaviside step (VRE edge model)
  - reflectance_model()             : polynomial + step + Gaussians
  - reflectance_model_short()       : polynomial + step only
  - bcl_reflectance_model()         : bacteriochlorophyll-specific model
  - fit_poly()                      : polynomial fit to a spectrum

Note: starry.config.lazy and starry.config.quiet are NOT set here — configure
them in your script before importing this module.
"""

import os
import glob
import re
import pickle

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import starry
from scipy import integrate, stats
from astropy.io import fits
from astropy.constants import h, c, G, M_sun, R_sun, au
from spectres import spectres

from . import obs_noise as obs

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_SPECTRA_DIR = os.path.join(_HERE, "input_spectra")
_BAYES_DIR = os.path.join(_SPECTRA_DIR, "bayes_paper")
_WATER_DIR = os.path.join(_SPECTRA_DIR, "water")
_COELHO_DIR = os.path.join(_SPECTRA_DIR, "Coelhoetal_2024_zenodo")
_CHLORO_DIR = os.path.join(_SPECTRA_DIR, "NaturalChlorophylls")
_PROXB_DIR = os.path.join(_SPECTRA_DIR, "proximab")


# ---------------------------------------------------------------------------
# Spectral loading functions
# ---------------------------------------------------------------------------

def load_surface_spectra():
    """
    Load standard surface reflectance spectra from the bayes_paper input
    directory (ASTER, MODIS, USGS sources).

    Returns
    -------
    dict
        Keys: 'basalt', 'granite', 'sand', 'grass', 'trees', 'cloud',
               'coast', 'snow', 'sea', 'earth_barrientos'
        Values: (N, 2) arrays with columns [wavelength_um, reflectance]
    """
    region_names = [
        "basalt", "granite", "sand", "grass", "trees",
        "cloud", "coast", "snow", "sea", "earth_barrientos",
    ]
    regions = dict.fromkeys(region_names)

    regions["basalt"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-rock-basalt-solid-ASTER.txt"), skiprows=26
    )
    regions["granite"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-rock-granite-solid-alkalic-ASTER.txt"), skiprows=26
    )
    regions["sand"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-soil-sand-brown_loamy_fine-ASTER.txt"), skiprows=26
    )
    regions["grass"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-vegetation-grass-lawn-ASTER.txt"), skiprows=26
    )
    regions["trees"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-vegetation-trees-deciduous-ASTER.txt"), skiprows=26
    )
    regions["cloud"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-water-cloud-MODIS.txt"), skiprows=26
    )
    regions["coast"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-water-coast-USGS+ASTER.txt"), skiprows=26
    )
    regions["snow"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-water-fine_snow-ASTER.txt"), skiprows=26
    )
    regions["sea"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "albedo-water-seawater-USGS+ASTER.txt"), skiprows=26
    )
    regions["earth_barrientos"] = np.loadtxt(
        os.path.join(_BAYES_DIR, "EarthSurface.txt")
    )

    # Convert percent reflectance → fractional where applicable
    for key in ("basalt", "granite", "sand", "grass", "trees", "snow"):
        regions[key][:, 1] = regions[key][:, 1] / 100.0

    # Some ASTER files are stored in reverse wavelength order
    for key in ("basalt", "granite", "sand"):
        regions[key] = np.flip(regions[key], axis=0)

    return regions


def load_chlorophyll_spectra():
    """
    Load natural chlorophyll absorption spectra (Chl a, b, d, f).

    Expects files in input_spectra/NaturalChlorophylls/ — this directory must
    be populated by the user from the photosynthesis spectral library.

    Returns
    -------
    dict
        Keys: 'cla', 'clb', 'cld', 'clf'
        Values: (N, 2) arrays with columns [wavelength, absorption]
    """
    chlorophyll_files = {
        "cla": "CHL006_Chl a, Et2O (Du, 1998).abs.txt",
        "clb": "CHL014_Chl b, Et2O (Du, 1998).abs.txt",
        "cld": "CHL022_Chl d, Et2O (Li, 2012).abs.txt",
        "clf": "CHL031_Ch f, Et2O (Kobayashi, 2013).abs.txt",
    }
    chlorophylls = {}
    for key, fname in chlorophyll_files.items():
        fpath = os.path.join(_CHLORO_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Chlorophyll spectrum '{fname}' not found at {fpath}. "
                "Please copy NaturalChlorophylls spectra into input_spectra/NaturalChlorophylls/."
            )
        chlorophylls[key] = np.loadtxt(fpath, skiprows=1)
    return chlorophylls


def load_purple_bacteria_spectra():
    """
    Load Coelho et al. 2024 purple/cyanobacterial reflectance spectra.

    Returns
    -------
    dict
        Nested dict: purple_spectra['PNSB']['wet']['RV'], etc.
        Top-level keys: 'PNSB', 'PSB', 'Cyano', 'Other'
    """
    reflect_files = glob.glob(os.path.join(_COELHO_DIR, "*.csv"))

    PNSB_wet = ["RV", "BV", "E01", "E02", "E11", "E18", "E23", "E26", "E33", "E45", "Contwet"]
    PNSB_dry = ["RVdry", "BVdry", "E01dry", "E02dry", "E11dry", "E18dry", "E23dry",
                "E26dry", "E33dry", "E45dry", "Contdry"]
    PSB_wet = ["E03", "E05", "E06", "E07", "E10", "E13", "E28", "E35", "E41",
               "E50", "E51", "E53", "Contwet"]
    PSB_dry = ["E03dry", "E05dry", "E06dry", "E07dry", "E10dry", "E13dry", "E28dry",
               "E35dry", "E41dry", "E50dry", "E51dry", "E53dry", "Contdry"]
    Cyano_wet = ["Ana", "Glo", "Contwet"]
    Cyano_dry = ["Anadry", "Glodry", "Contdry"]
    Other = ["E52", "E52dry"]

    purple_spectra = {
        "PNSB": {"wet": dict.fromkeys(PNSB_wet), "dry": dict.fromkeys(PNSB_dry)},
        "PSB":  {"wet": dict.fromkeys(PSB_wet),  "dry": dict.fromkeys(PSB_dry)},
        "Cyano": {"wet": dict.fromkeys(Cyano_wet), "dry": dict.fromkeys(Cyano_dry)},
        "Other": dict.fromkeys(Other),
    }

    for fname in reflect_files:
        bac = re.split("[/_]", fname)[-2]
        if bac in PNSB_wet:
            purple_spectra["PNSB"]["wet"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in PNSB_dry:
            purple_spectra["PNSB"]["dry"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in PSB_wet:
            purple_spectra["PSB"]["wet"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in PSB_dry:
            purple_spectra["PSB"]["dry"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in Cyano_wet:
            purple_spectra["Cyano"]["wet"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in Cyano_dry:
            purple_spectra["Cyano"]["dry"][bac] = np.loadtxt(fname, delimiter=",")
        if bac in Other:
            purple_spectra["Other"][bac] = np.loadtxt(fname, delimiter=",")

    return purple_spectra


def load_other_spectra():
    """
    Load water, trees continuum, hydrogenic, and Proxima b spectra.

    Some files may be missing if not yet added to input_spectra/; a warning is
    printed for each missing optional file rather than raising immediately.

    Returns
    -------
    dict
        Keys: 'waterabs', 'waterrefl', 'trees_continuum',
               'hyd_dry', 'hyd_wet', 'hyd_dry_cont', 'hyd_wet_cont',
               'proxb_incident', 'proxb_incident_nphot',
               'prox_cen_muscles', 'proxb_snr'
    """
    other_spec = {}

    # Water absorption (Kou et al. 1993)
    water_abs_path = os.path.join(_WATER_DIR, "water.txt")
    if os.path.exists(water_abs_path):
        other_spec["waterabs"] = np.loadtxt(water_abs_path)
    else:
        print(f"Warning: water absorption spectrum not found at {water_abs_path}")

    # Tap water reflectance (JHU spectral library)
    water_refl_path = os.path.join(
        _WATER_DIR,
        "water.tapwater.none.liquid.all.tapwater.jhu.becknic.spectrum.txt",
    )
    if os.path.exists(water_refl_path):
        other_spec["waterrefl"] = np.loadtxt(water_refl_path, skiprows=22)
    else:
        print(f"Warning: water reflectance spectrum not found at {water_refl_path}")

    # Trees continuum
    trees_cont_path = os.path.join(_SPECTRA_DIR, "trees_cont.txt")
    if os.path.exists(trees_cont_path):
        trees_cont = np.loadtxt(trees_cont_path)
        trees_cont[:, 1] = trees_cont[:, 1] / 100.0
        other_spec["trees_continuum"] = trees_cont
    else:
        print(f"Warning: trees continuum spectrum not found at {trees_cont_path}")

    # Hydrogenic spectra (pickle)
    hyd_pkl_path = os.path.join(_SPECTRA_DIR, "hydrogenic.pkl")
    if os.path.exists(hyd_pkl_path):
        with open(hyd_pkl_path, "rb") as f:
            hydrogenic = pickle.load(f)
        other_spec["hyd_dry"] = np.column_stack((hydrogenic["wav"] / 1000.0, hydrogenic["dry"]))
        other_spec["hyd_wet"] = np.column_stack((hydrogenic["wav"] / 1000.0, hydrogenic["wet"]))
        other_spec["hyd_dry_cont"] = np.column_stack(
            (hydrogenic["wav"] / 1000.0, hydrogenic["cont_dry"])
        )
        other_spec["hyd_wet_cont"] = np.column_stack(
            (hydrogenic["wav"] / 1000.0, hydrogenic["cont_wet"])
        )
    else:
        print(f"Warning: hydrogenic pickle not found at {hyd_pkl_path}")

    # Proxima b incident spectra
    proxb_txt_path = os.path.join(_PROXB_DIR, "prox_cen_b_incident_spec.txt")
    if os.path.exists(proxb_txt_path):
        prox_b_wav, prox_b_TAO, _, _ = np.loadtxt(
            proxb_txt_path, unpack=True, skiprows=1
        )
        other_spec["proxb_incident"] = np.column_stack((prox_b_wav, prox_b_TAO))
    else:
        print(f"Warning: Proxima b incident spectrum not found at {proxb_txt_path}")

    proxb_phot_path = os.path.join(_PROXB_DIR, "prox_cen_b_incident_spec_phot.txt")
    if os.path.exists(proxb_phot_path):
        prox_b_wav, prox_b_nphot, _ = np.loadtxt(
            proxb_phot_path, unpack=True, skiprows=1
        )
        other_spec["proxb_incident_nphot"] = np.column_stack((prox_b_wav, prox_b_nphot))
    else:
        print(f"Warning: Proxima b photon spectrum not found at {proxb_phot_path}")

    # Proxima Centauri MUSCLES SED
    muscles_path = os.path.join(
        _PROXB_DIR,
        "hlsp_muscles_multi_multi_gj551_broadband_v22_adapt-const-res-sed.fits",
    )
    if os.path.exists(muscles_path):
        proxima_spec = fits.getdata(muscles_path, 1)
        other_spec["prox_cen_muscles"] = np.column_stack(
            (proxima_spec["WAVELENGTH"] / 10000.0, proxima_spec["FLUX"] * 10000.0)
        )  # convert to microns and ergs/s/cm^2/um
    else:
        print(f"Warning: Proxima Centauri MUSCLES SED not found at {muscles_path}")

    # Pre-computed Proxima b SNR
    proxb_snr_path = os.path.join(_HERE, "proxb_snr_R100.txt")
    if os.path.exists(proxb_snr_path):
        other_spec["proxb_snr"] = np.loadtxt(proxb_snr_path, skiprows=1)
    else:
        print(f"Warning: Proxima b SNR file not found at {proxb_snr_path}")

    return other_spec


def build_continent_spectra(regions, other_spec, wav_deep=None):
    """
    Build composite continental surface spectra from standard surface types.

    Parameters
    ----------
    regions : dict
        Output of load_surface_spectra().
    other_spec : dict
        Output of load_other_spectra() — must contain 'trees_continuum'.
    wav_deep : array-like, optional
        Target wavelength grid. Defaults to obs.wav_grid_from_R(1000, 0.4, 2.5).

    Returns
    -------
    dict
        Updated regions dict with added composite keys:
        'cont', 'cont_warm', 'cont_noveg', 'cont_continuum',
        'new_cont', 'new_cont_warm', 'new_cont2', 'new_cont3'
    """
    if wav_deep is None:
        wav_deep = obs.wav_grid_from_R(1000, 0.4, 2.5)

    cont_components = {}
    for key in ("grass", "trees", "granite", "basalt", "sand", "snow", "sea"):
        cont_components[key] = spectres(
            wav_deep,
            regions[key][:, 0],
            regions[key][:, 1],
            fill=regions[key][:, 1][0],
        )

    grass = cont_components["grass"]
    trees = cont_components["trees"]
    granite = cont_components["granite"]
    basalt = cont_components["basalt"]
    sand = cont_components["sand"]
    snow = cont_components["snow"]
    sea = cont_components["sea"]

    trees_continuum = None
    if "trees_continuum" in other_spec and other_spec["trees_continuum"] is not None:
        trees_continuum = spectres(
            wav_deep,
            other_spec["trees_continuum"][:, 0],
            other_spec["trees_continuum"][:, 1],
            fill=other_spec["trees_continuum"][:, 1][0],
        )

    cont = 0.3 * grass + 0.3 * trees + 0.09 * granite + 0.09 * basalt + 0.07 * sand + 0.15 * snow
    cont_warm = (0.3 * grass + 0.3 * trees + 0.09 * granite + 0.09 * basalt + 0.07 * sand) / 0.85
    cont_noveg = (0.09 * granite + 0.09 * basalt + 0.07 * sand) / 0.25
    new_cont = (
        0.1 * grass + 0.1 * trees + 0.09 * granite * 2 + 0.09 * basalt * 2
        + 0.07 * sand * 2 + 0.15 * snow
    )
    new_cont2 = (
        0.1 * grass + 0.1 * trees + 0.09 * granite * 2.6 + 0.09 * basalt * 2.6
        + 0.07 * sand * 2.6 + 0.15 * snow
    )
    new_cont3 = (
        0.1 * grass + 0.1 * trees + 0.09 * granite * 2 + 0.09 * basalt * 2
        + 0.07 * sand * 2 + 0.15 * snow + 0.15 * sea
    )
    new_cont_warm = (
        0.125 * grass + 0.125 * trees + 0.09 * granite * 3
        + 0.09 * basalt * 3 + 0.07 * sand * 3
    )

    regions["cont"] = np.column_stack((wav_deep, cont))
    regions["cont_warm"] = np.column_stack((wav_deep, cont_warm))
    regions["cont_noveg"] = np.column_stack((wav_deep, cont_noveg))
    regions["new_cont"] = np.column_stack((wav_deep, new_cont))
    regions["new_cont2"] = np.column_stack((wav_deep, new_cont2))
    regions["new_cont3"] = np.column_stack((wav_deep, new_cont3))
    regions["new_cont_warm"] = np.column_stack((wav_deep, new_cont_warm))

    if trees_continuum is not None:
        cont_continuum = (
            0.6 * trees_continuum + 0.09 * granite + 0.09 * basalt + 0.07 * sand + 0.15 * snow
        )
        regions["cont_continuum"] = np.column_stack((wav_deep, cont_continuum))

    return regions


# ---------------------------------------------------------------------------
# Spectral fitting functions
# ---------------------------------------------------------------------------

def gaussian_fit(wav, mu=0.7, sigma=0.02, A=0.1):
    """Single Gaussian absorption feature (multiplicative dip)."""
    return 1.0 - A * np.exp(-0.5 * ((wav - mu) / sigma) ** 2)


def H(wav, b=0.1, smooth=80, edge_loc=0.72):
    """
    Smooth Heaviside step function for the Vegetation Red Edge.

    From Brandt (2014). Models an abrupt reflectance increase at edge_loc.
    """
    return b * (1 + np.exp(smooth * (edge_loc - wav))) ** (-1)


def reflectance_model(
    wav,
    poly_coeffs,
    smooth=80,
    edge_loc=0.72,
    edge_size=0.02,
    mu=(1.4, 1.9, 2.5),
    sigma=(0.02, 0.02, 0.02),
    A=(1e-4, 1e-4, 1e-4),
):
    """Polynomial + Heaviside step + Gaussian absorption features."""
    poly = poly_coeffs[0] * (wav - wav[0]) ** 2 + poly_coeffs[1] * (wav - wav[0]) + poly_coeffs[2]
    heaviside = H(wav, b=edge_size, smooth=smooth, edge_loc=edge_loc)
    gauss = np.ones_like(wav)
    for i in range(len(mu)):
        gauss *= gaussian_fit(wav, mu=mu[i], sigma=sigma[i], A=A[i])
    return poly * (1 + heaviside) * gauss


def reflectance_model_short(wav, poly_coeffs, smooth=80, edge_loc=0.72, edge_size=0.02):
    """Polynomial + Heaviside step only (no Gaussian features)."""
    poly = poly_coeffs[0] * wav ** 2 + poly_coeffs[1] * wav + poly_coeffs[2]
    heaviside = H(wav, b=edge_size, smooth=smooth, edge_loc=edge_loc)
    return poly * (1 + heaviside)


def bcl_reflectance_model(
    wav,
    poly_coeffs,
    Bcla_Qy_loc=0.85,
    Bcla_Qy_gap=0.015,
    Bcla_Qy_amp=(0.2, 0.1),
    Bcla_Qy_width=(0.02, 0.02),
    carot_Qx_soret_loc=0.4,
    carot_Qx_soret_amp=0.1,
    carot_Qx_soret_width=0.04,
    mu=(1.4, 1.9, 2.5),
    sigma=(0.02, 0.02, 0.02),
    A=(1e-4, 1e-4, 1e-4),
):
    """
    Bacteriochlorophyll-a specific reflectance model.

    Includes BChl-a Qy absorption (~0.85 μm) and carotenoid Qx/Soret (~0.4 μm),
    plus water absorption bands.
    """
    poly = poly_coeffs[0] * (wav - wav[0]) ** 2 + poly_coeffs[1] * (wav - wav[0]) + poly_coeffs[2]
    gauss = np.ones_like(wav)
    for i in range(len(mu)):
        gauss *= gaussian_fit(wav, mu=mu[i], sigma=sigma[i], A=A[i])
    gauss *= (
        gaussian_fit(wav, mu=Bcla_Qy_loc, sigma=Bcla_Qy_width[0], A=Bcla_Qy_amp[0])
        * gaussian_fit(wav, mu=Bcla_Qy_loc - Bcla_Qy_gap, sigma=Bcla_Qy_width[1], A=Bcla_Qy_amp[1])
        * gaussian_fit(wav, mu=carot_Qx_soret_loc, sigma=carot_Qx_soret_width, A=carot_Qx_soret_amp)
    )
    return poly * gauss


def fit_poly(wav, refl, deg=2):
    """Fit a polynomial to a reflectance spectrum and return the fit."""
    p = np.polyfit(wav, refl, deg)
    print(p)
    return np.polyval(p, wav)


# ---------------------------------------------------------------------------
# SpectralMap class
# ---------------------------------------------------------------------------

class SpectralMap:
    """
    Assigns wavelength-dependent reflectance spectra to surface regions on a
    planet map and encodes them as starry spherical-harmonic maps.

    Parameters
    ----------
    map_image : 2D array
        Greyscale map image. Values < 0.5 are ocean, >= 0.5 are continent.
    wav : array
        Wavelength grid (microns).
    cont_id : float
        Pixel ID for continental regions.
    cont_spec : array
        Reflectance spectrum for continents, length == len(wav).
    ocean_id : float
        Pixel ID for ocean regions.
    ocean_spec : array
        Reflectance spectrum for oceans, length == len(wav).
    cloud_id, cloud_spec, cloud_dims, cloud_num : optional
        Cloud parameters. cloud_dims = (min_size_deg, max_size_deg).
    snow_id, snow_spec, snow_dims : optional
        Polar cap parameters. snow_dims = (sea_north, cont_north, sea_south, cont_south) pixel rows.
    coast_id, coast_spec, coast_size : optional
        Coastline parameters.
    ydeg : int
        Spherical harmonic degree for the starry maps.
    scalar_wav : float
        Wavelength (microns) at which to produce a scalar visualisation map.
    smoothing : float or None
        Smoothing parameter passed to starry.Map.load().
    roughness : float
        Roughness parameter for starry maps.
    """

    def __init__(
        self,
        map_image,
        wav,
        cont_id,
        cont_spec,
        ocean_id,
        ocean_spec,
        cloud_id=None,
        cloud_spec=None,
        cloud_dims=(30, 140),
        cloud_num=6,
        snow_id=None,
        snow_spec=None,
        snow_dims=(25, 41, 25, 41),
        coast_id=None,
        coast_spec=None,
        coast_size=3,
        ydeg=20,
        scalar_wav=0.7,
        smoothing=None,
        roughness=0.0,
    ):
        map_image[map_image < 0.5] = ocean_id
        map_image[map_image >= 0.5] = cont_id

        self.map = np.flipud(map_image)
        self.wav = wav
        self.cont_id = cont_id
        self.cont_spec = cont_spec
        self.ocean_id = ocean_id
        self.ocean_spec = ocean_spec

        self.coast_id = coast_id
        if coast_id is None:
            self.coast_spec = np.nan * np.ones(len(wav))
        else:
            self.coast_spec = coast_spec
            self.map = self.add_coast(self.map, spec=coast_id, size=coast_size)

        self.snow_id = snow_id
        if snow_id is None:
            self.snow_spec = np.nan * np.ones(len(wav))
        else:
            self.snow_spec = snow_spec
            self.map = self.add_poles(
                self.map,
                spec=snow_id,
                sea_north=snow_dims[0],
                cont_north=snow_dims[1],
                sea_south=snow_dims[2],
                cont_south=snow_dims[3],
            )

        self.cloud_id = cloud_id
        if cloud_id is None:
            self.cloud_spec = np.nan * np.ones(len(wav))
        else:
            self.cloud_spec = cloud_spec
            self.map = self.add_clouds(
                self.map, spec=cloud_id, min_size=cloud_dims[0], max_size=cloud_dims[1], num=cloud_num
            )

        self.ydeg = ydeg
        self.scalar_wav = scalar_wav
        self.smoothing = smoothing
        self.roughness = roughness
        self.fullspecmap = starry.Map(
            ydeg=self.ydeg, reflected=True, nw=len(self.wav), wav=self.wav, roughness=self.roughness
        )
        self.scalarmap = starry.Map(ydeg=self.ydeg, reflected=True, roughness=self.roughness)
        _, _, Y2P, _, _, _ = self.scalarmap.get_pixel_transforms()
        p = Y2P.dot(self.scalarmap.y)
        self.specmap_p = np.zeros((len(self.wav), p.shape[0]))

        print("Spectral map created. Call get_specmap() to get the full spectral map.")
        self.surface_dist = self.get_pixel_ratio(to_print=False)
        self.surface_area = self.get_areas()

    def get_pixel_ratio(self, to_plot=True, to_print=True):
        ids, num = np.unique(self.map, return_counts=True)
        tot = self.map.shape[0] * self.map.shape[1]

        if to_print:
            print(
                "The continent is {} percent of the map pixels".format(
                    np.round(num[ids == self.cont_id] / tot * 100.0, decimals=1)
                )
            )
            print(
                "The ocean is {} percent of the map pixels".format(
                    np.round(num[ids == self.ocean_id] / tot * 100.0, decimals=1)
                )
            )
            if self.cloud_id is not None:
                print(
                    "The clouds are {} percent of the map pixels".format(
                        np.round(num[ids == self.cloud_id] / tot * 100.0, decimals=1)
                    )
                )
            if self.snow_id is not None:
                print(
                    "The poles are {} percent of the map pixels".format(
                        np.round(num[ids == self.snow_id] / tot * 100.0, decimals=1)
                    )
                )
            if self.coast_id is not None:
                print(
                    "The coast is {} percent of the map pixels".format(
                        np.round(num[ids == self.coast_id] / tot * 100.0, decimals=1)
                    )
                )

        if to_plot:
            fig = plt.figure()
            ax = plt.axes()
            im = ax.imshow(np.flipud(self.map))
            plt.title("Map with regions assigned", fontsize=18)
            cax = fig.add_axes(
                [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height]
            )
            plt.colorbar(im, cax=cax)
            plt.show()

        return dict(zip(ids, num))

    def get_areas(self, to_print=True):
        ids_glob = np.unique(self.map)
        delta_lat = np.pi / (self.map.shape[0] - 1)
        A_tot = 0.0
        A_surface_type = dict.fromkeys(ids_glob, 0.0)

        for row in range(self.map.shape[0]):
            ids, num = np.unique(self.map[row, :], return_counts=True)
            row_lat = row * delta_lat - np.pi / 2.0
            A_lat = 2 * np.pi * (np.sin(row_lat + delta_lat) - np.sin(row_lat))
            A_tot += A_lat
            for i, id_ in enumerate(ids):
                A_surface_type[id_] += num[i] / self.map.shape[1] * A_lat

        for key in A_surface_type.keys():
            A_surface_type[key] = A_surface_type[key] / A_tot

        if to_print:
            print(
                "The map has a total area of {}pi square radians".format(
                    np.round(A_tot / np.pi, decimals=1)
                )
            )
            print(
                "The continent is {} percent of the map area".format(
                    np.round(A_surface_type[self.cont_id] * 100.0, decimals=1)
                )
            )
            print(
                "The ocean is {} percent of the map area".format(
                    np.round(A_surface_type[self.ocean_id] * 100.0, decimals=1)
                )
            )
            if self.cloud_id is not None:
                print(
                    "The clouds are {} percent of the map area".format(
                        np.round(A_surface_type[self.cloud_id] * 100.0, decimals=1)
                    )
                )
            if self.snow_id is not None:
                print(
                    "The poles are {} percent of the map area".format(
                        np.round(A_surface_type[self.snow_id] * 100.0, decimals=1)
                    )
                )
            if self.coast_id is not None:
                print(
                    "The coast is {} percent of the map area".format(
                        np.round(A_surface_type[self.coast_id] * 100.0, decimals=1)
                    )
                )

        return A_surface_type

    def add_clouds(self, map, spec=4.0, min_size=30, max_size=140, num=6):
        """Place random circular clouds over the map. Call last, after add_coast."""
        map_dims = map.shape
        cloud_pix = np.zeros((num, 3))

        for i in range(num):
            size = np.random.uniform(min_size, max_size)
            lon = np.random.uniform(0, map_dims[1])
            lat = np.random.uniform(0, map_dims[0])
            r_pix = int(size / map_dims[0] * map_dims[1])

            for x, y in np.ndindex(map.shape):
                if (x - lat) ** 2 + (y - lon) ** 2 < r_pix ** 2:
                    map[x, y] = spec

            cloud_pix[i] = (lat, lon, r_pix)

        return map

    def add_coast(self, map, spec=2.0, size=3.0):
        """Add a coastline buffer of width `size` pixels around continent edges."""
        coast_ij = np.zeros((map.shape[0], map.shape[1]))

        for row in range(len(map[:, 0]) - 1):
            for col in range(len(map[0]) - 1):
                if map[row, col] != map[row, col + 1]:
                    coast_ij[row, col] = 1.0
                if map[row, col] != map[row + 1, col]:
                    coast_ij[row, col] = 0.8

        for row in range(len(coast_ij[:, 0])):
            for col in range(len(coast_ij[0])):
                if coast_ij[row, col] == 1.0:
                    map[row, col] = spec
                    for i in range(size):
                        if col + i < len(coast_ij[0]):
                            map[row, col + i] = spec
                        if col - i > 0:
                            map[row, col - i] = spec
                if coast_ij[row, col] == 0.8:
                    map[row, col] = spec
                    for i in range(size):
                        if row + i < len(coast_ij[:, 0]):
                            map[row + i, col] = spec
                        if row - i > 0:
                            map[row - i, col] = spec

        return map

    def add_poles(self, map, spec=3.0, sea_north=25, cont_north=41, sea_south=25, cont_south=41):
        """
        Add polar ice caps. Call after add_coast but before add_clouds.

        Parameters give pixel rows from each pole edge for sea ice and
        continental ice extents.
        """
        map[:sea_north] = spec
        map[-sea_south:] = spec
        map[sea_north:cont_north][map[sea_north:cont_north] == 1.0] = spec
        map[-cont_south:-sea_south][map[-cont_south:-sea_south] == 1.0] = spec
        return map

    def get_specmap(
        self,
        new_ydeg=None,
        new_wav=None,
        new_smoothing=None,
        new_scalar_wav=None,
        plot_scalar=True,
        return_map_only=True,
    ):
        """
        Compute the full multi-wavelength starry map from the surface map.

        Returns the fullspecmap (and optionally additional diagnostics).
        """
        if new_ydeg is not None:
            self.ydeg = new_ydeg
        if new_wav is not None:
            self.wav = new_wav
        if new_smoothing is not None:
            self.smoothing = new_smoothing
        if new_scalar_wav is not None:
            self.scalar_wav = new_scalar_wav

        map_refl_spec = np.ones((len(self.wav), self.map.shape[0], self.map.shape[1]))
        specmap_y = np.zeros((len(self.wav), (self.ydeg + 1) ** 2))
        specmap_y_shift = np.zeros((len(self.wav), (self.ydeg + 1) ** 2))
        minpix = np.zeros((len(self.wav)))
        maxpix = np.zeros((len(self.wav)))
        amps = np.zeros((len(self.wav)))

        for wl in range(len(self.wav)):
            (
                map_refl_spec[wl],
                specmap_y[wl],
                specmap_y_shift[wl],
                self.specmap_p[wl],
                minpix[wl],
                maxpix[wl],
                amps[wl],
            ) = self._comp_scalarmap(wl)

            if plot_scalar and self.wav[wl] <= self.scalar_wav and self.wav[wl + 1] > self.scalar_wav:
                fig = plt.figure()
                ax = plt.axes()
                im = ax.imshow(np.flipud(map_refl_spec[wl]))
                plt.title(
                    "Albedo distribution at {} um".format(
                        np.round(self.scalar_wav, decimals=2)
                    ),
                    fontsize=18,
                )
                cax = fig.add_axes(
                    [
                        ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height,
                    ]
                )
                plt.colorbar(im, cax=cax).set_label(label="Albedo", size=16)
                plt.show()

                self.scalarmap.amp = amps[wl]
                self.scalarmap[1:, :] = specmap_y_shift[wl][1:] / amps[wl]

            self.fullspecmap[1:, :, wl] = specmap_y_shift[wl][1:] / amps[wl]
            self.fullspecmap.amp[wl] = amps[wl] * 0.4

        if return_map_only:
            return self.fullspecmap
        else:
            return (
                self.fullspecmap,
                self.scalarmap,
                map_refl_spec,
                specmap_y,
                specmap_y_shift,
                self.specmap_p,
                minpix,
                maxpix,
                amps,
            )

    def get_scalarmap(self, wl, new_ydeg=None, plot_map=True):
        """Return the starry scalar map at wavelength wl (microns)."""
        if new_ydeg is not None:
            self.ydeg = new_ydeg

        i = np.abs(self.wav - wl).argmin()
        image, _, y, _, _, _, amp = self._comp_scalarmap(i)

        if plot_map:
            plt.imshow(np.flipud(image))
            plt.colorbar().set_label(label="Albedo", size=16)
            plt.title(
                "Albedo distribution at {} um".format(np.round(wl, decimals=3)), fontsize=18
            )
            plt.show()

        self.scalarmap.amp = amp
        self.scalarmap[1:, :] = y[1:] / amp
        return self.scalarmap

    def _comp_scalarmap(self, wav_index):
        """Internal: compute spherical harmonic map for a single wavelength index."""
        i = wav_index
        self.scalarmap.reset()

        image = self.map.copy()
        image[image == self.cont_id] = self.cont_spec[i]
        image[image == self.ocean_id] = self.ocean_spec[i]
        if self.cloud_id is not None:
            image[image == self.cloud_id] = self.cloud_spec[i]
        if self.snow_id is not None:
            image[image == self.snow_id] = self.snow_spec[i]
        if self.coast_id is not None:
            image[image == self.coast_id] = self.coast_spec[i]

        self.scalarmap.load(image, smoothing=self.smoothing, force_psd=True)
        specmap_y = self.scalarmap.y

        _, _, Y2P, P2Y, _, _ = self.scalarmap.get_pixel_transforms()
        p = Y2P.dot(self.scalarmap.y)

        # Pixel stretch to match max/min albedo
        maxi = np.nanmax(
            (self.cont_spec[i], self.ocean_spec[i], self.cloud_spec[i], self.snow_spec[i], self.coast_spec[i])
        )
        mini = np.nanmin(
            (self.cont_spec[i], self.ocean_spec[i], self.cloud_spec[i], self.snow_spec[i], self.coast_spec[i])
        )

        p = p / np.max(p) * (maxi - mini) + np.min(p)

        # Pixel shift to set minimum value
        shift = np.min(p) - mini
        p = p - shift

        specmap_p = p
        y = P2Y.dot(p)
        specmap_y_shift = y
        minpix = np.min(p)
        maxpix = np.max(p)
        amps = y[0]

        return image, specmap_y, specmap_y_shift, specmap_p, minpix, maxpix, amps


# ---------------------------------------------------------------------------
# Planet class
# ---------------------------------------------------------------------------

class Planet:
    """
    Exoplanet with spectral map and orbital/physical parameters.

    Wraps a SpectralMap and a scalar map into a starry System, and provides
    helpers for computing orbital positions and rotation angles.

    Default parameters are set to Proxima Centauri b.
    """

    def __init__(
        self,
        specmap,
        scalarmap,
        Rp=1.07 * 6371,           # Planet radius in km
        Mp=0.0,                    # Planet mass in Earth masses
        Rstar=0.141 * 695508,      # Star radius in km
        Mstar=None,                # Star mass in kg (derived from Kepler's 3rd law if None)
        star_spec=None,            # Stellar spectrum array
        semi_a=0.04856 * 150e6,    # Semi-major axis in km
        dist=1.30197,              # Distance to system in parsecs
        phase_init=90.0,           # Initial phase in degrees
        theta_init=0.0,            # Initial rotation angle in degrees
        prot=11.1868,              # Rotational period in days
        porb=11.1868,              # Orbital period in days
        obl=0.0,                   # Obliquity in degrees
        p_inc=90.0,                # Planet map inclination in degrees
        orb_inc=90.0,              # Orbital inclination in degrees
        ecc=0.0,                   # Orbital eccentricity
        omega=0.0,                 # Longitude of ascending node in degrees
    ):
        self.specmap = specmap
        self.scalarmap = scalarmap
        self.Rp = Rp
        self.Mp = Mp
        self.Rstar = Rstar
        self.star_spec = star_spec
        self.semi_a = semi_a
        self.dist = dist
        self.prot = prot
        self.porb = porb
        self.orb_inc = orb_inc
        self.ecc = ecc
        self.omega = omega

        self._obl = obl
        self._p_inc = p_inc
        self._phase_init = phase_init
        self._theta_init = theta_init

        # Derive stellar mass from Kepler's third law
        Mstar_kep = (
            (self.semi_a * 1000.0) ** 3
            / ((porb * 24.0 * 60.0 * 60.0) ** 2)
            * 4 * np.pi ** 2
            / 6.6743e-11
            - Mp
        )
        if Mstar is None:
            self.Mstar = Mstar_kep
        elif not np.isclose(Mstar, Mstar_kep):
            print(
                "Stellar mass is inconsistent with Kepler's third law. "
                "Using {} kg instead of {} kg.".format(Mstar_kep, Mstar)
            )
            self.Mstar = Mstar_kep
        else:
            self.Mstar = Mstar

        self.star_pos = starry.Primary(
            starry.Map(ydeg=1),
            r=Rstar,
            m=self.Mstar,
            length_unit=u.km,
            time_unit=u.day,
            mass_unit=u.kg,
        )
        self.planet_pos = starry.Secondary(
            starry.Map(ydeg=1, obl=obl, inc=p_inc),
            r=Rp,
            m=Mp,
            a=semi_a,
            prot=prot,
            omega=omega,
            inc=orb_inc,
            t0=0.0 - (phase_init - 180.0) / 360.0 * porb,
            ecc=ecc,
            time_unit=u.day,
            length_unit=u.km,
            mass_unit=u.Mearth,
        )
        self.system = starry.System(self.star_pos, self.planet_pos)

        self.scalarmap.obl = self._obl
        self.scalarmap.inc = self._p_inc
        self.specmap.obl = self._obl
        self.specmap.inc = self._p_inc

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, value):
        self._obl = value
        self.planet_pos.obl = value
        self.specmap.obl = value
        self.scalarmap.obl = value

    @property
    def p_inc(self):
        return self._p_inc

    @p_inc.setter
    def p_inc(self, value):
        self._p_inc = value
        self.planet_pos.inc = value
        self.specmap.inc = value
        self.scalarmap.inc = value

    @property
    def theta_init(self):
        return self._theta_init

    @theta_init.setter
    def theta_init(self, value):
        self._theta_init = value

    @property
    def phase_init(self):
        return self._phase_init

    @phase_init.setter
    def phase_init(self, value):
        self._phase_init = value
        self.planet_pos = starry.Secondary(
            starry.Map(ydeg=1, obl=self.obl, inc=self.p_inc),
            r=self.Rp,
            m=self.Mp,
            a=self.semi_a,
            prot=self.prot,
            omega=self.omega,
            inc=self.orb_inc,
            t0=0.0 - (value - 180.0) / 360.0 * self.porb,
            ecc=self.ecc,
            time_unit=u.day,
            length_unit=u.km,
            mass_unit=u.Mearth,
        )
        self.system = starry.System(self.star_pos, self.planet_pos)

    def get_system_positions(self, time):
        """
        Compute planet position vectors at given times (in minutes).

        Returns x, y, z in units of Rp, and angular separation in radians.
        """
        xp, yp, zp = self.system.position(time / 24.0 / 60.0)  # days

        x = -(xp[1] - xp[0]) / self.Rp
        y = (yp[1] - yp[0]) / self.Rp
        z = -(zp[1] - zp[0]) / self.Rp

        ang_sep = np.sqrt(x ** 2 + y ** 2) * self.Rp / (self.dist * 30856775814671.914)

        return x, y, z, ang_sep

    def get_theta(self, time):
        """Rotational angle (degrees) at observation times (minutes)."""
        return self._theta_init + time / (self.prot * 24 * 60.0) * 360.0

    def get_phase(self, time):
        """Orbital phase (degrees) at observation times (minutes)."""
        return self._phase_init + time / (self.porb * 24 * 60.0) * 360.0


# ---------------------------------------------------------------------------
# ReflectObservation class
# ---------------------------------------------------------------------------

class ReflectObservation:
    """
    Simulates a direct-imaging spectroscopic observation of a planet.

    Integrates the planet's reflected flux over exposure times, computes
    realistic noise (detector + sky + thermal background), and generates
    synthetic observations.

    Parameters
    ----------
    Telescope : obs_noise.Telescope
        Telescope hardware model.
    Planet : Planet
        Planet object with specmap, scalarmap, and orbital parameters.
    wav : array
        Wavelength grid (microns).
    rpow : float
        Spectral resolving power R = λ/Δλ.
    obs_per_night : int
        Number of exposures per night.
    num_nights : int
        Number of observing nights.
    texp : float
        Exposure time per integration in minutes.
    ndit : int
        Number of dithered sub-exposures per texp.
    cadence : float
        Time between exposure starts in minutes.
    airmass : float
        Observing airmass.
    T_bg : float
        Background temperature in K.
    comp_time : float
        Time step for internal starry flux evaluation in minutes.
    to_plot : bool
        Whether to generate diagnostic plots by default.
    """

    def __init__(
        self,
        Telescope,
        Planet,
        wav,
        rpow=100000,
        obs_per_night=30,
        num_nights=1,
        texp=30.0,
        ndit=1,
        cadence=60.0,
        airmass=1.0,
        T_bg=283.0,
        comp_time=1.0,
        to_plot=True,
    ):
        self._Planet = Planet
        self.telescope = Telescope
        self.wav = wav
        self.rpow = rpow
        self.obs_per_night = obs_per_night
        self.num_nights = num_nights
        self.texp = texp
        if texp > 30 * ndit:
            self.ndit = int(np.floor(texp / 30.0) + 1)
            print(
                f"Exposure time exceeds 30 minutes. "
                f"Number of dithered exposures will be {self.ndit}."
            )
        else:
            self.ndit = ndit

        self.cadence = cadence
        if obs_per_night * cadence > (12 * 60.0):
            print(
                "Warning! Observation time per night is greater than 12 hours. "
                "Consider breaking into multiple nights for ground-based observations."
            )
        if cadence < texp:
            print(
                "Warning! Cadence is less than exposure time. "
                "Correct if you want meaningful results."
            )

        self.airmass = airmass
        self.T_bg = T_bg
        self.comp_time = comp_time
        self.to_plot = to_plot

        self.sky_bg = np.array(
            [[0.36, 0.44, 0.55, 0.64, 0.8, 1.05, 1.25, 1.65, 2.16],
             [22.50, 22.50, 21.8, 21.5, 20.5, 20.5, 20.0, 19.5, 19.5]]
        )
        self.alpha_atm_trans = np.array(
            [[0.36, 0.44, 0.55, 0.70, 0.90, 1.0, 1.25, 1.65, 2.16, 2.60],
             [0.67, 0.83, 0.90, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0]]
        )

        # Total observation time in minutes
        self.total_obs_time = (
            (self.num_nights - 1) * 24 * 60 + self.obs_per_night * self.cadence
        )
        self.num_eval = int(self.total_obs_time / comp_time)

        self.t_obs = np.zeros((num_nights, obs_per_night))
        for night in range(num_nights):
            self.t_obs[night] = np.arange(
                night * 24.0 * 60.0,
                night * 24.0 * 60.0 + obs_per_night * cadence,
                cadence,
            )

        self.t_eval = np.linspace(0, self.total_obs_time, self.num_eval)

        self.theta = self.Planet.get_theta(self.t_eval)
        self.phase = self.Planet.get_phase(self.t_eval)
        self.x, self.y, self.z, self.ang_sep = self.Planet.get_system_positions(self.t_eval)

    @property
    def Planet(self):
        return self._Planet

    @Planet.setter
    def Planet(self, value):
        self._Planet = value
        self.theta = self._Planet.get_theta(self.t_eval)
        self.phase = self._Planet.get_phase(self.t_eval)
        self.x, self.y, self.z, self.ang_sep = self._Planet.get_system_positions(self.t_eval)

    def compute_flux(self):
        """Compute planet reflected flux and perfect-sphere flux at all t_eval times."""
        flux = (
            self.Planet.specmap.flux(
                theta=self.theta,
                xs=self.x,
                ys=self.y,
                zs=self.z,
                rs=self.Planet.Rstar / self.Planet.Rp,
            )
            * self.Planet.star_spec
        )

        planet_perfect = starry.Map(
            ydeg=self.Planet.specmap.ydeg, reflected=True, nw=len(self.wav), wav=self.wav
        )
        planet_perfect.amp = 1.0
        planet_perfect.obl = self.Planet.obl
        planet_perfect.inc = self.Planet.p_inc
        flux_perfect = (
            planet_perfect.flux(
                theta=self.theta,
                xs=self.x,
                ys=self.y,
                zs=self.z,
                rs=self.Planet.Rstar / self.Planet.Rp,
            )
            * self.Planet.star_spec
        )

        return flux, flux_perfect

    def integrate_over_texp(self):
        """
        Integrate the planet flux over each exposure window.

        Returns
        -------
        int_flux : (num_nights, obs_per_night, nwav) array
        int_stellar : (num_nights, obs_per_night, nwav) array
        int_flux_perfect : (num_nights, obs_per_night, nwav) array
        obs_params : dict with keys x, y, z, theta, ang_sep, obs_phase
        """
        int_flux = np.zeros((self.num_nights, self.obs_per_night, len(self.wav)))
        int_stellar = np.zeros((self.num_nights, self.obs_per_night, len(self.wav)))
        int_flux_perfect = np.zeros((self.num_nights, self.obs_per_night, len(self.wav)))

        flux, flux_perfect = self.compute_flux()

        if self.num_nights == 1:
            self.obs_params = {
                "x": np.zeros(2),
                "y": np.zeros(2),
                "z": np.zeros(2),
                "theta": np.zeros(2),
                "ang_sep": np.zeros(2),
                "obs_phase": np.zeros(2),
            }
            start = int(self.t_obs[0] / self.comp_time)
            end = start + int(self.texp / self.comp_time) - 1
            int_flux[0] = integrate.simpson(flux[start:end], dx=self.comp_time * 60.0, axis=0)
            int_stellar[0] = self.Planet.star_spec * (end - start) * self.comp_time * 60.0
            int_flux_perfect[0] = integrate.simpson(
                flux_perfect[start:end], dx=self.comp_time * 60.0, axis=0
            )

            self.obs_params["x"][0] = self.x[start]
            self.obs_params["y"][0] = self.y[start]
            self.obs_params["z"][0] = self.z[start]
            self.obs_params["theta"][0] = self.theta[start]
            self.obs_params["ang_sep"][0] = self.ang_sep[start]
            self.obs_params["obs_phase"][0] = self.phase[start]

            self.obs_params["x"][1] = self.x[end]
            self.obs_params["y"][1] = self.y[end]
            self.obs_params["z"][1] = self.z[end]
            self.obs_params["theta"][1] = self.theta[end]
            self.obs_params["ang_sep"][1] = self.ang_sep[end]
            self.obs_params["obs_phase"][1] = self.phase[end]

        return int_flux, int_stellar, int_flux_perfect, self.obs_params

    def noise_det(self):
        """Detector noise over spectrometer aperture (electrons)."""
        texp_s = self.texp * 60.0  # convert minutes → seconds
        return np.sqrt(
            self.telescope.n_readout_pix_per_res()
            * (
                self.ndit * self.telescope.read_noise ** 2 / self.telescope.pix_bin_fac
                + self.telescope.dar_curr / 3600.0 * texp_s
            )
        )

    def sky_bg_w_airmass(self):
        """Sky background surface brightness (mag/arcsec^2) with airmass correction."""
        bg = np.interp(self.wav, self.sky_bg[0], self.sky_bg[1])
        return bg - 0.4 * (self.airmass - 1)

    def atmospheric_trans(self):
        """Atmospheric transmission at the observing airmass."""
        alpha = np.interp(self.wav, self.alpha_atm_trans[0], self.alpha_atm_trans[1])
        return alpha ** self.airmass

    def total_eff(self):
        """Combined telescope + instrument + atmosphere throughput."""
        telescope_eff = self.telescope.telescope_eff(self.wav)
        return telescope_eff * self.telescope.inst_eff * self.atmospheric_trans()

    def bg_flux_in_spec_aper(self, print_breakdown=False, plot_breakdown=False):
        """Background flux (sky + thermal) in the spectrometer aperture (e-/s)."""
        sigma_sky = (
            10 ** ((16.85 - self.sky_bg_w_airmass()) / 2.5) / self.rpow
        )  # photons/cm^2/s/arcsec^2
        sigma_therm = (
            1.4e12 * self.telescope.emis_bg * np.exp(-14388.0 / (self.wav * self.T_bg))
        ) / (
            self.rpow * self.wav ** 3
        )  # photons/cm^2/s/arcsec^2
        bg_factor = (
            self.total_eff()
            * self.telescope.telescope_area()
            * np.pi / 4.0
            * self.telescope.d_spec_aper ** 2
        )
        sigma_sky = sigma_sky * bg_factor
        sigma_therm = sigma_therm * bg_factor

        if print_breakdown:
            print(f"Sky background: {sigma_sky}")
            print(f"Thermal background: {sigma_therm}")
        if plot_breakdown:
            fig, ax = plt.subplots()
            ax.plot(self.wav, sigma_sky.flatten(), label="Sky background")
            ax.plot(self.wav, sigma_therm.flatten(), label="Thermal background")
            ax.legend()
            ax.set_ylabel("Background Flux (e-/s)")
            ax.set_xlabel("Wavelength (um)")
            plt.title("Background Flux Breakdown")
            plt.show()

        return sigma_sky + sigma_therm

    def bg_noise(self):
        """Background noise in spectrometer aperture per resolution element (electrons)."""
        texp_s = self.texp * 60.0
        return np.sqrt(self.bg_flux_in_spec_aper() * texp_s)

    def noise_per_res_elem_from_nobj(self, flux=None):
        """Total noise per resolution element (electrons)."""
        return np.sqrt(self.noise_det() ** 2 + self.bg_noise() ** 2 + self.nobj(flux))

    def nobj_from_mag(self):
        """Object photon count from a magnitude-specified target."""
        return (
            self.telescope.slit_eff_interp(self.wav)
            * self.total_eff()
            * self.telescope.telescope_area()
            * self.texp * 60.0
            / self.rpow
            * 10 ** ((16.85 - self.mag) / 2.5)
        )

    def nobj(self, flux=None):
        """Object photon count per resolution element in the exposure."""
        if flux is None:
            flux, _, _, _ = self.integrate_over_texp()
        return (
            self.telescope.slit_eff_interp(self.wav)
            * self.total_eff()
            * self.telescope.telescope_area()
            / self.rpow
            * flux
        )

    def snr(self, print_breakdown=False, plot_breakdown=False):
        """Signal-to-noise ratio per resolution element."""
        return np.sqrt(self.nobj() / self.noise_per_res_elem_from_nobj())

    def generate_obs(self, if_plot=True):
        """
        Generate a synthetic observation.

        Returns flux, noisy obs, noise, albedo spectra, contrast spectra,
        and obs_params dict.
        """
        flux, flux_stellar, flux_perfect, obs_params = self.integrate_over_texp()
        noise = self.noise_per_res_elem_from_nobj(flux=flux)
        obs_data = flux + np.random.normal(0, noise, flux.shape)
        albedo = flux / flux_perfect
        albedo_noise = obs_data / flux_perfect
        albedo_err = noise / flux_perfect
        contrast = flux / flux_stellar
        contrast_noise = obs_data / flux_stellar
        contrast_err = noise / flux_stellar

        if if_plot:
            self.plot_obs(
                flux, obs_data, noise,
                albedo, albedo_noise, albedo_err,
                contrast, contrast_noise, contrast_err,
                obs_params,
            )

        return (
            flux, obs_data, noise,
            albedo, albedo_noise, albedo_err,
            contrast, contrast_noise, contrast_err,
            obs_params,
        )

    def plot_obs(
        self, flux, obs_data, noise,
        albedo, albedo_noise, albedo_err,
        contrast, contrast_noise, contrast_err,
        obs_params,
    ):
        """Plot a 5×2 diagnostic figure for a single-night observation."""
        if self.num_nights == 1:
            fig, axs = plt.subplots(5, 2, figsize=(20, 30))

            ax = axs[0][0]
            ax.plot(self.wav, flux[0][0])
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Flux (pixels per resolution element)")
            ax.set_ylim(np.min(flux[0][0]) * 0.8, np.max(flux[0][0]) * 1.2)
            ax.set_title("Simulated Observation (No Noise)")

            ax = axs[0][1]
            ax.errorbar(self.wav, obs_data[0][0], yerr=noise[0][0], fmt="o")
            ax.plot(self.wav, flux[0][0], color="black", alpha=0.5)
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Flux (pixels per resolution element)")
            ax.set_ylim(np.min(flux[0][0]) * 0.8, np.max(flux[0][0]) * 1.2)
            ax.set_title("Simulated Observation (With Noise)")

            ax = axs[1][0]
            ax.plot(self.wav, albedo[0][0])
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Apparent Albedo")
            ax.set_ylim(np.min(albedo[0][0]) * 0.8, np.max(albedo[0][0]) * 1.2)
            ax.set_title("Simulated Reflection Spectrum (No Noise)")

            ax = axs[1][1]
            ax.errorbar(self.wav, albedo_noise[0][0], yerr=albedo_err[0][0], fmt="o")
            ax.plot(self.wav, albedo[0][0], color="black", alpha=0.5)
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Apparent Albedo")
            ax.set_ylim(np.min(albedo[0][0]) * 0.8, np.max(albedo[0][0]) * 1.2)
            ax.set_title("Simulated Reflection Spectrum (With Noise)")

            ax = axs[2][0]
            ax.plot(self.wav, contrast[0][0])
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Contrast")
            ax.set_ylim(np.min(contrast[0][0]) * 0.8, np.max(contrast[0][0]) * 1.2)
            ax.set_title("Contrast Spectrum (No Noise)")

            ax = axs[2][1]
            ax.errorbar(self.wav, contrast_noise[0][0], yerr=contrast_err[0][0], fmt="o")
            ax.plot(self.wav, contrast[0][0], color="black", alpha=0.5)
            ax.set_xlabel("Wavelength (micron)")
            ax.set_ylabel("Contrast")
            ax.set_ylim(np.min(contrast[0][0]) * 0.8, np.max(contrast[0][0]) * 1.2)
            ax.set_title("Contrast Spectrum (With Noise)")

            ax = axs[3][0]
            self.Planet.scalarmap.show(
                theta=obs_params["theta"][0],
                xs=obs_params["x"][0],
                ys=obs_params["y"][0],
                zs=obs_params["z"][0],
                ax=ax,
            )
            ax.set_title("Illumination of Planet - Start of night")

            ax = axs[3][1]
            self.Planet.scalarmap.show(
                theta=obs_params["theta"][-1],
                xs=obs_params["x"][-1],
                ys=obs_params["y"][-1],
                zs=obs_params["z"][-1],
                ax=ax,
            )
            ax.set_title("Illumination of Planet - End of night")

            ax = axs[4][0]
            ax.plot(
                np.linspace(0, self.total_obs_time, 20),
                np.linspace(obs_params["ang_sep"][0], obs_params["ang_sep"][-1], 20),
            )
            ax.axhline(
                2 * self.wav[0] * 1e-6 / self.telescope.diam,
                color="red",
                linestyle="--",
                label="2λ/D",
            )
            ax.axhline(
                3 * self.wav[0] * 1e-6 / self.telescope.diam,
                color="blue",
                linestyle="--",
                label="3λ/D",
            )
            ax.set_xlabel("Obs Time (mins)")
            ax.set_ylabel("Angular Separation")
            ax.set_title("Angular Separation")
            ax.legend()

            ax = axs[4][1]
            ax.plot(
                np.linspace(0, self.total_obs_time, 20),
                np.linspace(obs_params["obs_phase"][0], obs_params["obs_phase"][-1], 20),
                label="Phase Angle",
            )
            ax.plot(
                np.linspace(0, self.total_obs_time, 20),
                np.linspace(obs_params["theta"][0], obs_params["theta"][-1], 20),
                label="Rotation Angle",
            )
            ax.set_xlabel("Obs Time (mins)")
            ax.set_ylabel("Angle")
            ax.set_title("Phase and Rotation Angles")
            ax.legend()

            plt.show()

    def viz_obs(self, num_maps=2, projection="ortho", title=None, ax=None):
        """Visualise the planet illumination geometry at observation start/end."""
        if num_maps == 2:
            fig, axs = plt.subplots(1, 2)
            axs[0].set_title("Illumination of Planet - Start of night")
            self.Planet.scalarmap.show(
                theta=self.theta[0],
                xs=self.x[0],
                ys=self.y[0],
                zs=self.z[0],
                ax=axs[0],
                projection=projection,
            )
            axs[1].set_title("Illumination of Planet - End of night")
            self.Planet.scalarmap.show(
                theta=self.theta[-1],
                xs=self.x[-1],
                ys=self.y[-1],
                zs=self.z[-1],
                ax=axs[1],
                projection=projection,
            )
            plt.show()

        elif num_maps == 1:
            if ax is None:
                fig, ax = plt.subplots()
            self.Planet.scalarmap.show(
                theta=self.theta[0],
                xs=self.x[0],
                ys=self.y[0],
                zs=self.z[0],
                ax=ax,
                projection=projection,
            )
            ax.set_title(title if title is not None else "Illumination of Planet - Start of night")

        else:
            idx = np.linspace(0, len(self.theta) - 1, num_maps, dtype=int)
            fig, axs = plt.subplots(1, num_maps)
            for i in range(num_maps):
                axs[i].set_title(
                    "Illumination of Planet - {} minutes after start of obs".format(
                        self.t_eval[idx[i]]
                    )
                )
                self.Planet.scalarmap.show(
                    theta=self.theta[idx[i]],
                    xs=self.x[idx[i]],
                    ys=self.y[idx[i]],
                    zs=self.z[idx[i]],
                    ax=axs[i],
                    projection=projection,
                )
            plt.show()
