"""
earth_kofman2024_reflection.py
==============================
Recreates the 9 Earth reflection spectra at quadrature from Kofman et al.
(2024) using the starry spherical-harmonic reflect model.

Reference
---------
Kofman et al. (2024), "The Pale Blue Dot: Using the Planetary Spectrum
Generator to Simulate Signals from Hyperrealistic Exoplanet Surfaces",
The Planetary Science Journal, 5, 197.

Geometry (all 9 configurations share the same geometry)
---------
  • Quadrature: star along +x, observer along +z
  • Summer solstice: Earth's north pole tilted 23.44° toward the star
    → p_inc = 90° (spin axis in plane of sky)
    → obl   = +23.44° (pole rotated from +y toward +x, i.e., toward star)
  • Sub-stellar latitude fixed at +23.44°N for all 9 cases

Sub-stellar longitude → theta mapping
--------------------------------------
With inc=90° and observer at +z, the default orientation (theta=0) places
the map centre (longitude 0°) facing the observer.  The star at +x is 90°
to the right (viewed from the north pole = +y direction).

By tracing the rotation (positive theta = CCW about the spin axis when viewed
from the north), longitude L_sub faces +x when:

    theta = 90° − L_sub                  (degrees)

with L_sub = (12 − UTC_hours) × 15°     (degrees East)

so equivalently:  theta = 15 × UTC_hours − 90°

Verification:
  UTC 00:00 → L_sub = +180° (Pacific)   → theta = −90°
  UTC 06:00 → L_sub =  +90° (Indian Ocean) → theta =  0°
  UTC 12:00 → L_sub =   0°  (Greenwich)  → theta = +90°
  UTC 18:00 → L_sub = −90° (Americas)   → theta = +180°
  UTC 19:09 → L_sub ≈ −107° (E. Pacific) → theta ≈ +197°  ← paper mentions
  UTC 22:25 → L_sub ≈ −156° (C. Pacific) → theta ≈ +246°  ← paper mentions

Usage
-----
  python earth_kofman2024_reflection.py

Outputs
-------
  earth_kofman_flux.npy    – (9, nwav) raw reflected flux array
  earth_kofman_albedo.npy  – (9, nwav) albedo normalised to perfect sphere
  earth_kofman_wav.npy     – (nwav,) wavelength grid in microns
  earth_kofman_spectra.png – quick-look plot

DATA ACQUISITION NOTES  (see README section at bottom of this file)
-------------------
For high-fidelity reproduction of the paper you need five external datasets:
  1. MODIS MCD12C1  – annual land cover (5 categories at 2°×2.5°)
  2. MODIS MOD10CM  – monthly snow cover (June 2022)
  3. NSIDC EASE-Grid sea-ice concentration (June 21 2022)
  4. USGS Spectral Library v7 (Kokaly et al. 2017)  – surface albedos
  5. MERRA-2 M2I3NVASM  – 3-hourly cloud fraction (June 21 2022)

Without those files this script falls back to:
  • cartopy Natural Earth coastlines for the land/ocean mask (realistic)
    or a simple geometric approximation if cartopy is unavailable
  • reflect_dev_dir ASTER/MODIS analog spectra (already bundled)
  • Fixed-location random cloud patches
"""

from __future__ import annotations

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive; change to "TkAgg" etc. if needed
import matplotlib.pyplot as plt

# ── starry configuration must be set BEFORE import ────────────────────────────
import starry
starry.config.lazy  = False
starry.config.quiet = True

# ── Import reflect_dev_dir tools ───────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_PKG  = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from starry.extensions.reflect_dev_dir import (
    SpectralMap,
    load_surface_spectra,
    load_other_spectra,
    build_continent_spectra,
    wav_grid_from_R,
)

try:
    from spectres import spectres
except ImportError:
    raise ImportError("spectres is required: pip install spectres")


# ══════════════════════════════════════════════════════════════════════════════
#  Physical / orbital constants
# ══════════════════════════════════════════════════════════════════════════════

R_EARTH_KM   = 6371.0          # Earth equatorial radius [km]
R_SUN_KM     = 695700.0        # Solar radius [km]
AU_KM        = 1.495978707e8   # 1 AU [km]
OBL_EARTH    = 23.44           # Obliquity at summer solstice [deg]
P_INC_EARTH  = 90.0            # Spin-axis inclination from line-of-sight [deg]
                                # 90° = equator-on / spin axis in plane of sky

# Star geometry (units of planet radii), quadrature: star along +x
RS_RP   = R_SUN_KM  / R_EARTH_KM   # stellar radius in Earth radii ≈ 109.2
DIST_RP = AU_KM     / R_EARTH_KM   # star–planet distance in Earth radii ≈ 23 484

# ── Nine UTC times (decimal hours) on 2022-Jun-21 ─────────────────────────────
# Kofman et al. (2024) run 9 PSG simulations that track MERRA-2's 3-hourly
# cadence (0000, 0300 … 2100 UTC) + one carry-over at 0000 Jun-22 for a full
# 24-hour rotation.  The exact DSCOVR/EPIC observation times picked to match
# each MERRA-2 snapshot are available in the paper's supplementary repository
# (https://psg.gsfc.nasa.gov or the journal's online article).
# Replace the values below with the precise times from the paper if available.
UTC_HOURS = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def subsolar_longitude(utc_hours: float) -> float:
    """
    Approximate sub-solar longitude (°E, range −180 to +180) at UTC_hours.

    At UTC 12:00 the sub-solar point is over the Greenwich meridian (0°E).
    Earth rotates 15°/hour westward, so the sub-solar longitude decreases by
    15° per hour past noon.

    Parameters
    ----------
    utc_hours : float
        UTC time in decimal hours (0–24).

    Returns
    -------
    float  Sub-solar longitude in degrees East.
    """
    lon = (12.0 - utc_hours) * 15.0
    # Wrap to (−180, +180]
    return (lon + 180.0) % 360.0 - 180.0


def utc_to_theta(utc_hours: float) -> float:
    """
    Map rotation angle (degrees) that places the sub-stellar longitude toward
    the star at +x, for a starry map with inc=90°.

    Derivation: theta = 90° − L_sub = 90° − (12 − utc)*15 = 15*utc − 90°

    Parameters
    ----------
    utc_hours : float

    Returns
    -------
    float  theta in degrees.
    """
    return 15.0 * utc_hours - 90.0


def make_land_ocean_mask(nlat: int = 91, nlon: int = 144) -> np.ndarray:
    """
    Return a (nlat × nlon) binary land/ocean mask.

    Row 0 = North Pole (+90°), row nlat−1 = South Pole (−90°).
    Column 0 = −180°E, column nlon−1 ≈ +180°E (West-to-East, not wrapping).

    Values: 1.0 = land (continent), 0.0 = ocean.

    Tries cartopy Natural Earth 110 m first; falls back to a simple
    geometric approximation if cartopy is unavailable.

    Paper resolution: 91 × 144 bins (≈ 2° × 2.5° per pixel).
    """
    lats = np.linspace(90.0,  -90.0, nlat)
    lons = np.linspace(-180.0, 177.5, nlon)   # 2.5° step, endpoint=False

    # ── Attempt 1: cartopy Natural Earth land polygons ────────────────────
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import Point
        from shapely.ops import unary_union

        shp = shpreader.natural_earth(
            resolution="110m", category="physical", name="land"
        )
        land_union = unary_union(list(shpreader.Reader(shp).geometries()))

        mask = np.zeros((nlat, nlon), dtype=np.float32)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if land_union.contains(Point(lon, lat)):
                    mask[i, j] = 1.0
        land_pct = mask.mean() * 100
        print(f"  cartopy mask built: {land_pct:.1f}% land "
              f"(Earth ~29%; coarse grid gives ~{land_pct:.0f}%)")
        return mask

    except Exception as err:
        print(f"  cartopy unavailable ({type(err).__name__}); using geometric fallback.")

    # ── Attempt 2: geometric approximation ────────────────────────────────
    # Major continental outlines only – captures the dominant land/ocean
    # contrast needed for the VRE signal and oceanic glint modulation.
    mask = np.zeros((nlat, nlon), dtype=np.float32)
    _CONTINENTS = [
        # (lon_min, lon_max, lat_min, lat_max)  approximate bounding boxes
        (-25,  60,  35,  72),   # Europe
        (-25,  60, -35,  35),   # Africa
        ( 25, 180,  10,  77),   # Asia (Eurasia east half)
        ( 25, 180, -10,  10),   # SE Asia / Maritime continent
        (110, 155, -45,  -8),   # Australia
        (-170,-50,  15,  75),   # North America
        ( -82, -35, -58,  15),  # South America
        ( -70,  50, -85, -65),  # Antarctica (partial)
        ( 50, 180, -85, -65),   # Antarctica (partial)
        (-180,-50, -85, -65),   # Antarctica (partial)
        ( 25,  60,  15,  30),   # Arabian Peninsula
        ( 70, 100,   5,  30),   # Indian subcontinent
        ( 95, 145,  -8,  25),   # Indochina / Malay Peninsula
    ]
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            for (lo0, lo1, la0, la1) in _CONTINENTS:
                if lo0 <= lon <= lo1 and la0 <= lat <= la1:
                    mask[i, j] = 1.0
                    break

    print(f"  Geometric mask built: {mask.mean()*100:.1f}% land "
          "(rough — recommend cartopy or MODIS MCD12C1 for fidelity).")
    return mask


def build_earth_spectra(wav: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build surface reflectance spectra on the wavelength grid `wav`.

    Paper surfaces (Kofman et al. 2024, §2.1)
    ------------------------------------------
    The paper uses USGS spectral library single-scattering albedos (Kokaly
    et al. 2017) with five categories mixed by areal fractions:
      ocean   – Lambertian sea water (+ Cox–Munk glint, not modelled here)
      snow    – fine snow (seasonal + sea ice)
      soil    – bare soil composite
      forest  – deciduous/conifer trees
      grass   – herbaceous vegetation / cropland

    reflect_dev_dir analog
    ----------------------
    The bundled ASTER/MODIS/USGS library provides close proxies:
      ocean  → 'sea'   (seawater USGS+ASTER)
      snow   → 'snow'  (fine snow ASTER)
      forest → 'trees' (deciduous trees ASTER)
      grass  → 'grass' (lawn grass ASTER)
      soil   → weighted mix of 'basalt' + 'granite' + 'sand'

    The continental composite (build_continent_spectra 'cont') pre-mixes:
      0.30×grass + 0.30×trees + 0.09×granite + 0.09×basalt + 0.07×sand
      + 0.15×snow
    This is a reasonable Earth-like mix; adjust weights or use individual
    USGS spectra for higher fidelity once you have the Kokaly et al. library.

    Returns
    -------
    dict with keys 'cont', 'ocean', 'cloud', 'snow'.
    Each value is a 1-D array of reflectance at wavelengths `wav`.
    """
    regions    = load_surface_spectra()
    other_spec = load_other_spectra()
    regions    = build_continent_spectra(regions, other_spec, wav_deep=wav)

    def _resample(key: str) -> np.ndarray:
        d = regions[key]
        if d.ndim == 2:
            return spectres(wav, d[:, 0], d[:, 1], fill=d[0, 1], verbose=False)
        # already on wav grid (build_continent_spectra produces column_stack)
        return d[:, 1] if d.ndim == 2 else d

    # 'cont' is already on wav from build_continent_spectra
    cont_spec  = regions["cont"][:, 1]
    ocean_spec = _resample("sea")
    cloud_spec = _resample("cloud")
    snow_spec  = _resample("snow")

    # Clip to physical range
    for arr in (cont_spec, ocean_spec, cloud_spec, snow_spec):
        np.clip(arr, 0.0, 1.0, out=arr)

    return {
        "cont":  cont_spec,
        "ocean": ocean_spec,
        "cloud": cloud_spec,
        "snow":  snow_spec,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SpectralMap builder
# ══════════════════════════════════════════════════════════════════════════════

def build_earth_spectralmap(
    wav:            np.ndarray,
    ydeg:           int   = 20,
    include_clouds: bool  = True,
    include_snow:   bool  = True,
    smoothing:      float = 1.5,
    cloud_num:      int   = 6,
    cloud_dims:     tuple = (30, 120),
    snow_dims:      tuple = (10, 20, 10, 20),
) -> SpectralMap:
    """
    Build a SpectralMap representing Earth's surface for the Kofman (2024)
    simulations.

    Parameters
    ----------
    wav : array
        Wavelength grid in microns.
    ydeg : int
        Spherical harmonic degree (paper-equivalent: use ≥ 20; higher is
        slower but retains finer land/ocean boundaries).
    include_clouds : bool
        Add stochastic cloud patches.  Set False to reproduce the paper's
        cloud-free spectra first, then True for the cloudy case.
    include_snow : bool
        Add polar ice caps (MOD10CM / NSIDC analog).
    smoothing : float or None
        Passed to starry.Map.load(); reduces Gibbs ringing at coastlines.
        None disables smoothing (sharper but ringing artefacts possible).
    cloud_num : int
        Number of random cloud patches (paper has realistic MERRA-2 clouds;
        this is a stochastic approximation).
    cloud_dims : (min_deg, max_deg)
        Angular diameter range of cloud patches in degrees.
    snow_dims : (sea_N, cont_N, sea_S, cont_S)
        Pixel rows from each pole edge for sea-ice and continental-ice extent.
        Default corresponds roughly to June sea-ice extent.

    Returns
    -------
    SpectralMap  (call .get_specmap() to obtain the starry map object)
    """
    print("Building land/ocean mask …")
    land_mask = make_land_ocean_mask(nlat=91, nlon=144)

    print("Loading surface reflectance spectra …")
    spectra = build_earth_spectra(wav)

    print(f"Constructing SpectralMap (ydeg={ydeg}) …")
    smap = SpectralMap(
        map_image  = land_mask.copy(),
        wav        = wav,
        # ── surface types ──────────────────────────────────────────────────
        cont_id    = 1.0,
        cont_spec  = spectra["cont"],
        ocean_id   = 0.0,
        ocean_spec = spectra["ocean"],
        # ── clouds (stochastic approximation of MERRA-2 coverage) ──────────
        cloud_id   = 2.0   if include_clouds else None,
        cloud_spec = spectra["cloud"] if include_clouds else None,
        cloud_dims = cloud_dims,
        cloud_num  = cloud_num,
        # ── polar ice (NSIDC / MOD10CM analog) ────────────────────────────
        snow_id    = 3.0   if include_snow else None,
        snow_spec  = spectra["snow"]  if include_snow  else None,
        snow_dims  = snow_dims,
        # ── SH decomposition settings ──────────────────────────────────────
        ydeg       = ydeg,
        scalar_wav = 0.7,      # reference wavelength for scalar visualisation
        smoothing  = smoothing,
        roughness  = 0.0,      # Lambertian (Oren-Nayar off), matches paper
    )
    return smap


# ══════════════════════════════════════════════════════════════════════════════
#  Core computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_quadrature_spectra(
    wav:            np.ndarray | None = None,
    utc_hours:      np.ndarray | None = None,
    ydeg:           int   = 20,
    include_clouds: bool  = False,
    include_snow:   bool  = True,
    plot_scalar:    bool  = False,
) -> dict:
    """
    Compute disk-integrated reflected spectra for the 9 Earth orientations
    matching the Kofman et al. (2024) summer-solstice quadrature simulations.

    Parameters
    ----------
    wav : array or None
        Wavelength grid in microns.  Defaults to the paper's R=70 grid over
        0.30–1.00 μm (UV to near-IR, matching the HWO simulation range).
    utc_hours : array or None
        UTC times (decimal hours) for the 9 configurations.  Defaults to
        UTC_HOURS defined at the top of this module.
    ydeg : int
        Spherical harmonic degree (≥ 20 recommended).
    include_clouds : bool
        False → cloud-free case (paper's left panel of Figure 5).
        True  → cloudy case (paper's right panel; uses stochastic patches
                 rather than MERRA-2; see DATA ACQUISITION NOTES).
    include_snow : bool
        Include polar ice caps.
    plot_scalar : bool
        Show the scalar (0.7 μm) map after loading.

    Returns
    -------
    dict with keys
      'wav'          (nwav,)    wavelength grid [μm]
      'flux'         (9, nwav)  raw starry reflected flux [normalised units]
      'albedo'       (9, nwav)  flux / perfect-Lambertian-sphere flux
      'flux_perfect' (nwav,)    reference perfect-sphere flux at quadrature
      'subsolar_lon' (9,)       sub-solar longitude [°E]
      'theta'        (9,)       map rotation angles used [°]
      'utc_hours'    (9,)       UTC times [decimal hours]
      'utc_labels'   list[str]  'HH:MM' labels
    """
    # ── wavelength grid ────────────────────────────────────────────────────
    if wav is None:
        wav = wav_grid_from_R(R=70, wav_min=0.30, wav_max=1.00)
    wav = np.asarray(wav, dtype=float)

    if utc_hours is None:
        utc_hours = UTC_HOURS
    utc_hours = np.asarray(utc_hours, dtype=float)
    n = len(utc_hours)

    # ── build spectral map ─────────────────────────────────────────────────
    smap    = build_earth_spectralmap(
        wav, ydeg=ydeg,
        include_clouds=include_clouds,
        include_snow=include_snow,
    )
    specmap = smap.get_specmap(plot_scalar=plot_scalar)

    # ── set summer-solstice obliquity ──────────────────────────────────────
    # p_inc = 90°: spin axis lies in the plane of sky (edge-on equatorial view)
    # obl   = +23.44°: in starry the obliquity rotates the pole from +y (north
    #          up in sky) toward +x (the star direction at quadrature).
    #          obl > 0 tilts the north pole toward the star → summer solstice.
    specmap.inc = P_INC_EARTH   # 90°
    specmap.obl = OBL_EARTH     # +23.44°

    # ── quadrature geometry ────────────────────────────────────────────────
    xs = float(DIST_RP)     # star in +x direction, distance = 1 AU in Rp units
    ys = 0.0
    zs = 0.0
    rs = float(RS_RP)       # solar radius in Earth-radius units

    # ── reference: perfect Lambertian sphere at quadrature ─────────────────
    ref_map     = starry.Map(ydeg=ydeg, reflected=True, nw=len(wav), wav=wav)
    ref_map.amp = 1.0
    ref_map.inc = P_INC_EARTH
    ref_map.obl = OBL_EARTH
    flux_perfect = np.array(
        ref_map.flux(theta=0.0, xs=xs, ys=ys, zs=zs, rs=rs)
    ).flatten()

    # ── loop over the 9 UTC times ──────────────────────────────────────────
    fluxes       = np.zeros((n, len(wav)))
    subsolar_lons = np.zeros(n)
    thetas        = np.zeros(n)

    print(f"\nComputing {n} reflected-flux configurations …")
    print(f"  {'UTC':>6}  {'Sub-sol lon':>12}  {'theta':>9}  "
          f"{'Dominant hemisphere':>22}")
    print("  " + "─" * 58)

    for i, utc_h in enumerate(utc_hours):
        L_sub          = subsolar_longitude(utc_h)
        theta          = utc_to_theta(utc_h)      # = 90° − L_sub = 15°×utc − 90°
        subsolar_lons[i] = L_sub
        thetas[i]        = theta

        label = f"{int(utc_h % 24):02d}:{int((utc_h % 1) * 60):02d}"
        if -180 < L_sub <= -60:
            hemi = "Americas / E. Pacific"
        elif -60 < L_sub <= 60:
            hemi = "Europe / Africa"
        elif 60 < L_sub <= 150:
            hemi = "Asia / Indian Ocean"
        else:
            hemi = "Pacific Ocean"
        print(f"  {label:>6}  {L_sub:>+11.1f}°  {theta:>+8.1f}°  {hemi:>22}")

        flux_i = specmap.flux(theta=theta, xs=xs, ys=ys, zs=zs, rs=rs)
        fluxes[i] = np.array(flux_i).flatten()

    # ── normalise to perfect sphere ────────────────────────────────────────
    denom  = np.where(flux_perfect > 0, flux_perfect, 1.0)
    albedo = fluxes / denom[np.newaxis, :]

    utc_labels = [
        f"{int(h % 24):02d}:{int((h % 1) * 60):02d}" for h in utc_hours
    ]

    return {
        "wav":          wav,
        "flux":         fluxes,
        "albedo":       albedo,
        "flux_perfect": flux_perfect,
        "subsolar_lon": subsolar_lons,
        "theta":        thetas,
        "utc_hours":    utc_hours,
        "utc_labels":   utc_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_spectra(result: dict, save_path: str | None = None) -> plt.Figure:
    """
    Quick-look plot replicating the style of Figure 5 (middle row) of
    Kofman et al. (2024): albedo vs wavelength for the 9 orientations.
    """
    wav    = result["wav"] * 1000.0   # μm → nm
    albedo = result["albedo"]
    labels = result["utc_labels"]
    lons   = result["subsolar_lon"]
    n      = len(labels)

    colors = plt.cm.plasma(np.linspace(0.10, 0.92, n))

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(n):
        ax.plot(
            wav, albedo[i],
            color=colors[i], lw=1.6,
            label=f"{labels[i]} UTC  (sub-sol lon {lons[i]:+.0f}°)",
        )

    # Mark vegetation red edge region
    ax.axvspan(700, 760, alpha=0.08, color="green", label="VRE region")

    ax.set_xlabel("Wavelength  [nm]", fontsize=12)
    ax.set_ylabel("Relative albedo  (normalised to Lambertian sphere)", fontsize=10)
    ax.set_title(
        "Earth reflection spectra at quadrature — summer solstice 2022\n"
        "9 sub-stellar longitudes  ·  starry SH model  "
        "(Kofman et al. 2024 analog)",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.set_xlim(wav[0], wav[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {save_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── wavelength grid: R=70, 300–1000 nm (matches paper's HWO simulations)
    wav = wav_grid_from_R(R=70, wav_min=0.30, wav_max=1.00)

    print("=" * 64)
    print("  Earth quadrature spectra — Kofman et al. (2024) starry analog")
    print("=" * 64)
    print(f"  λ grid    : {len(wav)} bins, "
          f"{wav[0]*1000:.0f}–{wav[-1]*1000:.0f} nm  (R=70)")
    print(f"  Star pos  : xs = {DIST_RP:.0f} Rp  (1 AU),  "
          f"rs = {RS_RP:.1f} Rp  (1 R_sun)")
    print(f"  Obliquity : {OBL_EARTH}°  (summer solstice, pole toward +x star)")
    print(f"  Clouds    : disabled for cloud-free run  (set include_clouds=True)")
    print()

    # ── Cloud-free case (reproduces paper's left panel of Figure 5) ─────────
    result_cf = compute_quadrature_spectra(
        wav            = wav,
        utc_hours      = UTC_HOURS,
        ydeg           = 20,
        include_clouds = False,   # cloud-free
        include_snow   = True,
        plot_scalar    = False,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(out_dir, "earth_kofman_flux.npy"),    result_cf["flux"])
    np.save(os.path.join(out_dir, "earth_kofman_albedo.npy"),  result_cf["albedo"])
    np.save(os.path.join(out_dir, "earth_kofman_wav.npy"),     result_cf["wav"])
    print("\nArrays saved → earth_kofman_{flux,albedo,wav}.npy")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  {'UTC':>6}  {'Sub-sol lon':>12}  {'theta':>9}  "
          f"{'Mean albedo':>12}  {'Max albedo':>10}")
    print("  " + "─" * 60)
    for i in range(len(result_cf["utc_labels"])):
        print(
            f"  {result_cf['utc_labels'][i]:>6}  "
            f"{result_cf['subsolar_lon'][i]:>+11.1f}°  "
            f"{result_cf['theta'][i]:>+8.1f}°  "
            f"{result_cf['albedo'][i].mean():>12.5f}  "
            f"{result_cf['albedo'][i].max():>10.5f}"
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plot_spectra(
        result_cf,
        save_path=os.path.join(out_dir, "earth_kofman_spectra.png"),
    )
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# DATA ACQUISITION NOTES
# ══════════════════════════════════════════════════════════════════════════════
#
# To reproduce the Kofman et al. (2024) surface assumptions with higher
# fidelity, acquire and ingest the following datasets.
#
# ── 1. Land cover map (MODIS MCD12C1) ─────────────────────────────────────────
#    URL : https://lpdaac.usgs.gov/products/mcd12c1v006/
#    File: MCD12C1.A2022001.061.*hdf  (annual, 2022)
#    Remap the 17-class IGBP scheme to 5 categories (areal fractions per pixel):
#      ocean    → 0  (water bodies, permanent wetlands)
#      grass    → weight for ENF, EBF, DNF, DBF, MF, CROS
#      forest   → weight for savanna, grasslands, permanent wetlands, croplands
#      soil     → weight for barren, urban
#      snow     → weight for snow/ice (supplement with MOD10CM)
#    Rebin from 0.05° (7200×3600) to 2°×2.5° (144×91) matching the paper.
#    Then build a per-pixel continental spectrum as the areal-weighted sum
#    of the individual USGS spectra (see §4 below).
#
# ── 2. Snow / sea-ice masks ────────────────────────────────────────────────────
#    Snow  : MODIS MOD10CM (monthly, June 2022)
#            https://nsidc.org/data/MOD10CM/versions/61
#    Sea ice: NSIDC EASE-Grid (Meier et al. 2021)
#            https://nsidc.org/data/nsidc-0051/versions/2
#    Process: reproject EASE-Grid polar stereographic → lat/lon, merge with
#    MOD10CM. Pixels with fractional snow/ice coverage > 0.5 → snow category.
#
# ── 3. Cloud coverage (MERRA-2 M2I3NVASM) ─────────────────────────────────────
#    URL : https://disc.gsfc.nasa.gov/datasets/M2I3NVASM_5.12.4/summary
#    File: MERRA2_400.inst3_3d_asm_Nv.20220621.nc4  (June 21 2022)
#    The 9 UTC snapshots are 0000, 0300, … 2100 on June 21, + 0000 June 22.
#    Extract CLOUD (cloud fraction) and QI (ice water content).
#    Rebin to 2°×2.5°. For each snapshot build a 2D cloud fraction image
#    (0 = clear, 1 = overcast). Load this as a per-snapshot additional layer
#    in SpectralMap (replace the stochastic cloud_num patches with the real map).
#    Implementation hint: instead of using SpectralMap's random cloud_num,
#    load the cloud image directly:
#
#        # For snapshot i:
#        merged_map = land_mask.copy()
#        merged_map[cloud_frac_i > 0.5] = CLOUD_ID   # overwrite with cloud
#        smap_i = SpectralMap(merged_map, wav, ...)
#        specmap_i = smap_i.get_specmap(...)
#        flux_i = specmap_i.flux(theta=theta_i, xs=xs, ys=ys, zs=zs, rs=rs)
#
# ── 4. USGS Spectral Library v7 (Kokaly et al. 2017) ──────────────────────────
#    URL : https://dx.doi.org/10.5066/F7RR1WDJ
#    The library contains single-scattering albedos (0.35–2.5 μm) for
#    hundreds of natural materials. Key spectra to substitute for the
#    reflect_dev_dir ASTER proxies:
#      grass / cropland → s06_grass_*  or  s07_grass_*
#      forest           → s06_veg_tree_*  (multiple species, average)
#      soil             → s06_soil_*       (multiple soil types, average)
#      snow             → s06_water_snow_*
#      ocean            → s06_water_seawater_*
#    Load as (N×2) wavelength/reflectance arrays, resample to your wav grid
#    with spectres, and pass directly as cont_spec / ocean_spec etc.
#    For the per-pixel areal mixing used in the paper:
#        pixel_spec = (f_grass * grass + f_forest * forest
#                      + f_soil * soil + f_snow * snow)
#    where fractions f_* come from the rebinned MCD12C1 map.
#    This produces a distinct cont_spec per geographic pixel — richer than
#    the single continental composite but requires building a 3D spectral
#    image (nlat × nlon × nwav) before loading into starry.
#
# ── 5. Ocean glint (Cox–Munk) ──────────────────────────────────────────────────
#    The paper includes Cox–Munk specular reflection from ocean surfaces
#    (wind speed 8 m/s, Jackson & Alpers 2010 refinements).  starry uses
#    Lambertian scattering only — there is no built-in glint model.
#    Glint increases the total ocean albedo by ~5–20% near the specular
#    point depending on viewing geometry. At quadrature its contribution is
#    reduced vs. eclipse/full-disk geometry, but it is not zero.
#    Approximate it by increasing the ocean_spec amplitude by a wavelength-
#    independent factor ~1.05–1.15, or leave it out for a cloud-free surface
#    comparison.
#
# ── 6. Exact UTC times from the paper ──────────────────────────────────────────
#    The precise DSCOVR/EPIC observation times selected for each MERRA-2
#    snapshot are listed in the paper's supplementary configuration files at:
#        https://psg.gsfc.nasa.gov   (GlobES database)
#    or the journal's online supplementary material.
#    Update UTC_HOURS at the top of this script with those values.
#    Times 1909 and 2225 (UTC) are identified in the paper's text as two of
#    the nine Pacific-facing snapshots.
