"""
kofman2024_surface_map.py
=========================
Processes MODIS land-cover, snow, and NSIDC sea-ice data into the 5-class
surface fraction maps described in Kofman et al. (2024) and assembles a
``map_image`` for use with :class:`SpectralMap`.

Required environment variable
------------------------------
KOFMAN_DATA_DIR
    Base directory that contains the external HDF / NetCDF data files.
    None of these large files are committed to the git repository.

    Example (.env or shell rc)::

        export KOFMAN_DATA_DIR=/Volumes/AndrewEXT/kofman2024_inputdata

Optional per-file overrides
-----------------------------
If set, these override the glob-based lookup inside KOFMAN_DATA_DIR:

KOFMAN_MCD12C1    absolute path to  MCD12C1.*.hdf
KOFMAN_MOD10CM    absolute path to  MOD10CM.*.hdf
KOFMAN_NSIDC_ICE  absolute path to  iceage_nh_*.nc  (NSIDC sea-ice, NH only)

Public API
----------
get_data_paths()
    Resolve and validate the three file paths from env vars.

process_mcd12c1(filepath) -> dict
    Read MODIS annual land cover and return per-pixel areal fractions of the
    5 Kofman classes at the target 91 × 144 grid.

process_snow_ice(mod10cm_path, nsidc_path, *, snow_threshold, target_date) -> ndarray
    Build a (91, 144) boolean snow/ice mask by merging MOD10CM monthly snow
    cover with NSIDC weekly sea-ice age data.

build_map_image(land_fracs, snow_mask) -> tuple[ndarray, ndarray]
    Combine land-cover fractions and snow mask into a (91, 144) float
    map_image suitable for SpectralMap, plus a flipped snow mask for direct
    injection into smap.map after construction.

build_spectralmap_from_modis(wav, *, land_fracs, snow_mask, ...) -> SpectralMap
    End-to-end convenience wrapper: builds and returns a fully configured
    SpectralMap with the real-data snow mask injected.

IGBP → 5-class mapping used
----------------------------
IGBP class index → Kofman category:

  ocean  :  0 (water bodies)
  forest :  1 (ENF), 2 (EBF), 3 (DNF), 4 (DBF), 5 (mixed forests)
  grass  :  9 (savannas), 10 (grasslands), 12 (croplands),
            14 (cropland/NV mosaic)
  soil   :  6 (closed shrublands), 7 (open shrublands),
            8 (woody savannas), 11 (permanent wetlands),
            13 (urban), 16 (barren)
  snow   : 15 (snow/ice from IGBP) — extended by MOD10CM + NSIDC

Notes on input files
--------------------
MCD12C1 (MODIS/Terra+Aqua Land Cover Type, 0.05°):
  - 3600 × 7200 pixels, row 0 = 90°N, col 0 = 180°W
  - Uses ``Land_Cover_Type_1_Percent`` (shape 3600 × 7200 × 17, uint8 0–100)
    which gives the areal percentage of each IGBP class within each pixel.
    This is preferred over the majority-vote layer for area-weighted rebinning.

MOD10CM (MODIS Monthly Snow Cover, 0.05°):
  - Same 3600 × 7200 grid as MCD12C1
  - ``Snow_Cover_Monthly_CMG`` values:
      0–100   percent snow cover in cell
      211     polar night (no illumination; treated as unknown here)
      250     cloud-obscured
      253     no decision
      254     water mask
      255     fill / missing
  - NOTE: the user-supplied file is from October 2021 (A2021274).  For the
    summer-solstice (June 2022) simulation a June MOD10CM file would be more
    appropriate.  The NSIDC sea-ice layer compensates for the NH; use
    ``snow_threshold`` to tune sensitivity.

NSIDC NH sea-ice (EASE-Grid 12.5 km, weekly):
  - 52 weekly snapshots for 2022 (722 × 722 grid, Lambert azimuthal equal-area)
  - Pre-computed ``latitude`` / ``longitude`` arrays embedded in the NetCDF
    file — no external projection library required.
  - ``age_of_sea_ice`` values:
      0     open ocean
      1–19  sea ice (age in years)
      20    land mask
      21    no-data near coast / lakes
  - Only NH is available (lat ≥ ~30°N); SH ice coverage falls back to
    MOD10CM snow values.
"""

from __future__ import annotations

import os
import glob
import warnings
import datetime
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Grid constants (Kofman 2024 target resolution)
# ══════════════════════════════════════════════════════════════════════════════

OUT_NLAT: int = 91      # 2° latitude spacing  (90°N → 90°S, inclusive)
OUT_NLON: int = 144     # 2.5° longitude spacing (180°W → 177.5°E)

# Input (MODIS CMG) grid
IN_NLAT:  int = 3600    # 0.05° lat
IN_NLON:  int = 7200    # 0.05° lon
IN_DLAT:  float = 0.05  # degrees per row  (north-to-south)
IN_DLON:  float = 0.05  # degrees per col  (west-to-east)

# Derived: exactly 50 input columns per output column (7200 / 144 = 50)
COL_FACTOR: int = IN_NLON // OUT_NLON   # = 50

# IGBP class index → Kofman 5-class mapping
# Each entry: (igbp_class_index, kofman_category_name)
_IGBP_TO_KOFMAN: dict[int, str] = {
    0:  "ocean",    # water bodies
    1:  "forest",   # evergreen needleleaf
    2:  "forest",   # evergreen broadleaf
    3:  "forest",   # deciduous needleleaf
    4:  "forest",   # deciduous broadleaf
    5:  "forest",   # mixed forests
    6:  "soil",     # closed shrublands
    7:  "soil",     # open shrublands
    8:  "soil",     # woody savannas (mixed tree/shrub/grass)
    9:  "grass",    # savannas
    10: "grass",    # grasslands
    11: "soil",     # permanent wetlands
    12: "grass",    # croplands
    13: "soil",     # urban and built-up
    14: "grass",    # cropland / natural vegetation mosaic
    15: "snow",     # snow and ice (IGBP static)
    16: "soil",     # barren or sparsely vegetated
}
_KOFMAN_CLASSES: tuple[str, ...] = ("ocean", "forest", "grass", "soil", "snow")

# SpectralMap pixel IDs used in map_image
MAP_ID_OCEAN: float = 0.0
MAP_ID_LAND:  float = 1.0   # continent (forest / grass / soil combined)
MAP_ID_SNOW:  float = 3.0   # matches SpectralMap default snow_id


# ══════════════════════════════════════════════════════════════════════════════
#  Environment variable helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_data_paths() -> dict[str, str]:
    """
    Resolve the three input file paths from environment variables.

    Priority
    --------
    1. Individual override  (``KOFMAN_MCD12C1``, etc.)
    2. Glob search inside   ``KOFMAN_DATA_DIR``

    Returns
    -------
    dict with keys ``'mcd12c1'``, ``'mod10cm'``, ``'nsidc_ice'`` → absolute paths.

    Raises
    ------
    EnvironmentError
        If ``KOFMAN_DATA_DIR`` is not set and no individual override is given
        for a required file, or if a located path does not exist.
    """
    data_dir = os.environ.get("KOFMAN_DATA_DIR", "")

    def _resolve(env_var: str, glob_pattern: str, label: str) -> str:
        # 1) explicit override
        explicit = os.environ.get(env_var, "")
        if explicit:
            if not os.path.isfile(explicit):
                raise EnvironmentError(
                    f"{env_var}={explicit!r} does not point to an existing file."
                )
            return explicit

        # 2) glob inside data_dir
        if not data_dir:
            raise EnvironmentError(
                f"Neither {env_var} nor KOFMAN_DATA_DIR is set.  "
                f"Set KOFMAN_DATA_DIR to the directory containing {label}, "
                f"or set {env_var} directly."
            )
        matches = sorted(glob.glob(os.path.join(data_dir, glob_pattern)))
        if not matches:
            raise EnvironmentError(
                f"No file matching '{glob_pattern}' found in KOFMAN_DATA_DIR={data_dir!r}. "
                f"Alternatively set {env_var} explicitly."
            )
        if len(matches) > 1:
            warnings.warn(
                f"Multiple matches for '{glob_pattern}' in {data_dir!r}; "
                f"using the most recent: {matches[-1]}"
            )
        return matches[-1]

    return {
        "mcd12c1":  _resolve("KOFMAN_MCD12C1",   "MCD12C1.*.hdf",        "MCD12C1 land cover"),
        "mod10cm":  _resolve("KOFMAN_MOD10CM",   "MOD10CM.*.hdf",        "MOD10CM snow cover"),
        "nsidc_ice": _resolve("KOFMAN_NSIDC_ICE", "iceage_nh_*.nc",       "NSIDC NH sea-ice"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Rebinning helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rebin_cols_exact(arr_2d: np.ndarray) -> np.ndarray:
    """
    Average columns from IN_NLON (7200) to OUT_NLON (144) by a factor of 50.
    Input shape: (nrows, 7200).  Output shape: (nrows, 144).
    """
    nrows = arr_2d.shape[0]
    return arr_2d.reshape(nrows, OUT_NLON, COL_FACTOR).mean(axis=2)


def _rebin_rows(arr_2d: np.ndarray) -> np.ndarray:
    """
    Average rows from IN_NLAT (3600) to OUT_NLAT (91) using equal-size chunks.

    3600 / 91 ≈ 39.56 → np.array_split creates 51 chunks of 40 rows and
    40 chunks of 39 rows (51*40 + 40*39 = 2040+1560 = 3600 ✓).
    The sub-degree mis-alignment is negligible for the ~2° target resolution.

    Input shape: (3600, ncols).  Output shape: (91, ncols).
    """
    chunks = np.array_split(arr_2d, OUT_NLAT, axis=0)
    return np.stack([c.mean(axis=0) for c in chunks], axis=0)


def _rebin_2d(arr_2d: np.ndarray) -> np.ndarray:
    """Rebin a (3600, 7200) array to (91, 144) in two passes."""
    arr_col = _rebin_cols_exact(arr_2d)   # (3600, 144)
    return _rebin_rows(arr_col)            # (91,   144)


# ══════════════════════════════════════════════════════════════════════════════
#  1. MCD12C1 land-cover processor
# ══════════════════════════════════════════════════════════════════════════════

def process_mcd12c1(filepath: str) -> dict[str, np.ndarray]:
    """
    Read MODIS annual land cover (MCD12C1 IGBP Type 1) and return per-pixel
    areal fractions of the 5 Kofman classes on the 91 × 144 target grid.

    The ``Land_Cover_Type_1_Percent`` SDS (shape 3600 × 7200 × 17, uint8)
    gives the percentage of each of the 17 IGBP classes within every 0.05°
    pixel.  Reading class-by-class limits peak memory to ~26 MB per pass.

    Parameters
    ----------
    filepath : str
        Absolute path to the MCD12C1 HDF4 file.

    Returns
    -------
    dict[str, ndarray]
        Keys: ``'ocean'``, ``'forest'``, ``'grass'``, ``'soil'``, ``'snow'``
        Values: float32 arrays of shape (91, 144), values in [0, 1] representing
        the fraction of that category within each 2° × 2.5° cell.

    Notes
    -----
    * Fractions are normalised so they sum to 1.0 per pixel (any residual from
      fill pixels is absorbed into the dominant class).
    * The snow fraction here reflects only the IGBP static snow/ice class (15)
      and should be supplemented by :func:`process_snow_ice`.
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ImportError(
            "pyhdf is required to read HDF4 MODIS files.  Install with:\n"
            "  conda install -c conda-forge pyhdf"
        ) from exc

    print(f"[MCD12C1] Opening {os.path.basename(filepath)} …")
    hdf = SD(filepath, SDC.READ)
    pct_sds = hdf.select("Land_Cover_Type_1_Percent")   # (3600, 7200, 17)

    # Accumulate summed fractions for each Kofman class (float32 to save RAM)
    kofman_accum: dict[str, np.ndarray] = {
        cls: np.zeros((IN_NLAT, IN_NLON), dtype=np.float32)
        for cls in _KOFMAN_CLASSES
    }
    fill_val = 255

    print(f"  Reading 17 IGBP bands and mapping to 5 Kofman classes …")
    for igbp_idx in range(17):
        kofman_cls = _IGBP_TO_KOFMAN[igbp_idx]

        # Read one band:  pct_sds[:, :, igbp_idx]  → (3600, 7200) uint8
        band = np.array(pct_sds[:, :, igbp_idx], dtype=np.float32)

        # Mask fill values (255 → 0 contribution)
        band[band == fill_val] = 0.0

        # Convert percent → fraction
        band /= 100.0

        kofman_accum[kofman_cls] += band

    hdf.end()

    # Rebin each class to (91, 144)
    print(f"  Rebinning {IN_NLAT}×{IN_NLON} → {OUT_NLAT}×{OUT_NLON} …")
    fracs: dict[str, np.ndarray] = {}
    for cls in _KOFMAN_CLASSES:
        fracs[cls] = _rebin_2d(kofman_accum[cls]).astype(np.float32)

    # Normalise rows so fractions sum to 1 per pixel
    total = sum(fracs[cls] for cls in _KOFMAN_CLASSES)
    nz = total > 0
    for cls in _KOFMAN_CLASSES:
        fracs[cls][nz] /= total[nz]

    _print_class_summary(fracs)
    return fracs


def _print_class_summary(fracs: dict[str, np.ndarray]) -> None:
    """Print global mean areal fractions."""
    print("  Global areal fractions (91×144 grid):")
    for cls in _KOFMAN_CLASSES:
        pct = float(fracs[cls].mean()) * 100
        print(f"    {cls:8s}: {pct:5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Snow / ice processor (MOD10CM + NSIDC)
# ══════════════════════════════════════════════════════════════════════════════

def process_snow_ice(
    mod10cm_path: str,
    nsidc_path: str,
    *,
    snow_threshold: int = 50,
    target_date: datetime.date = datetime.date(2022, 6, 21),
) -> np.ndarray:
    """
    Build a (91, 144) boolean snow/ice mask by merging MOD10CM monthly snow
    cover with NSIDC weekly NH sea-ice data.

    Strategy
    --------
    1. MOD10CM provides the base: every pixel with ≥ ``snow_threshold`` %
       snow cover is flagged as snow.  Cloud-obscured and night pixels are
       treated as "no decision" (not overriding any existing ice flag).
    2. NSIDC NH sea-ice is reprojected onto the 91 × 144 grid using the
       embedded ``latitude`` / ``longitude`` arrays (no pyproj needed).  Any
       pixel with ``age_of_sea_ice`` in 1–19 (i.e., ice present) is added to
       the mask.  This compensates for ocean pixels that MOD10CM water-masks
       (254) and fills in NH Arctic coverage.
    3. The merged mask covers both hemispheres: MOD10CM handles SH snow and
       continental NH snow; NSIDC handles NH sea ice.

    Parameters
    ----------
    mod10cm_path : str
        Absolute path to the MOD10CM HDF4 file.
    nsidc_path : str
        Absolute path to the NSIDC NH sea-ice NetCDF.
    snow_threshold : int, optional
        Minimum percent snow cover in MOD10CM cell to call it "snowy".
        Range 1–100.  Default 50 (majority-covered).
    target_date : datetime.date, optional
        Date of the simulation; used to select the nearest NSIDC weekly
        snapshot.  Default 2022-06-21 (summer solstice).

    Returns
    -------
    ndarray
        Boolean array of shape (91, 144), True where snow or sea ice is
        present.  Row 0 = North Pole, row 90 = South Pole.

    Warnings
    --------
    The MOD10CM file date (encoded in the filename as DOY) may differ from
    ``target_date``.  For summer-solstice simulations a June MOD10CM file
    is preferred; the supplied October 2021 file will over-estimate NH snow
    extent.  Adjust ``snow_threshold`` upward (e.g., 80) to compensate.
    """
    snow_mask_91x144 = _process_mod10cm(mod10cm_path, snow_threshold)
    ice_mask_91x144  = _process_nsidc(nsidc_path, target_date)

    # Union: snowy OR icy
    merged = snow_mask_91x144 | ice_mask_91x144

    n_snow = int(snow_mask_91x144.sum())
    n_ice  = int(ice_mask_91x144.sum())
    n_both = int(merged.sum())
    pct    = n_both / (OUT_NLAT * OUT_NLON) * 100
    print(f"  Snow/ice mask: MOD10CM={n_snow} cells, NSIDC={n_ice} cells, "
          f"merged={n_both} cells ({pct:.1f}% of grid)")

    return merged


def _process_mod10cm(filepath: str, snow_threshold: int) -> np.ndarray:
    """
    Return a (91, 144) boolean array: True where MOD10CM snow ≥ threshold %.

    MOD10CM special values handled:
      0–100  → actual % snow cover
      211    → polar night; treated as no-decision (False)
      250    → cloud-obscured; treated as no-decision (False)
      253    → no decision; False
      254    → water mask; False  (NSIDC sea-ice will cover Arctic ocean)
      255    → fill; False
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ImportError("pyhdf required — conda install -c conda-forge pyhdf") from exc

    # Parse the file date from the filename for an informative warning
    bname = os.path.basename(filepath)
    _warn_mod10cm_date(bname)

    print(f"[MOD10CM ] Opening {bname} …")
    hdf = SD(filepath, SDC.READ)
    ds  = hdf.select("Snow_Cover_Monthly_CMG")   # (3600, 7200) uint8
    raw = np.array(ds[:, :], dtype=np.int16)     # int16 avoids overflow
    hdf.end()

    # Snow: values in [snow_threshold, 100]
    snow_hi = (IN_NLAT, IN_NLON)
    is_snow = np.zeros(snow_hi, dtype=np.float32)

    valid = (raw >= 1) & (raw <= 100)
    is_snow[valid] = (raw[valid] >= snow_threshold).astype(np.float32)
    # uncertain pixels (cloud/night/fill) leave is_snow = 0 (no flag)

    # Rebin: average fraction → > 0.5 → boolean at output resolution
    snow_coarse = _rebin_2d(is_snow)
    result = snow_coarse > 0.5

    n = int(result.sum())
    print(f"  MOD10CM snow cells (threshold={snow_threshold}%): {n} of {OUT_NLAT*OUT_NLON}")
    return result


def _warn_mod10cm_date(filename: str) -> None:
    """
    Parse YYYYDDD from MOD10CM filename and warn if it is far from June.

    MOD10CM filename format: MOD10CM.AYYYYDDD.VVV.YYYYDDDHHMMSS.hdf
    """
    parts = filename.split(".")
    if len(parts) >= 2 and parts[1].startswith("A") and len(parts[1]) == 8:
        try:
            year = int(parts[1][1:5])
            doy  = int(parts[1][5:8])
            file_date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
            if file_date.month not in (5, 6, 7):
                warnings.warn(
                    f"MOD10CM file is from {file_date} (month={file_date.month}).  "
                    "For a June summer-solstice simulation a June MOD10CM file "
                    "would be more representative.  Consider increasing "
                    "snow_threshold to reduce over-estimation of NH snow.",
                    UserWarning,
                    stacklevel=3,
                )
        except ValueError:
            pass


def _process_nsidc(filepath: str, target_date: datetime.date) -> np.ndarray:
    """
    Reproject NSIDC NH sea-ice onto the 91 × 144 grid and return a boolean
    mask (True = sea ice present).

    The NetCDF file contains pre-computed ``latitude`` and ``longitude`` arrays
    (722 × 722) so no external projection library is needed.

    Algorithm
    ---------
    For each NSIDC pixel (i, j) that has sea ice (age 1–19):
      * Look up its geographic lat/lon from the embedded arrays.
      * Find the nearest output (row, col) using direct index arithmetic on
        the regular 2° × 2.5° target grid.
      * Set that output cell to True.

    This scatter operation is O(N_ice_pixels) and is fast enough for the
    ~50 000 ice pixels typical of the summer NH minimum.
    """
    try:
        import netCDF4 as nc
    except ImportError as exc:
        raise ImportError("netCDF4 required — conda install netCDF4") from exc

    print(f"[NSIDC   ] Opening {os.path.basename(filepath)} …")
    ds = nc.Dataset(filepath, "r")

    # ── Select the weekly snapshot closest to target_date ───────────────────
    times      = np.array(ds["time"][:])
    base       = datetime.date(1970, 1, 1)
    dates      = [base + datetime.timedelta(days=float(d)) for d in times]
    deltas     = [abs((d - target_date).days) for d in dates]
    best_idx   = int(np.argmin(deltas))
    best_date  = dates[best_idx]
    best_delta = deltas[best_idx]
    print(f"  Using week index {best_idx} ({best_date}, Δ={best_delta}d "
          f"from {target_date})")

    # ── Load ice age, lat, lon ───────────────────────────────────────────────
    # Use [:] to materialise the masked array before casting — some netCDF4
    # versions have __array__(self) without a dtype parameter, which causes a
    # TypeError if np.array(..., dtype=X) is called directly on the variable.
    ice_age = np.asarray(ds["age_of_sea_ice"][best_idx]).astype(np.uint8)  # (722, 722)
    lat_2d  = np.asarray(ds["latitude"][:]).astype(np.float32)             # (722, 722)
    lon_2d  = np.asarray(ds["longitude"][:]).astype(np.float32)            # (722, 722)
    ds.close()

    # Ice is present where age_of_sea_ice is in [1, 19]
    is_ice = (ice_age >= 1) & (ice_age <= 19)
    n_ice_pts = int(is_ice.sum())
    print(f"  Ice-present EASE-grid pixels: {n_ice_pts}")

    if n_ice_pts == 0:
        return np.zeros((OUT_NLAT, OUT_NLON), dtype=bool)

    # ── Scatter ice pixels onto target 91 × 144 grid ────────────────────────
    # Target grid:
    #   Row 0 = 90°N, row 90 = 90°S  → lat = 90 - row * (180 / (OUT_NLAT-1))
    #   Col 0 = 180°W, col 143 = 177.5°E  → lon = -180 + col * 2.5
    #
    # Inverse:
    #   row = round((90 - lat) * (OUT_NLAT - 1) / 180)
    #   col = round((lon + 180) / 2.5)  mod OUT_NLON

    out_ice = np.zeros((OUT_NLAT, OUT_NLON), dtype=bool)

    # Extract only ice pixels to minimise loop iterations
    ice_rows, ice_cols = np.where(is_ice)
    lats_ice = lat_2d[ice_rows, ice_cols]
    lons_ice = lon_2d[ice_rows, ice_cols]

    # Vectorised nearest-grid-cell computation
    out_row = np.round((90.0 - lats_ice) * (OUT_NLAT - 1) / 180.0).astype(int)
    out_col = np.round((lons_ice + 180.0) / 2.5).astype(int) % OUT_NLON

    # Clip rows to valid range (lat_2d can reach ~30°N → row ≤ 30)
    valid = (out_row >= 0) & (out_row < OUT_NLAT)
    out_row = out_row[valid]
    out_col = out_col[valid]

    out_ice[out_row, out_col] = True

    n_cells = int(out_ice.sum())
    print(f"  NSIDC sea-ice cells on 91×144 grid: {n_cells}")
    return out_ice


# ══════════════════════════════════════════════════════════════════════════════
#  3. Assemble map_image for SpectralMap
# ══════════════════════════════════════════════════════════════════════════════

def build_map_image(
    land_fracs: dict[str, np.ndarray],
    snow_mask: np.ndarray,
    *,
    ocean_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine land-cover fractions and a snow mask into a ``map_image`` array
    suitable for passing to :class:`SpectralMap`.

    ``SpectralMap.__init__`` applies::

        map_image[map_image <  0.5] = ocean_id   # 0.0
        map_image[map_image >= 0.5] = cont_id    # 1.0

    and then (if snow_id is not None) calls ``add_poles()`` or the user can
    inject the snow mask directly into ``smap.map`` after construction — see
    :func:`build_spectralmap_from_modis` for the recommended workflow.

    Parameters
    ----------
    land_fracs : dict[str, ndarray]
        Output of :func:`process_mcd12c1`.  Keys: ocean, forest, grass,
        soil, snow.  Values: (91, 144) float32 in [0, 1].
    snow_mask : ndarray
        Boolean (91, 144) from :func:`process_snow_ice`.
        Row 0 = North Pole (standard geographic orientation).
    ocean_threshold : float, optional
        Pixels where the ocean fraction exceeds this value are assigned
        ocean; all others are assigned land.  Default 0.5.

    Returns
    -------
    map_image : ndarray, float32, shape (91, 144)
        • 0.0 = ocean  (passed as ``ocean_id`` to SpectralMap)
        • 1.0 = land   (passed as ``cont_id``  to SpectralMap)
        Snow pixels are encoded as land here; use the returned
        ``snow_mask_for_injection`` to overwrite them after SpectralMap
        construction.

    snow_mask_for_injection : ndarray, bool, shape (91, 144)
        The combined snow mask in **flipped** orientation (row 0 = South
        Pole) matching ``smap.map`` after ``SpectralMap.__init__`` applies
        ``np.flipud``.  Inject directly::

            smap.map[snow_mask_for_injection] = smap.snow_id

    Notes
    -----
    Global coverage fractions printed for reference.
    """
    ocean_frac = land_fracs["ocean"]      # (91, 144), fraction 0–1

    # Binary land / ocean at output resolution
    map_image = np.where(ocean_frac >= ocean_threshold,
                         MAP_ID_OCEAN, MAP_ID_LAND).astype(np.float32)

    # Report coverage
    n_ocean = int((map_image == MAP_ID_OCEAN).sum())
    n_land  = int((map_image == MAP_ID_LAND ).sum())
    n_snow  = int(snow_mask.sum())
    total   = OUT_NLAT * OUT_NLON
    print(f"\n[map_image] ocean={n_ocean} ({n_ocean/total*100:.1f}%), "
          f"land={n_land} ({n_land/total*100:.1f}%), "
          f"snow_overlay={n_snow} ({n_snow/total*100:.1f}%)")

    # SpectralMap stores self.map = np.flipud(map_image)
    # → flip the snow mask to match that internal orientation
    snow_mask_for_injection = np.flipud(snow_mask)

    return map_image, snow_mask_for_injection


# ══════════════════════════════════════════════════════════════════════════════
#  4. End-to-end SpectralMap builder
# ══════════════════════════════════════════════════════════════════════════════

def build_spectralmap_from_modis(
    wav: "np.ndarray",
    *,
    land_fracs: Optional[dict[str, np.ndarray]] = None,
    snow_mask:  Optional[np.ndarray] = None,
    data_paths: Optional[dict[str, str]] = None,
    snow_threshold: int   = 50,
    target_date: datetime.date = datetime.date(2022, 6, 21),
    ydeg:        int   = 20,
    smoothing:   float = 1.5,
    roughness:   float = 0.0,
) -> "SpectralMap":
    """
    End-to-end convenience function: process MODIS/NSIDC data and return a
    fully configured :class:`SpectralMap` with the real snow mask injected.

    If ``land_fracs`` or ``snow_mask`` are already computed (e.g. from a
    previous run or from saved ``.npy`` files), pass them directly to skip
    the expensive MODIS I/O.

    Parameters
    ----------
    wav : ndarray
        Wavelength grid in microns (passed to SpectralMap).
    land_fracs : dict, optional
        Pre-computed output of :func:`process_mcd12c1`.
    snow_mask : ndarray, optional
        Pre-computed output of :func:`process_snow_ice`.
    data_paths : dict, optional
        Override data-file paths.  Defaults to :func:`get_data_paths`.
    snow_threshold : int
        Passed to :func:`process_snow_ice`.
    target_date : datetime.date
        NSIDC weekly snapshot selection.
    ydeg : int
        Spherical-harmonic degree for SpectralMap.
    smoothing : float
        Passed to starry Map.load() to reduce Gibbs ringing.
    roughness : float
        Oren-Nayar roughness (0 = Lambertian, matching the paper).

    Returns
    -------
    SpectralMap
        Configured map with:
        - Binary land/ocean from MCD12C1
        - Snow pixels injected from MOD10CM + NSIDC
        - Continental spectrum from reflect_dev_dir ASTER library
        - Ocean spectrum from seawater USGS+ASTER

    Usage example
    -------------
    ::

        import os
        os.environ["KOFMAN_DATA_DIR"] = "/Volumes/AndrewEXT/kofman2024_inputdata"

        from starry.extensions.reflect_dev_dir import wav_grid_from_R
        from starry.extensions.reflect_dev_dir.kofman2024_surface_map import (
            build_spectralmap_from_modis
        )

        wav  = wav_grid_from_R(R=70, wav_min=0.30, wav_max=1.00)
        smap = build_spectralmap_from_modis(wav)
        specmap = smap.get_specmap(plot_scalar=False)
    """
    # ── Lazy imports (starry and its deps only needed at call time) ──────────
    from starry.extensions.reflect_dev_dir.reflect import (
        SpectralMap,
        load_surface_spectra,
        load_other_spectra,
        build_continent_spectra,
    )
    try:
        from spectres import spectres
    except ImportError as exc:
        raise ImportError("spectres required — pip install spectres") from exc

    # ── Resolve data files ───────────────────────────────────────────────────
    if data_paths is None:
        data_paths = get_data_paths()

    # ── Compute land fractions if not supplied ───────────────────────────────
    if land_fracs is None:
        print("\n── Step 1/3: Land cover (MCD12C1) ──────────────────────────")
        land_fracs = process_mcd12c1(data_paths["mcd12c1"])

    # ── Compute snow mask if not supplied ────────────────────────────────────
    if snow_mask is None:
        print("\n── Step 2/3: Snow / sea-ice mask ────────────────────────────")
        snow_mask = process_snow_ice(
            data_paths["mod10cm"],
            data_paths["nsidc_ice"],
            snow_threshold=snow_threshold,
            target_date=target_date,
        )

    # ── Build map_image ──────────────────────────────────────────────────────
    print("\n── Step 3/3: Building map_image ─────────────────────────────")
    map_image, snow_mask_flipped = build_map_image(land_fracs, snow_mask)

    # ── Load surface spectra ─────────────────────────────────────────────────
    regions    = load_surface_spectra()
    other_spec = load_other_spectra()
    regions    = build_continent_spectra(regions, other_spec, wav_deep=wav)

    cont_spec  = regions["cont"][:, 1]
    ocean_spec = spectres(
        wav, regions["sea"][:, 0], regions["sea"][:, 1],
        fill=float(regions["sea"][0, 1]), verbose=False,
    )
    cloud_spec = spectres(
        wav, regions["cloud"][:, 0], regions["cloud"][:, 1],
        fill=float(regions["cloud"][0, 1]), verbose=False,
    )
    snow_spec  = spectres(
        wav, regions["snow"][:, 0], regions["snow"][:, 1],
        fill=float(regions["snow"][0, 1]), verbose=False,
    )
    for arr in (cont_spec, ocean_spec, cloud_spec, snow_spec):
        np.clip(arr, 0.0, 1.0, out=arr)

    # ── Construct SpectralMap ────────────────────────────────────────────────
    # snow_dims=(0,0,0,0) prevents add_poles() from overwriting our custom mask
    smap = SpectralMap(
        map_image  = map_image.copy(),
        wav        = wav,
        cont_id    = MAP_ID_LAND,
        ocean_id   = MAP_ID_OCEAN,
        snow_id    = MAP_ID_SNOW,
        snow_spec  = snow_spec,
        snow_dims  = (0, 0, 0, 0),   # disable geometric polar caps
        ydeg       = ydeg,
        scalar_wav = 0.7,
        smoothing  = smoothing,
        roughness  = roughness,
    )

    # ── Inject real snow mask into smap.map ──────────────────────────────────
    # smap.map is stored flipped (row 0 = SP) after SpectralMap.__init__
    n_injected = int(snow_mask_flipped.sum())
    smap.map[snow_mask_flipped] = MAP_ID_SNOW
    print(f"\n  Injected {n_injected} real-data snow/ice cells into smap.map "
          f"(id={MAP_ID_SNOW})")
    print("  SpectralMap ready.  Call smap.get_specmap() to compute SH coefficients.")

    return smap


# ══════════════════════════════════════════════════════════════════════════════
#  Caching helpers  (save / load processed arrays to avoid re-running I/O)
# ══════════════════════════════════════════════════════════════════════════════

def save_processed(
    land_fracs: dict[str, np.ndarray],
    snow_mask:  np.ndarray,
    out_dir:    str = ".",
) -> None:
    """
    Save the processed arrays as compressed .npz files so MODIS I/O only
    needs to run once.

    Files written
    -------------
    ``{out_dir}/kofman_land_fracs.npz``   — land_fracs dict
    ``{out_dir}/kofman_snow_mask.npy``    — snow_mask bool array
    """
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "kofman_land_fracs.npz"),
        **{k: v for k, v in land_fracs.items()},
    )
    np.save(os.path.join(out_dir, "kofman_snow_mask.npy"), snow_mask)
    print(f"Saved processed arrays to {out_dir}/kofman_land_fracs.npz "
          f"and kofman_snow_mask.npy")


def load_processed(cache_dir: str = ".") -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load previously saved processed arrays.

    Returns
    -------
    (land_fracs, snow_mask)

    Raises
    ------
    FileNotFoundError if the cache files are not present.
    """
    fracs_path = os.path.join(cache_dir, "kofman_land_fracs.npz")
    snow_path  = os.path.join(cache_dir, "kofman_snow_mask.npy")

    if not os.path.isfile(fracs_path) or not os.path.isfile(snow_path):
        raise FileNotFoundError(
            f"Cache files not found in {cache_dir!r}.  "
            "Run process_mcd12c1() and process_snow_ice() first, then "
            "call save_processed()."
        )

    data = np.load(fracs_path)
    land_fracs = {k: data[k] for k in _KOFMAN_CLASSES}
    snow_mask  = np.load(snow_path)
    print(f"Loaded cached arrays from {cache_dir}")
    return land_fracs, snow_mask


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process MODIS/NSIDC data into Kofman (2024) surface map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save-cache", metavar="DIR", default=".",
        help="Directory to save processed .npz/.npy cache files.",
    )
    parser.add_argument(
        "--snow-threshold", type=int, default=50,
        help="MOD10CM percent-snow threshold for marking a cell as snowy.",
    )
    parser.add_argument(
        "--target-date", default="2022-06-21",
        help="Simulation date for NSIDC weekly snapshot selection (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip diagnostic map plots.",
    )
    args = parser.parse_args()

    target = datetime.date.fromisoformat(args.target_date)

    # Resolve data paths from environment
    paths = get_data_paths()
    print(f"MCD12C1 : {paths['mcd12c1']}")
    print(f"MOD10CM : {paths['mod10cm']}")
    print(f"NSIDC   : {paths['nsidc_ice']}")

    print("\n" + "═" * 60)
    print("  Processing land cover …")
    print("═" * 60)
    land_fracs = process_mcd12c1(paths["mcd12c1"])

    print("\n" + "═" * 60)
    print("  Processing snow / sea-ice …")
    print("═" * 60)
    snow_mask = process_snow_ice(
        paths["mod10cm"], paths["nsidc_ice"],
        snow_threshold=args.snow_threshold,
        target_date=target,
    )

    map_image, snow_mask_flipped = build_map_image(land_fracs, snow_mask)

    # Save cache
    save_processed(land_fracs, snow_mask, out_dir=args.save_cache)

    # Optional diagnostic plots
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            titles = ["ocean", "forest", "grass", "soil", "snow", "map_image"]
            data   = [land_fracs[c] for c in _KOFMAN_CLASSES] + [map_image]

            for ax, title, d in zip(axes.flat, titles, data):
                im = ax.imshow(d, origin="upper", aspect="auto",
                               extent=[-180, 180, -90, 90],
                               cmap="viridis" if title != "map_image" else "RdBu_r")
                ax.set_title(title)
                ax.set_xlabel("Lon")
                ax.set_ylabel("Lat")
                plt.colorbar(im, ax=ax, fraction=0.03)

            plt.suptitle(
                f"Kofman (2024) surface fractions — {OUT_NLAT}×{OUT_NLON} grid",
                fontsize=13,
            )
            plt.tight_layout()
            out_fig = os.path.join(args.save_cache, "kofman_surface_map.png")
            plt.savefig(out_fig, dpi=120, bbox_inches="tight")
            print(f"\nDiagnostic plot saved → {out_fig}")
        except Exception as e:
            print(f"(Plot skipped: {e})")

    print("\nDone.")
