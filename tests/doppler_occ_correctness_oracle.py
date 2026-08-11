"""
Correctness oracle for DopplerMap.flux() during occultation (Phase 0.2 of
the DopplerMap occultation speedup plan).

Evaluates DopplerMap.flux(theta=..., xo=..., yo=..., ro=...) -- which
dispatches through get_flux_from_dotconv_occ -> get_kT_occ -> get_rT_occ,
the exact code path Phase 2's epoch-axis vectorization will rewrite -- at
a small, fixed, representative set of numeric inputs, and saves the
result as a regression fixture.

Test cases (5 epochs, one ro value per case):
  epoch 0: far off-disk                (xo, yo) = (5.0, 5.0)
  epoch 1: fully on-disk, near center   (xo, yo) = (0.0, 0.0)
  epoch 2: on-disk, near limb           (xo, yo) = (0.9, 0.0)
  epoch 3: boundary -- grazing overlap  (xo, yo) = (1.0 + ro - eps, 0.0)
  epoch 4: on-disk, off-center          (xo, yo) = (0.3, 0.4)
run across ro in {0.0 (degenerate/no occultation), 0.05 (small), 0.2 (large)}.

Any future rewrite of get_kT_occ/get_rT_occ (e.g. Phase 2's epoch-axis
vectorization) MUST reproduce this fixture to np.allclose(atol=1e-12)
before being trusted. This is a forward-pass-only oracle; Phase 2.2 ALSO
requires a separate finite-difference gradient check (see the plan) --
this script does not replace that, it only covers correctness of the
forward pass.

Usage:
    python tests/doppler_occ_correctness_oracle.py --save   # compute + write the fixture (run once, on known-good code)
    python tests/doppler_occ_correctness_oracle.py --check  # compare current code against the saved fixture
"""

import argparse
import os

import numpy as np

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "doppler_occ_oracle_fixture.npz")

# Kept small (ydeg=3) so this runs fast and is easy to reason about by
# hand -- NOT meant to be representative of production ydeg/nt, only of
# the *behaviors* get_kT_occ/get_rT_occ must reproduce exactly (off-disk,
# on-disk, boundary overlap, degenerate ro=0).
YDEG = 3
UDEG = 2
NT = 5
NW = 41
VSINI_MAX = 25000.0

XO_BASE = np.array([5.0, 0.0, 0.9, np.nan, 0.3])  # epoch 3 filled in per ro case
YO = np.array([5.0, 0.0, 0.0, 0.0, 0.4])
THETA = np.zeros(NT)

RO_CASES = {"ro_0p05": 0.05, "ro_0p2": 0.2, "ro_0": 0.0}


def build_map():
    import starry

    rv_grid = np.linspace(-30, 30, NW)
    wav_grid = 600.0 * (1 + rv_grid / 299792.458)
    dm = starry.DopplerMap(
        ydeg=YDEG, udeg=UDEG, nt=NT, wav=wav_grid, interpolate=True,
        lazy=False, vsini_max=VSINI_MAX,
    )
    dm[1] = 0.4
    dm[2] = 0.2
    dm.veq = 20000.0
    dm.inc = 90.0
    # Load an actual absorption line into the rest-frame spectrum. Without
    # this, the default rest-frame spectrum is flat continuum (uniform
    # 1.0), so there is no line structure for occultation geometry to
    # distort -- every ro value produces an identical flat-continuum
    # output, and this fixture would be unable to detect ANY correctness
    # regression in get_kT_occ/get_rT_occ.
    wav0 = dm.wav0
    line_ctrst, line_wav, line_sigma = 0.6, 600.0, 0.5
    intr_spec = 1.0 - line_ctrst * np.exp(-0.5 * ((wav0 - line_wav) / line_sigma) ** 2)
    dm.load(spectrum=intr_spec)
    dm.obl = 0.0
    return dm


def evaluate_case(dm, ro):
    xo = XO_BASE.copy()
    xo[3] = 1.0 + ro - 1e-3  # just inside the grazing-overlap boundary
    yo = YO.copy()
    flux = dm.flux(theta=THETA, xo=xo, yo=yo, ro=ro)
    return np.asarray(flux)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="Compute and save the fixture")
    g.add_argument("--check", action="store_true", help="Compare current code against the saved fixture")
    args = p.parse_args()

    dm = build_map()

    outputs = {}
    for name, ro in RO_CASES.items():
        print(f"Evaluating case: {name} (ro={ro}) ...")
        outputs[name] = evaluate_case(dm, ro)
        print(f"  shape={outputs[name].shape}  sum={outputs[name].sum():.10e}")

    if args.save:
        np.savez(FIXTURE_PATH, **outputs)
        print(f"\nSaved oracle fixture to {FIXTURE_PATH}")
        print("Commit this fixture alongside any Phase 2 code change so --check")
        print("has a known-good baseline to compare against.")
    else:
        if not os.path.exists(FIXTURE_PATH):
            raise SystemExit(
                f"No fixture found at {FIXTURE_PATH}. Run with --save first, "
                "on the current (pre-Phase-2) code, to create the baseline."
            )
        fixture = np.load(FIXTURE_PATH)
        all_ok = True
        for name in outputs:
            ok = np.allclose(outputs[name], fixture[name], atol=1e-12, rtol=0)
            status = "OK" if ok else "MISMATCH"
            if not ok:
                all_ok = False
                diff = np.abs(outputs[name] - fixture[name])
                print(f"  {name}: {status}  max_abs_diff={diff.max():.3e}")
            else:
                print(f"  {name}: {status}")
        if all_ok:
            print("\nAll cases match the saved oracle fixture.")
        else:
            print("\nMISMATCH DETECTED -- do not trust this code change until resolved.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
