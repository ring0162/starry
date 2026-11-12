"""
Does the DopplerMap occultation model vary smoothly as the occultor moves across the velocity grid?

The occultation kernel is sampled at discrete lines of constant velocity ("taps"). With the occultor's chord
point-sampled at the tap centres, the flux it blocks depends on where it sits relative to the taps, so the model
ripples with a period of one tap as the occultor moves; for a planet only a few taps wide the ripple is a
sizeable fraction of the planet's signal, and the likelihood becomes rough in every parameter that moves the
planet (obliquity, b, t0, period, eccentricity). `starry/_core/core.py::get_rT_occ` now averages the chord over
each tap (STARRY_DOPPLER_OCC_ANTIALIAS, default on).

This script slides an occultor across two taps in fine steps at fixed y, for the point-sampled and the
anti-aliased kernel (each in its own process), and reports

  * the ripple: scatter of the wavelength-summed flux about a smooth (cubic) trend in xo, relative to the
    occultor's total signal (unocculted minus occulted flux);
  * the same for a single wavelength channel near the line core (where the shadow lives).

Expected: a ripple of a few per cent of the signal for the point-sampled kernel, and at least ~10x less for
the anti-aliased one.

Usage:
    python tests/doppler_occ_antialias_check.py
(then also run `python tests/doppler_orbit_grad_check.py` and `python tests/doppler_orbit_params_grad_check.py`
 to confirm the gradients of the anti-aliased kernel).
"""

import json
import os
import subprocess
import sys

import numpy as np

NT = 61
RO = 0.06
YO = 0.25
XO_CENTRE = 0.30
VEQ = 8600.0          # m/s: a slow rotator, so the planet is only a few taps wide
NW = 101
RV_HALF = 25.0        # km/s


def run_one():
    import starry

    rv = np.linspace(-RV_HALF, RV_HALF, NW)
    wav = 600.0 * (1 + rv / 299792.458)
    dm = starry.DopplerMap(ydeg=3, udeg=2, nt=NT, wav=wav, interpolate=True, lazy=False, vsini_max=12000.0)
    dm[1] = 0.4
    dm[2] = 0.2
    dm.veq = VEQ
    dm.inc = 90.0
    dm.obl = 0.0
    wav0 = dm.wav0
    dm.load(spectrum=1.0 - 0.5 * np.exp(-0.5 * ((wav0 - 600.0) / 0.0086) ** 2))

    # tap spacing in x (stellar radii): one velocity pixel of the internal grid / vsini
    dv = (rv[1] - rv[0]) * 1000.0
    tap = dv / VEQ
    xo = XO_CENTRE + np.linspace(-1.0, 1.0, NT) * tap
    theta = np.zeros(NT)
    occ = np.asarray(dm.flux(theta=theta, xo=xo, yo=np.full(NT, YO), ro=RO, normalize=False))
    far = np.asarray(dm.flux(theta=theta, xo=np.full(NT, 50.0), yo=np.full(NT, 50.0), ro=RO, normalize=False))
    signal = far - occ                                   # what the occultor removes, (NT, NW)

    def ripple(y):
        z = np.linspace(-1, 1, len(y))
        res = y - np.polyval(np.polyfit(z, y, 3), z)
        return float(res.max() - res.min())

    total = signal.sum(axis=1)
    k = int(np.argmax(signal.mean(axis=0)))              # channel where the shadow is strongest
    out = dict(
        tap=float(tap), taps_across_planet=float(2 * RO / tap),
        ripple_total=ripple(total) / float(total.mean()),
        ripple_channel=ripple(signal[:, k]) / float(signal[:, k].mean()),
    )
    print("RESULT " + json.dumps(out))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run_one()
        return
    results = {}
    for flag, name in (("0", "point-sampled"), ("1", "anti-aliased")):
        env = dict(os.environ, STARRY_DOPPLER_OCC_ANTIALIAS=flag)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--one"], env=env,
                              capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
            raise SystemExit(f"the {name} run failed")
        results[name] = json.loads(line[-1][7:])
    r0, r1 = results["point-sampled"], results["anti-aliased"]
    print(f"occultor radius {RO} R*, tap spacing {r0['tap']:.4f} R*: {r0['taps_across_planet']:.1f} taps across the planet")
    print(f"{'kernel':16s} {'ripple, summed flux':>22s} {'ripple, line-core channel':>28s}   (fraction of the occultor's signal)")
    for name, r in results.items():
        print(f"{name:16s} {r['ripple_total']:22.2e} {r['ripple_channel']:28.2e}")
    gain_t = r0["ripple_total"] / max(r1["ripple_total"], 1e-300)
    gain_c = r0["ripple_channel"] / max(r1["ripple_channel"], 1e-300)
    print(f"reduction: x{gain_t:.0f} (summed flux), x{gain_c:.0f} (line-core channel)")
    ok = gain_t > 5 and gain_c > 3
    print("PASS" if ok else "FAIL: the anti-aliased kernel is not clearly smoother")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
