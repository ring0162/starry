"""
Does the DopplerMap occultation model vary smoothly as the occultor moves across the velocity grid -- in value
AND in gradient?

The occultation kernel is sampled at discrete lines of constant velocity ("taps"). With the occultor's chord
point-sampled at the tap centres, the flux it blocks depends on where it sits relative to the taps, so the model
ripples with a period of one tap as the occultor moves, and the likelihood becomes rough in every parameter that
moves the planet (obliquity, b, t0, period, eccentricity). `starry/_core/core.py::get_rT_occ` therefore averages
the chord over a window around each tap (STARRY_DOPPLER_OCC_ANTIALIAS):

  "0"    point-sampled (no anti-aliasing)
  "box"  average over each tap's width: smooth values, but the gradient has square-root cusps whenever a tap
         edge crosses the chord's edge (they stall optimisers and hurt HMC)
  "tent" (default) triangular window of half-width one tap: twice continuously differentiable; also smooths
         the clipping of the chord at the stellar limb over one tap (STARRY_DOPPLER_OCC_LIMB_SMOOTH, default on),
         which otherwise kinks the model in ingress / egress frames

Each kernel is run with the occultor inside the disc and on the stellar limb.

This script slides an occultor across two taps in fine steps at fixed y, for each kernel (each in its own
process), and reports

  * ripple: scatter of the occultor's signal about a smooth (cubic) trend in xo, relative to the signal
    (wavelength-summed, and in the line-core channel);
  * gradient check: the analytic d(signal)/d(xo) (Theano) against central finite differences with steps of
    0.02 and 0.2 taps -- the larger step straddles tap edges -- as the worst mismatch relative to the largest
    gradient; a cusp shows up as a large mismatch at the larger step only;
  * curvature spikes: the largest jump of the finite-difference second derivative between neighbouring xo,
    relative to its typical size.

Expected: point-sampled -- ripple of a few per cent; box -- ripple >~10x smaller, but gradient mismatch and
curvature spikes at tap crossings; tent -- ripple as small as box and a gradient that matches at both steps.

Usage:
    python tests/doppler_occ_antialias_check.py
(then also run `python tests/doppler_orbit_grad_check.py` and `python tests/doppler_orbit_params_grad_check.py`
 to confirm the gradients through the orbit).
"""

import json
import os
import subprocess
import sys

import numpy as np

NT = 241
RO = 0.06
YO = 0.25
XO_CENTRE = 0.30
VEQ = 8600.0          # m/s: a slow rotator, so the planet is only a few taps wide
NW = 101
RV_HALF = 25.0        # km/s


XO_LIMB = 0.955       # occultor overlapping the stellar limb (sqrt(0.955^2 + 0.25^2) ~ 0.99)


def run_one():
    out = {}
    for tag, xc in (("", XO_CENTRE), ("limb_", XO_LIMB)):
        out.update({tag + k: v for k, v in _run_at(xc).items()})
    print("RESULT " + json.dumps(out))


def _run_at(xo_centre):
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
    xo = xo_centre + np.linspace(-1.0, 1.0, NT) * tap
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

    # curvature spikes of the line-core signal in xo (finite-difference second derivative)
    y = signal[:, k]
    d2 = np.diff(y, 2)
    spike = float(np.max(np.abs(np.diff(d2))) / max(np.median(np.abs(d2)), 1e-300))

    # analytic gradient (Theano) vs finite differences, line-core channel, at every xo
    import theano
    import theano.tensor as tt

    dl = starry.DopplerMap(ydeg=3, udeg=2, nt=NT, wav=wav, interpolate=True, lazy=True, vsini_max=12000.0)
    dl[1] = 0.4
    dl[2] = 0.2
    dl.veq = VEQ
    dl.inc = 90.0
    dl.obl = 0.0
    dl.load(spectrum=1.0 - 0.5 * np.exp(-0.5 * ((dl.wav0 - 600.0) / 0.0086) ** 2))
    xo_t = tt.dvector("xo")
    fl = dl.flux(theta=np.zeros(NT), xo=xo_t, yo=np.full(NT, YO), ro=RO, normalize=False)
    ch = -fl[:, k]                                       # the occultor's signal in that channel (+ const)
    f_val = theano.function([xo_t], ch)
    f_grad = theano.function([xo_t], theano.grad(tt.sum(ch), xo_t))   # epochs are independent
    g = np.asarray(f_grad(xo))
    gmax = float(np.max(np.abs(g)))
    mism = {}
    for frac in (0.02, 0.2):
        h = frac * tap
        fd = (np.asarray(f_val(xo + h)) - np.asarray(f_val(xo - h))) / (2 * h)
        mism[frac] = float(np.max(np.abs(fd - g)) / gmax)

    out = dict(
        tap=float(tap), taps_across_planet=float(2 * RO / tap),
        ripple_total=ripple(total) / float(total.mean()),
        ripple_channel=ripple(signal[:, k]) / float(signal[:, k].mean()),
        grad_mismatch_small=mism[0.02], grad_mismatch_large=mism[0.2], curvature_spike=spike,
    )
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run_one()
        return
    results = {}
    for flag, limb, name in (("0", "1", "point-sampled"), ("box", "1", "box"),
                             ("tent", "0", "tent, hard limb"), ("tent", "1", "tent")):
        env = dict(os.environ, STARRY_DOPPLER_OCC_ANTIALIAS=flag, STARRY_DOPPLER_OCC_LIMB_SMOOTH=limb)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--one"], env=env,
                              capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:])
            raise SystemExit(f"the {name} run failed")
        results[name] = json.loads(line[-1][7:])
    r0 = results["point-sampled"]
    print(f"occultor radius {RO} R*, tap spacing {r0['tap']:.4f} R*: {r0['taps_across_planet']:.1f} taps across the planet")
    print(f"  {'kernel':16s} {'ripple, summed':>13s} {'ripple, core':>13s} {'grad err 0.02 tap':>18s} "
          f"{'grad err 0.2 tap':>17s} {'curvature spike':>16s}")
    for tag, title in (("", f"occultor inside the disc (xo ~ {XO_CENTRE})"),
                       ("limb_", f"occultor on the stellar limb (xo ~ {XO_LIMB}, ingress / egress)")):
        print(title)
        for name, r in results.items():
            print(f"  {name:16s} {r[tag + 'ripple_total']:13.2e} {r[tag + 'ripple_channel']:13.2e} "
                  f"{r[tag + 'grad_mismatch_small']:18.2e} {r[tag + 'grad_mismatch_large']:17.2e} "
                  f"{r[tag + 'curvature_spike']:16.3g}")
    print("(ripple: fraction of the occultor's signal; grad err: worst |FD - analytic| / max |analytic|; "
          "curvature spike: largest jump of the second difference / its median)")
    rb, rt, rh = results["box"], results["tent"], results["tent, hard limb"]
    checks = {
        "limb: tent gradient matches FD at 0.02 tap (<1e-3)": rt["limb_grad_mismatch_small"] < 1e-3,
        "limb: tent gradient matches FD at 0.2 tap (<3e-2)": rt["limb_grad_mismatch_large"] < 3e-2,
        "limb: smoothing lowers the curvature spikes (vs hard limb)":
            rt["limb_curvature_spike"] < 0.5 * rh["limb_curvature_spike"],
        "tent ripple >5x below point-sampled (summed)": r0["ripple_total"] / max(rt["ripple_total"], 1e-300) > 5,
        "tent ripple >3x below point-sampled (core)": r0["ripple_channel"] / max(rt["ripple_channel"], 1e-300) > 3,
        "tent gradient matches FD at 0.02 tap (<1e-3)": rt["grad_mismatch_small"] < 1e-3,
        "tent gradient matches FD at 0.2 tap (<3e-2)": rt["grad_mismatch_large"] < 3e-2,
        "tent curvature spikes well below box": rt["curvature_spike"] < 0.2 * rb["curvature_spike"],
    }
    for k, v in checks.items():
        print(("  ok   " if v else "  FAIL ") + k)
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
