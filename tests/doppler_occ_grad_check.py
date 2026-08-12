"""
Finite-difference gradient check for DopplerMap.flux() during occultation
(Phase 2.2, step 3 of the DopplerMap occultation speedup plan).

The correctness oracle (doppler_occ_correctness_oracle.py) only validates
the FORWARD pass. A correct forward pass with a broken backward pass is a
silent failure mode: it wouldn't crash, it would just bias posteriors in
the real PyMC3 obliquity fit. This script builds theano.grad(loss, param)
for each of the parameters the real fit differentiates w.r.t. -- obl,
veq, inc, and a limb-darkening coefficient (u1) -- through the rewritten
get_kT_occ/get_rT_occ, and compares it against a central finite-difference
estimate computed by re-evaluating flux numerically at param +/- eps.

Uses the same small (ydeg=3, udeg=2, nt=5) config as the correctness
oracle so this runs in seconds, with one clearly on-disk occultation case
(ro=0.2) so every epoch's tt.switch branch sits away from any boundary
(finite differences are unreliable right at a switch's kink).

Usage:
    python tests/doppler_occ_grad_check.py
"""

import numpy as np

YDEG = 3
UDEG = 2
NT = 5
NW = 41
VSINI_MAX = 25000.0
RO = 0.2

XO = np.array([5.0, 0.0, 0.5, 0.6, 0.3])
YO = np.array([5.0, 0.0, 0.0, 0.3, 0.4])
THETA = np.zeros(NT)

NOMINAL = dict(obl=0.3, veq=20000.0, inc=80.0, u1=0.4)
EPS = dict(obl=1e-6, veq=1e-1, inc=1e-6, u1=1e-6)
# Loose tolerance: finite differences on a chain this deep (rotation ->
# occultation integral -> limb-darkening operator -> convolution) pick up
# O(eps) truncation error and float64 cancellation noise; this is a sanity
# check for gross sign/shape errors in the backward pass, not a
# high-precision numerical test.
RTOL = 1e-3


def _rv_wav_grid():
    rv_grid = np.linspace(-30, 30, NW)
    wav_grid = 600.0 * (1 + rv_grid / 299792.458)
    return wav_grid


def _load_line(dm):
    wav0 = dm.wav0
    line_ctrst, line_wav, line_sigma = 0.6, 600.0, 0.5
    intr_spec = 1.0 - line_ctrst * np.exp(
        -0.5 * ((wav0 - line_wav) / line_sigma) ** 2
    )
    dm.load(spectrum=intr_spec)


def build_eager(**overrides):
    import starry

    params = dict(NOMINAL)
    params.update(overrides)

    dm = starry.DopplerMap(
        ydeg=YDEG, udeg=UDEG, nt=NT, wav=_rv_wav_grid(), interpolate=True,
        lazy=False, vsini_max=VSINI_MAX,
    )
    dm[1] = params["u1"]
    dm[2] = 0.2
    dm.veq = params["veq"]
    dm.inc = params["inc"]
    _load_line(dm)
    dm.obl = params["obl"]
    return dm


def loss_eager(**overrides):
    dm = build_eager(**overrides)
    flux = np.asarray(dm.flux(theta=THETA, xo=XO, yo=YO, ro=RO))
    return float(np.sum(flux ** 2))


def finite_diff_grad(param):
    eps = EPS[param]
    plus = dict(NOMINAL)
    minus = dict(NOMINAL)
    plus[param] = NOMINAL[param] + eps
    minus[param] = NOMINAL[param] - eps
    return (loss_eager(**plus) - loss_eager(**minus)) / (2 * eps)


def analytic_grad(param):
    import theano
    import theano.tensor as tt
    import starry

    free = tt.dscalar(param + "_free")
    params = dict(NOMINAL)

    dm = starry.DopplerMap(
        ydeg=YDEG, udeg=UDEG, nt=NT, wav=_rv_wav_grid(), interpolate=True,
        lazy=True, vsini_max=VSINI_MAX,
    )
    dm[1] = free if param == "u1" else params["u1"]
    dm[2] = 0.2
    dm.veq = free if param == "veq" else params["veq"]
    dm.inc = free if param == "inc" else params["inc"]
    _load_line(dm)
    dm.obl = free if param == "obl" else params["obl"]

    flux_graph = dm.flux(theta=THETA, xo=XO, yo=YO, ro=RO)
    loss = tt.sum(flux_graph ** 2)
    grad_graph = theano.grad(loss, free)
    grad_fn = theano.function([free], grad_graph)
    return float(grad_fn(NOMINAL[param]))


def main():
    print(f"Config: ydeg={YDEG}, udeg={UDEG}, nt={NT}, ro={RO} (on-disk case)")
    print(f"{'param':6s} {'analytic':>16s} {'finite-diff':>16s} {'rel_err':>10s}  status")

    all_ok = True
    for param in ("obl", "veq", "inc", "u1"):
        a = analytic_grad(param)
        f = finite_diff_grad(param)
        denom = max(abs(a), abs(f), 1e-12)
        rel_err = abs(a - f) / denom
        ok = rel_err < RTOL
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"{param:6s} {a:16.6e} {f:16.6e} {rel_err:10.2e}  {status}")

    if all_ok:
        print("\nAll gradients match finite differences within tolerance.")
    else:
        print("\nGRADIENT MISMATCH DETECTED -- do not trust this code for "
              "gradient-based (NUTS/PyMC3) fitting until resolved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
