"""
Finite-difference gradient checks for the planet's position and orbit when the
primary is a DopplerMap (`System(..., orbit_grad=True)`).

Background: `System._flux_doppler` used to cut the planet's position off from the
gradient (disconnected_grad), so b/inc, the period, t0 and the planet radius could
never be sampled with NUTS. `orbit_grad` makes that cut optional. This script checks
that, when the cut is lifted, the gradients are right:

  A. kernel level: d(flux)/d(xo, yo, ro) of DopplerMap.flux itself (these inputs
     were never differentiated before), against central finite differences.
  B. system level: d(flux)/d(inc, porb, t0, planet radius, stellar mass) of a lazy
     System with a DopplerMap primary and `orbit_grad=True`, against finite differences
     of the greedy System.
  C. control: the same orbital gradients for an ordinary starry Map primary (this path is
     used with PyMC3/exoplanet but had no gradient test in the repo): checks the harness.
  D. default: with `orbit_grad=False` the gradient w.r.t. the orbit is exactly zero.
  E. `orbit_grad="xy"`: the planet-radius gradient is exactly zero, the position gradient is not.

The loss is sum(flux**2). The occultation integral is only piecewise smooth (kinks where a
velocity slice touches the occultor's edge, xo +- ro), where finite differences are
unreliable, so each finite difference is evaluated at two step sizes (h and h/2) and an
epoch/parameter is reported as "kink" (and not counted) if the two disagree by more than
1e-2 relative. The pass criterion is: every non-kink comparison agrees within RTOL, the
number of kinks is printed, no gradient is NaN or inf.

Usage:
    python tests/doppler_orbit_grad_check.py            # all parts
    python tests/doppler_orbit_grad_check.py A B        # selected parts
"""

import sys

import numpy as np

import doppler_occ_grad_check as base

RTOL = 1e-3
KINK_RTOL = 1e-2
ATOL = 1e-9

# --- part A: epochs for the kernel-level check (nt = 5) -------------------------------
EPOCH_SETS = {
    "interior": (np.array([5.0, 0.0, 0.5, 0.6, 0.3]), np.array([5.0, 0.0, 0.0, 0.3, 0.4]), 0.2),
    "near limb / offset": (np.array([0.95, 0.1, -0.4, 0.0, 7.0]), np.array([0.0, 0.7, 0.2, -0.5, 0.0]), 0.15),
}

# --- parts B-E: a small transiting system around a DopplerMap star --------------------
SYS_NOMINAL = dict(inc=88.0, porb=5.0, t0=0.0, rp=0.1, mstar=1.0)
# inc uses a larger step: d(loss)/d(inc) is tiny (~1e-5; the per-epoch normalisation and the near-limb-darkening
# symmetry of the occultation make the flux weakly sensitive to the impact parameter) so a 1e-5 step is
# dominated by float64 round-off in the loss (~1e-8 absolute in the difference)
SYS_EPS = dict(inc=1e-3, porb=1e-5, t0=1e-6, rp=1e-6, mstar=1e-6)
# gradients below this in both estimates are "zero" (e.g. t0 for an ordinary Map primary, where a common shift of
# the star's and the planet's t0 changes nothing) and are not compared relatively
ATOL_ZERO = 1e-7
# five epochs inside the transit (duration ~0.12 d at porb = 5 d): the planet is in front of
# the disc and away from ingress/egress
T_EPOCHS = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
PROT = 8.0


def _fd_pair(f, x, h):
    """Central differences at h and h/2; returns (value, kink flag)."""
    d1 = (f(x + h) - f(x - h)) / (2 * h)
    d2 = (f(x + h / 2) - f(x - h / 2)) / h
    scale = max(abs(d1), abs(d2), 1e-12)
    return d2, abs(d1 - d2) / scale > KINK_RTOL


def _compare(label, analytic, fd, kink, rows, bad):
    denom = max(abs(analytic), abs(fd), ATOL)
    rel = abs(analytic - fd) / denom
    finite = np.isfinite(analytic) and np.isfinite(fd)
    if not finite:
        status = "NONFINITE"
        bad.append(label)
    elif abs(analytic) < ATOL_ZERO and abs(fd) < ATOL_ZERO:
        status = "OK (both ~0)"
    elif kink:
        status = "kink (not counted)"
    elif rel < RTOL or abs(analytic - fd) < ATOL:
        status = "OK"
    else:
        status = "MISMATCH"
        bad.append(label)
    rows.append((label, analytic, fd, rel, status))


def _print(title, rows):
    print(f"\n{title}")
    print(f"{'quantity':34s} {'analytic':>15s} {'finite-diff':>15s} {'rel_err':>10s}  status")
    for label, a, f, rel, status in rows:
        print(f"{label:34s} {a:15.6e} {f:15.6e} {rel:10.2e}  {status}")


# ======================================================================================
# A. kernel level
# ======================================================================================
def part_a():
    import theano
    import theano.tensor as tt
    import starry

    rows, bad = [], []
    dm_eager = base.build_eager()

    for name, (xo0, yo0, ro0) in EPOCH_SETS.items():
        xo = tt.dvector("xo")
        yo = tt.dvector("yo")
        ro = tt.dscalar("ro")

        dm = starry.DopplerMap(
            ydeg=base.YDEG, udeg=base.UDEG, nt=base.NT, wav=base._rv_wav_grid(), interpolate=True,
            lazy=True, vsini_max=base.VSINI_MAX,
        )
        dm[1] = base.NOMINAL["u1"]
        dm[2] = 0.2
        dm.veq = base.NOMINAL["veq"]
        dm.inc = base.NOMINAL["inc"]
        base._load_line(dm)
        dm.obl = base.NOMINAL["obl"]
        flux = dm.flux(theta=base.THETA, xo=xo, yo=yo, ro=ro)
        loss = tt.sum(flux ** 2)
        g_xo, g_yo, g_ro = theano.grad(loss, [xo, yo, ro])
        fn = theano.function([xo, yo, ro], [g_xo, g_yo, g_ro])
        a_xo, a_yo, a_ro = fn(xo0, yo0, float(ro0))

        def loss_np(xo_, yo_, ro_):
            return float(np.sum(np.asarray(dm_eager.flux(theta=base.THETA, xo=xo_, yo=yo_, ro=ro_)) ** 2))

        for i in range(base.NT):
            def f_x(v, i=i):
                x = xo0.copy(); x[i] = v
                return loss_np(x, yo0, ro0)

            def f_y(v, i=i):
                y = yo0.copy(); y[i] = v
                return loss_np(xo0, y, ro0)

            fd, kink = _fd_pair(f_x, xo0[i], 1e-6)
            _compare(f"[{name}] d/dxo[{i}] (xo={xo0[i]:+.2f}, yo={yo0[i]:+.2f})", float(a_xo[i]), fd, kink, rows, bad)
            fd, kink = _fd_pair(f_y, yo0[i], 1e-6)
            _compare(f"[{name}] d/dyo[{i}]", float(a_yo[i]), fd, kink, rows, bad)
        fd, kink = _fd_pair(lambda v: loss_np(xo0, yo0, v), float(ro0), 1e-6)
        _compare(f"[{name}] d/dro (ro={ro0})", float(a_ro), fd, kink, rows, bad)

    _print("A. DopplerMap.flux gradients w.r.t. the occultor (xo, yo, ro)", rows)
    return bad, rows


# ======================================================================================
# B-E. system level
# ======================================================================================
def _doppler_dm(lazy):
    import starry

    dm = starry.DopplerMap(
        ydeg=base.YDEG, udeg=base.UDEG, nt=len(T_EPOCHS), wav=base._rv_wav_grid(), interpolate=True,
        lazy=lazy, vsini_max=base.VSINI_MAX,
    )
    dm[1] = base.NOMINAL["u1"]
    dm[2] = 0.2
    dm.veq = base.NOMINAL["veq"]
    dm.inc = base.NOMINAL["inc"]
    base._load_line(dm)
    dm.obl = base.NOMINAL["obl"]
    return dm


def _system(primary_map, p, lazy, planet_nt_nw=None, **system_kwargs):
    """A Primary + one massless Secondary with orbit parameters p (numbers or Theano scalars)."""
    import starry
    from astropy import units as u

    star = starry.Primary(
        primary_map, prot=PROT, r=1.0, m=p["mstar"], t0=p["t0"], theta0=0.0,
        length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
    )
    if planet_nt_nw is None:
        planet_map = starry.Map(ydeg=1, amp=0.0, lazy=lazy)
    else:
        planet_map = starry.Map(ydeg=1, amp=0.0, nt=planet_nt_nw[0], nw=planet_nt_nw[1], lazy=lazy)
    planet = starry.Secondary(
        planet_map, r=p["rp"], porb=p["porb"], inc=p["inc"], m=0.0, t0=p["t0"], ecc=0.0, w=90.0,
        length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
    )
    return starry.System(star, planet, **system_kwargs)


def _lazy_grads(kind, orbit_grad):
    """Theano gradients of sum(flux^2) w.r.t. each orbital parameter, one compiled function per parameter."""
    import theano
    import theano.tensor as tt
    import starry

    out = {}
    for name in SYS_NOMINAL:
        free = tt.dscalar(name + "_free")
        p = dict(SYS_NOMINAL)
        p[name] = free
        if kind == "doppler":
            dm = _doppler_dm(lazy=True)
            sysm = _system(dm, p, lazy=True, planet_nt_nw=(len(T_EPOCHS), base.NW), orbit_grad=orbit_grad)
        else:
            m = starry.Map(ydeg=1, udeg=2, lazy=True)
            m[1] = base.NOMINAL["u1"]
            m[2] = 0.2
            sysm = _system(m, p, lazy=True)
        flux = sysm.flux(t=T_EPOCHS)
        loss = tt.sum(flux ** 2)
        g = theano.grad(loss, free, disconnected_inputs="ignore")
        out[name] = float(theano.function([free], g)(SYS_NOMINAL[name]))
    return out


def _eager_loss(kind, dm_eager, **override):
    import starry

    p = dict(SYS_NOMINAL)
    p.update(override)
    if kind == "doppler":
        sysm = _system(dm_eager, p, lazy=False, planet_nt_nw=(len(T_EPOCHS), base.NW))
    else:
        m = starry.Map(ydeg=1, udeg=2, lazy=False)
        m[1] = base.NOMINAL["u1"]
        m[2] = 0.2
        sysm = _system(m, p, lazy=False)
    return float(np.sum(np.asarray(sysm.flux(t=T_EPOCHS)) ** 2))


def _system_part(kind, title):
    rows, bad = [], []
    dm_eager = _doppler_dm(lazy=False) if kind == "doppler" else None
    grads = _lazy_grads(kind, orbit_grad=True)
    for name, nominal in SYS_NOMINAL.items():
        fd, kink = _fd_pair(lambda v, name=name: _eager_loss(kind, dm_eager, **{name: v}), nominal, SYS_EPS[name])
        _compare(f"d/d{name} (nominal {nominal})", grads[name], fd, kink, rows, bad)
    _print(title, rows)
    return bad, rows


def part_b():
    return _system_part("doppler", "B. System with a DopplerMap primary, orbit_grad=True: orbital gradients")


def part_c():
    return _system_part("ordinary", "C. Control: System with an ordinary Map primary: orbital gradients")


def part_d():
    g = _lazy_grads("doppler", orbit_grad=False)
    rows, bad = [], []
    # t0 is not checked here: the same t0 also sets the star's rotation phase, which is connected by design
    for name in ("inc", "porb", "mstar"):
        ok = g[name] == 0.0
        rows.append((f"d/d{name} (orbit_grad=False)", g[name], 0.0, abs(g[name]), "OK" if ok else "MISMATCH"))
        if not ok:
            bad.append(name)
    _print("D. orbit_grad=False: the planet's position must stay disconnected (gradient exactly 0)", rows)
    return bad, rows


def part_e():
    g = _lazy_grads("doppler", orbit_grad="xy")
    rows, bad = [], []
    ok = g["rp"] == 0.0
    rows.append(("d/drp (orbit_grad='xy')", g["rp"], 0.0, abs(g["rp"]), "OK" if ok else "MISMATCH"))
    if not ok:
        bad.append("rp")
    ok = g["inc"] != 0.0 and np.isfinite(g["inc"])
    rows.append(("d/dinc (orbit_grad='xy')", g["inc"], float("nan"), float("nan"), "OK (non-zero)" if ok else "MISMATCH"))
    if not ok:
        bad.append("inc")
    _print("E. orbit_grad='xy': position gradients on, planet-radius gradient off", rows)
    return bad, rows


def main(argv):
    parts = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}
    selected = [a.upper() for a in argv] or list(parts)
    failures = {}
    for key in selected:
        bad, rows = parts[key]()
        n_kink = sum(1 for r in rows if r[4].startswith("kink"))
        n_ok = sum(1 for r in rows if r[4].startswith("OK"))
        print(f"  part {key}: {n_ok} OK, {n_kink} kink (excluded), {len(bad)} failing")
        if bad:
            failures[key] = bad
    if failures:
        print(f"\nFAILURES: {failures}\nDo not use orbit_grad for sampling until these are resolved.")
        raise SystemExit(1)
    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
