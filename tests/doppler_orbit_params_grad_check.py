"""
Finite-difference gradient checks for the orbit/stellar parameters that the global starry RM model
(starry_rm_global_model.py) differentiates through the planet's track, beyond the ones already covered by
doppler_orbit_grad_check.py (inc, porb, t0, planet radius, stellar mass):

    ecc, omega (argument of periastron), rho_star, r_star, ror (= planet radius / r_star)

In the model these are Theano tensors that reach starry through
    m_star = f(rho_star, r_star)                       -> Primary(m=...) and the orbit scale
    aor    = (G m_star P^2 / 4 pi^2)^(1/3) / r_star    -> inclination from the impact parameter b
    inc    = arccos(b / aor (1 + e sin w) / (1 - e^2)) -> Secondary(inc=...)
    Primary(r=r_star), Secondary(r=ror * r_star, ecc=ecc, w=omega) -> System(orbit_grad=True)
and (with fit_ecc=True) ecc, omega come from the unit-disk point ecs = (sqrt(e) cos w, sqrt(e) sin w).

Parts (select with the command line, default all):

  F. library level, orbit_grad=True: d/d(ecc, w, r_star (Primary r), planet radius, stellar mass) of a lazy System
     with a DopplerMap primary, against finite differences of the greedy System, at three eccentric orbits.
  G. glue, no starry: d/d(rho_star, r_star, ecc, omega, b) of m_star, a/R*, the inclination and (ecc, omega)
     from ecs, built exactly like the model builds them, against numpy finite differences.
  H. composite: ONE lazy graph from the model's own parameters (rho_star, r_star, ror, ecc, omega, b, porb, t0)
     to sum(flux^2); every gradient against finite differences of the same compiled function (so this tests that
     the gradient is the derivative of the forward function, not the forward function itself).
  I. composite in the model's eccentricity parametrisation: ecs1 = sqrt(e) cos w, ecs2 = sqrt(e) sin w.
  J. controls and robustness: orbit_grad=False (gradients of ecc, omega, rho_star, b, porb must be exactly 0;
     the ror gradient stays connected and equal to the orbit_grad=True one),
     orbit_grad="xy" (ror gradient exactly 0, others as in H), and a sweep of eccentricities and
     arguments of periastron checking that every gradient is finite.

The loss is sum(flux**2). As in doppler_orbit_grad_check.py, the occultation integral is only piecewise smooth, so
each finite difference is taken at steps h and h/2 and an entry is "kink" (not counted) if they disagree by more than
1e-2; the pass criterion is that every non-kink comparison agrees within RTOL and nothing is NaN/inf.

Keep the glue formulas (m_star, a/R*, inclination, ecs map) in sync with starry_rm_global_model.py.

Usage:
    python tests/doppler_orbit_params_grad_check.py            # all parts
    python tests/doppler_orbit_params_grad_check.py F H        # selected parts
Paste the printed tables (and `pip list | grep -i -E "theano|pymc3|exoplanet"`, `git rev-parse HEAD`) back for review.
"""

import sys
import time

import numpy as np

import doppler_occ_grad_check as base
import doppler_orbit_grad_check as og

RTOL = og.RTOL
ATOL_ZERO = og.ATOL_ZERO
T_EPOCHS = og.T_EPOCHS
PROT = og.PROT
T_REF = 0.3  # rotation reference time, away from t0 so theta0 depends on t0 as in the model
ECC_MAX = 1.0 - 1e-6

from astropy import constants as _const, units as _u

# Rsun^3 / Msun / day^2: the same constant starry uses (~2942)
G_GRAV = float(_const.G.to(_u.Rsun ** 3 / _u.Msun / _u.day ** 2).value)
R_SUN_CM = float(_u.Rsun.to("cm"))
M_SUN_G = float(_const.M_sun.to("g").value)

# nominal orbits: (label, dict). rho_star in g/cm^3 gives ~1 Msun at r_star = 1 Rsun
ORBITS = {
    "e=0.32 w=23.6 (TIC455-like)": dict(rho=1.41, rstar=1.0, ror=0.10, ecc=0.32, omega=np.radians(23.6), b=0.2, porb=5.0, t0=0.0),
    "e=0.05 w=90": dict(rho=1.41, rstar=1.0, ror=0.10, ecc=0.05, omega=np.radians(90.0), b=0.3, porb=5.0, t0=0.0),
    "e=0.60 w=60": dict(rho=1.41, rstar=1.0, ror=0.10, ecc=0.60, omega=np.radians(60.0), b=0.2, porb=5.0, t0=0.0),
}
# finite-difference steps (absolute); inc-like sensitivities are tiny, hence the generous steps for b
STEPS = dict(rho=1e-5, rstar=1e-6, ror=1e-6, ecc=1e-6, omega=1e-5, b=2e-4, porb=1e-5, t0=1e-6,
             ecs1=1e-5, ecs2=1e-5, w=1e-4, rp=1e-6, mstar=1e-6)


def mstar_np(rho, rstar):
    return rho * (4.0 / 3.0 * np.pi * (rstar * R_SUN_CM) ** 3) / M_SUN_G


def aor_np(m, porb, rstar):
    return (G_GRAV * m * porb ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / rstar


def inc_np(b, aor, ecc, omega):
    return np.degrees(np.arccos(b / aor * (1.0 + ecc * np.sin(omega)) / (1.0 - ecc ** 2)))


def _theano():
    import theano
    import theano.tensor as tt

    return theano, tt


# ======================================================================================
# F. library level
# ======================================================================================
def _lib_system(dm, p, lazy, **system_kwargs):
    import starry
    from astropy import units as u

    star = starry.Primary(
        dm, prot=PROT, r=p["rstar"], m=p["mstar"], t0=p["t0"], theta0=-360.0 * (T_REF - p["t0"]) / PROT,
        length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
    )
    planet_map = starry.Map(ydeg=1, amp=0.0, nt=len(T_EPOCHS), nw=base.NW, lazy=lazy)
    planet = starry.Secondary(
        planet_map, r=p["rp"], porb=p["porb"], inc=p["inc"], m=0.0, t0=p["t0"], ecc=p["ecc"], w=p["w"],
        length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
    )
    return starry.System(star, planet, **system_kwargs)


def _lib_nominal(o):
    m = float(mstar_np(o["rho"], o["rstar"]))
    aor = float(aor_np(m, o["porb"], o["rstar"]))
    return dict(
        rstar=o["rstar"], mstar=m, porb=o["porb"], t0=o["t0"], rp=o["ror"] * o["rstar"],
        inc=float(inc_np(o["b"], aor, o["ecc"], o["omega"])), ecc=o["ecc"], w=float(np.degrees(o["omega"])),
    )


def part_f():
    theano, tt = _theano()
    rows, bad = [], []
    for oname, o in ORBITS.items():
        nom = _lib_nominal(o)
        dm_eager = og._doppler_dm(lazy=False)
        for name in ("ecc", "w", "rstar", "rp", "mstar"):
            free = tt.dscalar(name + "_free")
            p = dict(nom)
            p[name] = free
            dm = og._doppler_dm(lazy=True)
            sysm = _lib_system(dm, p, lazy=True, orbit_grad=True)
            loss = tt.sum(sysm.flux(t=T_EPOCHS) ** 2)
            g = theano.grad(loss, free, disconnected_inputs="ignore")
            analytic = float(theano.function([free], g)(nom[name]))

            def f(v, name=name):
                q = dict(nom)
                q[name] = v
                return float(np.sum(np.asarray(_lib_system(dm_eager, q, lazy=False).flux(t=T_EPOCHS)) ** 2))

            fd, kink = og._fd_pair(f, nom[name], STEPS[name])
            og._compare(f"[{oname}] d/d{name} (={nom[name]:.6g})", analytic, fd, kink, rows, bad)
    og._print("F. System (DopplerMap primary, orbit_grad=True): ecc, w, r_star, planet radius, stellar mass", rows)
    return bad, rows


# ======================================================================================
# G. glue (no starry)
# ======================================================================================
def part_g():
    theano, tt = _theano()
    rows, bad = [], []
    rho, rstar, ecc, omega, b, porb = (tt.dscalar(n) for n in ("rho", "rstar", "ecc", "omega", "b", "porb"))
    m = rho * (4.0 / 3.0 * np.pi * (rstar * R_SUN_CM) ** 3) / M_SUN_G
    aor = (G_GRAV * m * porb ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / rstar
    inc = tt.arccos(b / aor * (1.0 + ecc * tt.sin(omega)) / (1.0 - ecc ** 2)) * 180.0 / np.pi
    ins = [rho, rstar, ecc, omega, b, porb]
    outs = {"m_star": m, "a/R*": aor, "inc (deg)": inc}
    fns = {k: theano.function(ins, [v] + theano.grad(v, ins, disconnected_inputs="ignore")) for k, v in outs.items()}
    np_fn = {
        "m_star": lambda r, s, e, w, b_, P: mstar_np(r, s),
        "a/R*": lambda r, s, e, w, b_, P: aor_np(mstar_np(r, s), P, s),
        "inc (deg)": lambda r, s, e, w, b_, P: inc_np(b_, aor_np(mstar_np(r, s), P, s), e, w),
    }
    names = ["rho", "rstar", "ecc", "omega", "b", "porb"]
    for oname, o in ORBITS.items():
        x0 = [o[n] for n in names]
        for qname in outs:
            res = fns[qname](*x0)
            val, grads = res[0], res[1:]
            assert abs(val - np_fn[qname](*x0)) <= 1e-9 * max(1.0, abs(val)), f"forward mismatch for {qname}"
            for i, n in enumerate(names):
                def f(v, i=i, qname=qname):
                    x = list(x0)
                    x[i] = v
                    return float(np_fn[qname](*x))

                fd, kink = og._fd_pair(f, x0[i], max(STEPS.get(n, 1e-6), 1e-7 * abs(x0[i])))
                og._compare(f"[{oname}] d({qname})/d{n}", float(grads[i]), fd, kink, rows, bad)

    # ecs -> (ecc, omega) as the model does it
    e1, e2 = tt.dscalar("ecs1"), tt.dscalar("ecs2")
    ecc_t = tt.minimum(e1 ** 2 + e2 ** 2, ECC_MAX)
    om_t = tt.arctan2(e2, e1)
    for oname, o in ORBITS.items():
        x0 = [np.sqrt(o["ecc"]) * np.cos(o["omega"]), np.sqrt(o["ecc"]) * np.sin(o["omega"])]
        for qname, expr, npf in (
            ("ecc", ecc_t, lambda a, c: min(a * a + c * c, ECC_MAX)),
            ("omega", om_t, lambda a, c: np.arctan2(c, a)),
        ):
            fn = theano.function([e1, e2], [expr] + theano.grad(expr, [e1, e2]))
            res = fn(*x0)
            for i, n in enumerate(("ecs1", "ecs2")):
                def f(v, i=i, npf=npf):
                    x = list(x0)
                    x[i] = v
                    return float(npf(*x))

                fd, kink = og._fd_pair(f, x0[i], 1e-6)
                og._compare(f"[{oname}] d({qname})/d{n}", float(res[1 + i]), fd, kink, rows, bad)
    og._print("G. glue: m_star, a/R*, inclination from b, and (ecc, omega) from ecs (Theano vs numpy)", rows)
    return bad, rows


# ======================================================================================
# H, I, J. composite
# ======================================================================================
class Composite:
    """sum(flux^2) of the model's lazy System as a function of the model's own parameters, with all gradients.

    ecs=False: parameters (rho, rstar, ror, ecc, omega, b, porb, t0).
    ecs=True : parameters (rho, rstar, ror, ecs1, ecs2, b, porb, t0), ecc/omega derived as in the model.
    """

    def __init__(self, ecs=False, orbit_grad=True):
        theano, tt = _theano()
        import starry
        from astropy import units as u

        self.names = ["rho", "rstar", "ror", "ecs1" if ecs else "ecc", "ecs2" if ecs else "omega", "b", "porb", "t0"]
        ins = [tt.dscalar(n) for n in self.names]
        P = dict(zip(self.names, ins))
        if ecs:
            ecc_t = tt.minimum(P["ecs1"] ** 2 + P["ecs2"] ** 2, ECC_MAX)
            omega_t = tt.arctan2(P["ecs2"], P["ecs1"])
        else:
            ecc_t, omega_t = P["ecc"], P["omega"]
        m = P["rho"] * (4.0 / 3.0 * np.pi * (P["rstar"] * R_SUN_CM) ** 3) / M_SUN_G
        aor = (G_GRAV * m * P["porb"] ** 2 / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / P["rstar"]
        inc = tt.arccos(P["b"] / aor * (1.0 + ecc_t * tt.sin(omega_t)) / (1.0 - ecc_t ** 2)) * 180.0 / np.pi

        dm = og._doppler_dm(lazy=True)
        star = starry.Primary(
            dm, prot=PROT, r=P["rstar"], m=m, t0=P["t0"], theta0=-360.0 * (T_REF - P["t0"]) / PROT,
            length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
        )
        planet_map = starry.Map(ydeg=1, amp=0.0, nt=len(T_EPOCHS), nw=base.NW, lazy=True)
        planet = starry.Secondary(
            planet_map, r=P["ror"] * P["rstar"], porb=P["porb"], inc=inc, m=0.0, t0=P["t0"],
            ecc=ecc_t, w=omega_t * (180.0 / np.pi),
            length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
        )
        system = starry.System(star, planet, orbit_grad=orbit_grad) if orbit_grad else starry.System(star, planet)
        loss = tt.sum(system.flux(t=T_EPOCHS) ** 2)
        grads = [theano.grad(loss, x, disconnected_inputs="ignore") for x in ins]
        self.fn = theano.function(ins, [loss] + grads, on_unused_input="ignore")

    def point(self, o):
        if "ecs1" in self.names:
            vals = dict(o, ecs1=np.sqrt(o["ecc"]) * np.cos(o["omega"]), ecs2=np.sqrt(o["ecc"]) * np.sin(o["omega"]))
        else:
            vals = o
        return [float(vals[n]) for n in self.names]

    def __call__(self, x):
        out = self.fn(*x)
        return float(out[0]), np.array([float(np.asarray(g)) for g in out[1:]])


def _composite_part(title, ecs):
    rows, bad = [], []
    t0 = time.time()
    comp = Composite(ecs=ecs, orbit_grad=True)
    print(f"  (graph compiled in {time.time() - t0:.0f} s)")
    for oname, o in ORBITS.items():
        x0 = comp.point(o)
        loss0, grads = comp(x0)
        print(f"  [{oname}] loss = {loss0:.8g}")
        for i, n in enumerate(comp.names):
            def f(v, i=i):
                x = list(x0)
                x[i] = v
                return comp(x)[0]

            fd, kink = og._fd_pair(f, x0[i], STEPS[n])
            og._compare(f"[{oname}] d/d{n} (={x0[i]:.6g})", float(grads[i]), fd, kink, rows, bad)
    og._print(title, rows)
    return bad, rows


def part_h():
    return _composite_part("H. composite (rho_star, r_star, ror, ecc, omega, b, porb, t0) -> sum(flux^2), orbit_grad=True", ecs=False)


def part_i():
    return _composite_part("I. composite in the model's parametrisation: ecs = (sqrt(e) cos w, sqrt(e) sin w)", ecs=True)


def part_j():
    rows, bad = [], []
    o = ORBITS["e=0.32 w=23.6 (TIC455-like)"]

    # orbit_grad=False: the planet's POSITION is disconnected from every orbit parameter. The radius ratio is not:
    # System only cuts ro = r_planet / r_star for orbit_grad="xy", so d/dror must stay non-zero and equal the
    # orbit_grad=True value.
    c_off = Composite(ecs=False, orbit_grad=False)
    c_on0 = Composite(ecs=False, orbit_grad=True)
    _, g = c_off(c_off.point(o))
    _, g_on = c_on0(c_on0.point(o))
    for n in ("ecc", "omega", "rho", "b", "porb"):
        v = g[c_off.names.index(n)]
        ok = v == 0.0
        rows.append((f"orbit_grad=False d/d{n}", v, 0.0, abs(v), "OK" if ok else "MISMATCH"))
        if not ok:
            bad.append(f"off:{n}")
    i_ror = c_off.names.index("ror")
    rel = abs(g[i_ror] - g_on[i_ror]) / max(abs(g_on[i_ror]), 1e-30)
    ok = np.isfinite(g[i_ror]) and g[i_ror] != 0.0 and rel < 1e-6
    rows.append(("orbit_grad=False d/dror (== True)", g[i_ror], g_on[i_ror], rel, "OK (still connected)" if ok else "MISMATCH"))
    if not ok:
        bad.append("off:ror")
    for n in ("rstar", "t0"):  # informational: r_star / t0 also act through the star's own rotation phase and scale
        rows.append((f"orbit_grad=False d/d{n} (info)", g[c_off.names.index(n)], float("nan"), float("nan"), "info"))

    # orbit_grad="xy": the radius ratio is cut, the position parameters are not
    c_xy = Composite(ecs=False, orbit_grad="xy")
    _, g = c_xy(c_xy.point(o))
    v = g[c_xy.names.index("ror")]
    rows.append(("orbit_grad='xy' d/dror", v, 0.0, abs(v), "OK" if v == 0.0 else "MISMATCH"))
    if v != 0.0:
        bad.append("xy:ror")
    for n in ("ecc", "omega", "rho", "b"):
        v = g[c_xy.names.index(n)]
        ok = np.isfinite(v) and v != 0.0
        rows.append((f"orbit_grad='xy' d/d{n}", v, float("nan"), float("nan"), "OK (non-zero)" if ok else "MISMATCH"))
        if not ok:
            bad.append(f"xy:{n}")

    # finiteness sweep (orbit_grad=True)
    c_on = Composite(ecs=False, orbit_grad=True)
    n_bad = n_all = 0
    for ecc in (0.001, 0.05, 0.2, 0.32, 0.5, 0.7):
        for w_deg in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            q = dict(o, ecc=ecc, omega=np.radians(w_deg))
            try:
                loss0, g = c_on(c_on.point(q))
                finite = np.isfinite(loss0) and np.all(np.isfinite(g))
            except Exception as exc:  # e.g. the transit leaves the grid (arccos > 1) is NaN, not an exception
                finite = False
                print(f"  sweep e={ecc} w={w_deg}: exception {exc!r}")
            n_all += 1
            if not finite:
                n_bad += 1
                print(f"  sweep e={ecc} w={w_deg}: non-finite loss/gradient")
    rows.append((f"finite sweep ({n_all} (e, w) pairs)", float(n_bad), 0.0, float(n_bad), "OK" if n_bad == 0 else "MISMATCH"))
    if n_bad:
        bad.append("sweep")
    og._print("J. controls (orbit_grad False / 'xy') and finiteness sweep over (ecc, omega)", rows)
    return bad, rows


def main(argv):
    parts = {"F": part_f, "G": part_g, "H": part_h, "I": part_i, "J": part_j}
    selected = [a.upper() for a in argv] or list(parts)
    failures = {}
    for key in selected:
        t0 = time.time()
        bad, rows = parts[key]()
        n_kink = sum(1 for r in rows if r[4].startswith("kink"))
        n_ok = sum(1 for r in rows if r[4].startswith("OK"))
        print(f"  part {key}: {n_ok} OK, {n_kink} kink (excluded), {len(bad)} failing  [{time.time() - t0:.0f} s]")
        if bad:
            failures[key] = bad
    if failures:
        print(f"\nFAILURES: {failures}")
        print("Failing ecc/omega/rho_star/r_star/ror gradients: do not free those parameters in the global model "
              "(fit_ecc=False; fix r_star/rho_star; orbit_grad='xy' to freeze ror) until resolved.")
        raise SystemExit(1)
    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main(sys.argv[1:])
