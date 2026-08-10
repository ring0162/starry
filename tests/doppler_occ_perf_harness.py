"""
Performance harness for DopplerMap flux-during-occultation compilation and
evaluation speed (Phase 0.1 of the DopplerMap occultation speedup plan).

Standalone -- no pm.Model, no NUTS. Times, separately:
  (a) graph construction  (building the System.flux() symbolic expression)
  (b) theano.function compile (forward pass)
  (c) first numerical eval (forward pass)
  (d) building + compiling + evaluating a gradient w.r.t. one free scalar
      parameter that feeds into the occultation kernel machinery (veq is
      used as the stand-in here; obliquity would exercise the same
      get_kT_occ/get_rT_occ graph, so this is representative of the
      "differentiate through occultation" cost regardless of which
      specific parameter is free)

Matches the DopplerMap configuration used in tests/ccf_model2025.py's
`fit_all` model: ydeg=9, nt=31 (default; override via --nt), wav grid from
rv_grid = arange(-100, 100.5, 0.5) (401 points), interpolate=True.

Usage:
    python tests/doppler_occ_perf_harness.py [--nt N] [--ydeg N] [--skip-grad]

Iterate at small --nt (e.g. 4) for fast turnaround; confirm at the real
scale (--nt 31, the ccf_model2025.py default) for final numbers. Run this
unmodified across environments/configs (Intel-emulated vs arm64-native
conda envs, before/after removing tests/ccf_model2025.py's
optimizer='None' override, fast_compile vs fast_run, etc.) to get
directly comparable timings for Phase 1 of the speedup plan.
"""

import argparse
import contextlib
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--nt", type=int, default=31,
        help="Number of epochs (default: 31, matching ccf_model2025.py)",
    )
    p.add_argument(
        "--ydeg", type=int, default=9,
        help="Spherical harmonic degree (default: 9, matching ccf_model2025.py)",
    )
    p.add_argument("--udeg", type=int, default=2, help="Limb-darkening degree (default: 2)")
    p.add_argument(
        "--skip-grad", action="store_true",
        help="Skip the gradient timing stage (forward-only timing)",
    )
    return p.parse_args()


@contextlib.contextmanager
def timed(label, results):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    results[label] = dt
    print(f"  [{dt:8.2f}s] {label}")


def main():
    args = parse_args()
    results = {}

    print(f"Config: nt={args.nt}, ydeg={args.ydeg}, udeg={args.udeg}")
    print("Stage timings:")

    with timed("import starry / theano", results):
        import theano
        import theano.tensor as tt
        import starry
        import astropy.units as u
        import astropy.constants as const

    with timed("build DopplerMap + System + flux() graph (graph construction)", results):
        rv_grid = np.arange(-100, 100.5, 0.5)
        wav_grid = 1e4 * (1 + rv_grid / const.c.to("km/s").value)
        nt = args.nt

        starmap = starry.DopplerMap(
            ydeg=args.ydeg, udeg=args.udeg, nt=nt, wav=wav_grid,
            interpolate=True, lazy=True,
        )
        planetmap = starry.Map(ydeg=1, amp=0.0, nt=nt, nw=len(wav_grid), lazy=True)

        wav_grid_pad = starmap.wav0
        intr_spec = 1.0 * (
            1 - 0.6 * tt.exp(-0.5 * ((wav_grid_pad - 10000.0) / 0.15) ** 2)
        )
        starmap.load(spectrum=intr_spec)

        starmap.amp = tt.as_tensor_variable(1.0)
        starmap.veq = 24.0 * 1000.0
        starmap.obl = 0.0
        starmap.inc = 90.0

        star = starry.Primary(
            starmap, prot=2.24, r=1.08, m=1.094, t0=0.0, theta0=0.0,
            length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
        )
        planet = starry.Secondary(
            planetmap, r=0.0253 * 1.08, porb=7.713057, inc=87.14,
            m=0.0, t0=0.0, ecc=0.32, w=23.64034616,
            length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
        )
        system = starry.System(star, planet)

        times_all = np.linspace(-0.14, 0.09, nt)
        flux_graph = system.flux(t=times_all, total=True)

    with timed("compile forward theano.function", results):
        forward_fn = theano.function([], flux_graph)

    with timed("first forward eval", results):
        flux_val = forward_fn()

    print(f"  flux shape: {np.asarray(flux_val).shape}")

    if not args.skip_grad:
        with timed(
            "build gradient graph (d loss / d veq, through the occultation kernel)",
            results,
        ):
            # veq is used as a stand-in differentiated scalar: it feeds into
            # vsini -> get_x -> get_rT_occ exactly like obliquity would (via
            # the xo/yo rotation), so this exercises the same graph-size cost
            # a differentiated obliquity would, without needing the full
            # planet-occultor wiring duplicated here.
            veq_free = tt.dscalar("veq_free")
            starmap2 = starry.DopplerMap(
                ydeg=args.ydeg, udeg=args.udeg, nt=nt, wav=wav_grid,
                interpolate=True, lazy=True,
            )
            starmap2.load(spectrum=intr_spec)
            starmap2.amp = tt.as_tensor_variable(1.0)
            starmap2.veq = veq_free
            starmap2.obl = 0.0
            starmap2.inc = 90.0
            star2 = starry.Primary(
                starmap2, prot=2.24, r=1.08, m=1.094, t0=0.0, theta0=0.0,
                length_unit=u.Rsun, mass_unit=u.Msun, time_unit=u.day,
            )
            system2 = starry.System(star2, planet)
            flux_graph2 = system2.flux(t=times_all, total=True)
            loss = tt.sum(flux_graph2 ** 2)
            grad_graph = theano.grad(loss, veq_free)

        with timed("compile gradient theano.function", results):
            grad_fn = theano.function([veq_free], grad_graph)

        with timed("first gradient eval", results):
            grad_val = grad_fn(24000.0)

        print(f"  d(loss)/d(veq) = {grad_val}")

    total = sum(results.values())
    print(f"\nTotal wall-clock: {total:.2f}s")
    print("\nSummary (for comparing across environments/configs):")
    for label, dt in results.items():
        print(f"  {dt:10.2f}s  {label}")


if __name__ == "__main__":
    main()
