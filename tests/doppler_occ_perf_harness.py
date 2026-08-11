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
        [--theano-mode {fast_compile,fast_run,fast_run_no_fusion}]
        [--timeout SECONDS]

Iterate at small --nt (e.g. 4) for fast turnaround; confirm at the real
scale (--nt 31, the ccf_model2025.py default) for final numbers. Run this
unmodified across environments/configs (Intel-emulated vs arm64-native
conda envs, before/after removing tests/ccf_model2025.py's
optimizer='None' override, fast_compile vs fast_run, etc.) to get
directly comparable timings for Phase 1 of the speedup plan.

--theano-mode controls which Theano optimizer/mode is used, for the
Phase 1.3 diagnostic (is the graph-size-vs-compile-time tradeoff that
motivated starry/kepler.py's forced fast_compile setting reproducible on
this machine/architecture, or was it emulation-specific?):
  fast_compile      (default) -- starry's own forced setting, unchanged.
  fast_run          -- disables starry's override (via the
                       STARRY_DOPPLER_FORCE_FAST_COMPILE=0 env var) and
                       lets Theano's own out-of-the-box default apply.
                       This is the setting that originally caused C
                       compilation failures in February -- start with a
                       SMALL --nt/--ydeg here and scale up incrementally,
                       do not jump straight to --nt 31.
  fast_run_no_fusion -- disables starry's override and installs
                       theano.compile.get_default_mode().excluding("fusion")
                       as the cached default mode (the exact technique
                       from git history commit 0231d73) -- keeps most
                       fast_run optimizations but skips the fusion pass
                       specifically implicated in the original failure.

--timeout SECONDS applies a per-stage wall-clock limit (via SIGALRM, so
Unix/macOS only) so a runaway fast_run compile attempt can't hang
indefinitely -- it raises a clear TimeoutError and exits instead. Strongly
recommended whenever --theano-mode is not fast_compile.
"""

import argparse
import contextlib
import os
import signal
import time

import numpy as np


THEANO_MODE_CHOICES = ("fast_compile", "fast_run", "fast_run_no_fusion")


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
    p.add_argument(
        "--theano-mode", choices=THEANO_MODE_CHOICES, default="fast_compile",
        help="Which Theano optimizer/mode to use (default: fast_compile, "
             "starry's own forced setting). See module docstring for details.",
    )
    p.add_argument(
        "--timeout", type=float, default=None,
        help="Per-stage wall-clock timeout in seconds (SIGALRM-based, Unix/macOS "
             "only). Recommended whenever --theano-mode is not fast_compile.",
    )
    return p.parse_args()


class StageTimeout(Exception):
    pass


@contextlib.contextmanager
def timed(label, results, timeout=None):
    def _on_alarm(signum, frame):
        raise StageTimeout(
            f"Stage '{label}' did not complete within {timeout:.0f}s -- aborting."
        )

    old_handler = None
    if timeout is not None:
        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(timeout) + 1)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if timeout is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    dt = time.perf_counter() - t0
    results[label] = dt
    print(f"  [{dt:8.2f}s] {label}")


def configure_theano_mode(mode):
    """Must be called AFTER importing theano but BEFORE constructing any
    DopplerMap/System, so the setting is in place before starry's own
    System.__init__ forced-mode logic runs (see starry/kepler.py)."""
    import theano
    import theano.compile.mode as _tcm

    if mode == "fast_compile":
        return  # starry's own forced setting applies unchanged

    # Disable starry's forced override so our own setting below sticks.
    os.environ["STARRY_DOPPLER_FORCE_FAST_COMPILE"] = "0"

    if mode == "fast_run":
        pass  # Theano's own out-of-the-box default is fast_run; nothing to set.
    elif mode == "fast_run_no_fusion":
        _mode = theano.compile.get_default_mode()
        _mode = _mode.excluding("fusion")
        _tcm.instantiated_default_mode = _mode


def main():
    args = parse_args()
    results = {}
    timeout = args.timeout

    print(f"Config: nt={args.nt}, ydeg={args.ydeg}, udeg={args.udeg}, theano_mode={args.theano_mode}")
    if args.theano_mode != "fast_compile" and timeout is None:
        print(
            "  WARNING: --theano-mode is not fast_compile and no --timeout was "
            "given. This is the setting that originally caused C compilation "
            "failures/hangs -- strongly consider Ctrl-C-ing and re-running with "
            "--timeout, and a small --nt/--ydeg first."
        )
    print("Stage timings:")

    with timed("import starry / theano", results, timeout):
        import theano
        import theano.tensor as tt
        import starry
        import astropy.units as u
        import astropy.constants as const

        configure_theano_mode(args.theano_mode)

    with timed("build DopplerMap + System + flux() graph (graph construction)", results, timeout):
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

    with timed("compile forward theano.function", results, timeout):
        forward_fn = theano.function([], flux_graph)

    with timed("first forward eval", results, timeout):
        flux_val = forward_fn()

    print(f"  flux shape: {np.asarray(flux_val).shape}")

    if not args.skip_grad:
        with timed(
            "build gradient graph (d loss / d veq, through the occultation kernel)",
            results, timeout,
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

        with timed("compile gradient theano.function", results, timeout):
            grad_fn = theano.function([veq_free], grad_graph)

        with timed("first gradient eval", results, timeout):
            grad_val = grad_fn(24000.0)

        print(f"  d(loss)/d(veq) = {grad_val}")

    total = sum(results.values())
    print(f"\nTotal wall-clock: {total:.2f}s")
    print("\nSummary (for comparing across environments/configs):")
    for label, dt in results.items():
        print(f"  {dt:10.2f}s  {label}")


if __name__ == "__main__":
    main()
