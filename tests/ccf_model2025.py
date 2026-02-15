
# os.environ['THEANO_FLAGS'] = 'cxx=/usr/bin/clang++,optimizer=fast_compile,exception_verbosity=high'
import sys
sys.setrecursionlimit(50000)  # Default is 1000, increase to 10000
print(f"Recursion limit: {sys.getrecursionlimit()}")

import os
os.environ['THEANO_FLAGS'] = 'exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'optimizer=None'

import theano
theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'None'
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy as sp
from scipy import linalg
from scipy.optimize import minimize
import scipy.signal as ss
import scipy.interpolate
from astropy import units as u
from astropy import constants as const
import astropy.units as u
from astropy.io import fits
import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as xo
import os
import starry
from corner import corner
# import theano
# theano.config.cxx = ''
import theano.tensor as tt
from tqdm.auto import tqdm
import arviz as az
from theano import printing
import matplotlib.patches as mpatches

if starry.compat.USE_AESARA:
    theano_config = dict(aesara_config=dict(compute_test_value="ignore"))
else:
    theano_config = dict(theano_config=dict(compute_test_value="ignore"))

np.set_printoptions(threshold=np.inf,  # Print all elements
                    linewidth=200,      # Wider lines
                    precision=8,        # Decimal precision
                    suppress=True) 

pl = {
    "porb": 7.713057,               #From Alexis'analysis (Ed's email June 9) [-0.0000327 +0.0000326] - days
    "porb_unc": 0.000021,
    "R": 3.00,
    "R_unc": 0.30,                      #From Alexis' Proposal - Earth Radii
    "t0": 2459037.8704,   
    "t0_unc": 0.0022,            #From Alexis' Ephemeris (Ed's email, June 9) [-0.0021035 +0.0022118] - BJD
    "tdur": 2.47,
    "tdur_unc": 0.43,                   #From Alexis'analysis (Ed
    "ecc": 0.32,
    "ecc_unc": 0.20,                     #From Alexis'analysis (Ed)
    "omega": 17.0,
    "omega_unc": 67.0,                  #From Alexis'analysis (Ed)
    "inc": 87.14,
    "inc_unc": 0.17,                    #From Alexis'analysis (Ed)
    "aor": 15.7,
    "aor_unc": 1.6,                     #From Alexis'analysis (Ed
    "ror": 0.0253,

#     "t0_transit": 2460340.72376,
#     "t0_transit_unc": 0.0035     #From exofop - BJD
}
b = pl['aor'] * np.cos(pl['inc'] * np.pi/180.) 
# print("impact parameter (if circ):", b)
b = b * (1 - pl['ecc'] ** 2)/(1 + pl['ecc'] * np.sin(pl['omega'] * np.pi/180.))
# print("impact parameter (ecc):", b)
st = {
    "prot": 2.24,
    "prot_unc": 0.11,                    #From Alexis' analysis (Ed's email, June 9) +/- 0.110 - days
    "vsini": 24.4,     #From exofop - km/s
    "vsini_unc": 1.0,
    "R": 1.08,
    "R_unc": 0.11,                                      #From exofop - solar units
    "M": 1.094,
    "M_unc": 0.024,                                            #From exofop - solar units
    # "inc": 90.0,
    # "inc_unc": 3.,                    #From Alexis' analysis (Ed's email, June 9), falls to zero probablility at 75 degrees - degrees
    # "LD": [0.7211, 0.0354],             #From Ed on Slack, June 17
    # "LD_unc": [0.04, 0.088],
    # "LDq": [0.57229225, 0.4766027759418374],         #Reparameterised to q according to Kipping 2013
    # "LDq_unc": [0.007312659423219434, 0.011088353442533332]
}

# omega = np.linspace(0, 360, 1000)
# b = pl['aor'] * np.cos(pl['inc'] * np.pi/180.) 
# # # print("impact parameter (if circ):", b)
# b = b * (1 - pl['ecc'] ** 2)/(1 + pl['ecc'] * np.sin(omega * np.pi/180.))
# tdur = pl['porb'] * 24 / np.pi * np.arcsin(np.sqrt(1 + 0.0253 **2 - b**2) / np.sin(pl['inc'] * np.pi/180.) / (pl['aor'])) * np.sqrt(1 - pl['ecc']**2) / (1 + pl['ecc'] * np.sin(omega * np.pi/180.))

from scipy.optimize import fsolve
def f(x):
    b = pl['aor'] * np.cos(pl['inc'] * np.pi/180.)
    b = b * (1 - pl['ecc'] ** 2)/(1 + pl['ecc'] * np.sin(x * np.pi/180.))
    return pl['porb'] * 24 / np.pi * np.arcsin(np.sqrt(1 + 0.0253 **2 - b**2) / np.sin(pl['inc'] * np.pi/180.) / (pl['aor'])) * np.sqrt(1 - pl['ecc']**2) / (1 + pl['ecc'] * np.sin(x * np.pi/180.)) - pl['tdur']
solution = fsolve(f, x0=20.0)  # x0 is initial guess
print(solution)

# fig, ax = plt.subplots()
# ax.plot(omega, tdur)
# ax.axhline(pl['tdur'], color='k', ls='--')
# ax.set_xlabel('Argument of periastron (degrees)')
# ax.set_ylabel('Transit Duration (hours)')
# fig.savefig("transit_duration_vs_omega.png")

with open('/Users/andrew/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/transit1.pkl', 'rb') as f:
    data = pickle.load(f)

rv_grid = np.arange(-100, 100.5, 0.5)

# times = np.zeros(32)
# spec_map = np.zeros((32, len(rv_grid)))
# spec_map_err = np.zeros((32, len(rv_grid)))
# normed_ccfs = np.zeros((32, len(rv_grid)))
# normed_ccf_errs = np.zeros((32, len(rv_grid)))
# mean_normed_ccf_num = np.zeros(len(rv_grid))
# mean_normed_ccf_den = np.zeros(len(rv_grid))
# mean_normed_ccf = np.zeros(len(rv_grid))

# for i in range(32):
#     times[i] = data[i]['time']
#     spec_map[i,:] = data[i]['ccf']
#     spec_map_err[i,:] = data[i]['ccf_err']
#     normed_ccfs[i,:] = 1. - data[i]['ccf'] / data[i]['ccf_fit']['continuum']
#     normed_ccf_errs[i,:] = data[i]['ccf_err'] / data[i]['ccf_fit']['continuum']
#     if i in data['out_frames']:
#         mean_normed_ccf_num += normed_ccfs[i,:] / (normed_ccf_errs[i,:]**2)
#         mean_normed_ccf_den += 1. / (normed_ccf_errs[i,:]**2)

# mean_normed_ccf = mean_normed_ccf_num / mean_normed_ccf_den
# mean_normed_ccf_err = np.sqrt(1. / mean_normed_ccf_den)


wav_grid = 1E4 * (1 + rv_grid / const.c.to('km/s').value)

def gaussian_profile(x, ctrst, mu, sigma, cont):
    return cont * (1 - ctrst * np.exp(-0.5 * ((x - mu) / sigma) ** 2))

def gaussian_profile_tt(x, ctrst, mu, sigma, cont):
    return cont * (1 - ctrst * tt.exp(-0.5 * ((x - mu) / sigma) ** 2))

times = np.zeros(32)
spec_map = np.zeros((32, len(wav_grid)))
spec_map_err = np.zeros((32, len(wav_grid)))
observed_resids_grid = np.zeros((32, len(wav_grid)))

for i in range(32):
    times[i] = data[i]['time']
    spec_map[i,:] = data[i]['ccf']/data[i]['ccf_fit']['continuum']
    spec_map_err[i,:] = data[i]['ccf_err']/data[i]['ccf_fit']['continuum']
    observed_resids_grid[i,:] = data['resids_grid'][i]

times_transit = times + 2400000.5 - (pl['t0'] + 143 * pl['porb'])
# out_times = times_transit[data['out_frames']]
# print("phases:", times_transit/st['prot'] * 180./np.pi)

bad_mask = np.ones_like(times_transit, dtype=bool)
bad_mask[30] = False
# out_mask = data['out_frames']
# times_all = times_transit[bad_mask]
spec_map_all = spec_map[bad_mask,:]
spec_map_all_err = spec_map_err[bad_mask,:]
observed_resids_grid = observed_resids_grid[bad_mask,:]
# out_times = times_transit[out_mask]
# spec_map_out = spec_map[out_mask,:]
# spec_map_out_err = spec_map_err[out_mask,:]

times_all = np.array([-0.13843236, -0.12758718, -0.12408308, -0.11323257, -0.10972847, -0.09887237,
            -0.09536818, -0.08452161, -0.08101766, -0.07018954, -0.06668435, -0.05581392,
            -0.05230991, -0.04146606, -0.03796203, -0.02711089, -0.02360503, -0.0127589,
            -0.0092549,   0.00159535,  0.00509934,  0.01594276,  0.0194469,   0.03042782,
            0.03393085,  0.04475181,  0.04825581,  0.05910783,  0.06261193,  0.07343319,  0.08778506])

times_transit = times_all[11:27]

print(len(times_all))

ndeg = 9

tsyn = np.linspace(-pl['porb']/2, pl['porb']/2, 100)

nt = len(times_all)
lazy = True

test_obl= 0.
test_inc = 90.

starmap_all = starry.DopplerMap(ydeg=ndeg, udeg = 2, nt = nt, wav = wav_grid, interpolate=True, lazy=lazy)
starmap_quiet = starry.DopplerMap(ydeg=ndeg, udeg = 2, nt = nt, wav = wav_grid, interpolate=True, lazy=lazy)
starmap_shm = starry.Map(ydeg=ndeg, udeg = 2, lazy=lazy)
planetmap = starry.Map(ydeg=1, amp=0., nt = nt, nw = len(wav_grid), lazy=lazy)

with pm.Model(**theano_config) as fit_intr_priors:

    intr_cont = pm.Uniform("intr_cont", lower = 0.9, upper = 1.1)
    intr_ctrst = pm.Uniform("intr_ctrst", lower = 0.4, upper = 0.8)
    intr_sigma = pm.Uniform("intr_sigma", lower = 0.05, upper = 0.25)
    intr_wav = pm.Uniform("intr_wav", lower = 9999.5, upper = 10000.)

    wav_grid_pad = starmap_all.wav0
    pm.Deterministic("wav_grid_pad", wav_grid_pad)

    intr_spec = gaussian_profile_tt(wav_grid_pad, intr_ctrst, intr_wav, intr_sigma, intr_cont)
    pm.Deterministic("intr_spec", intr_spec)
    starmap_all.load(spectrum = intr_spec)
    starmap_quiet.load(spectrum = intr_spec)

    spot1_contrast = pm.Uniform("spot_contrast", lower = 0.01, upper = 0.99)
    spot1_radius = pm.Uniform("spot_radius", lower = 5., upper = 75.)  # degrees
    spot1_lon = pm.Uniform("spot_lon", lower = -80, upper = 20.)  # degrees
    spot1_lat = pm.Uniform("spot_lat", lower = -60., upper = 60.)  # degrees

    starmap_shm.spot(contrast = spot1_contrast, radius = spot1_radius, lon = spot1_lon, lat = spot1_lat)
  
    amp = starmap_shm.y[0]
    y = starmap_shm.y[1:]

    starmap_all.amp = amp
    starmap_all[1:, :] = y
    y =pm.Deterministic("y", starmap_shm.y)

    # veq_fit = pm.Normal("veq", mu = 24., sigma = 2.)
    # veq_fit = pm.Uniform("veq", lower = 20., upper = 28.)
    veq_fit = 24.
    starmap_all.veq = veq_fit * 1000.
    # obl_fit = pm.Uniform("obl", lower = -90., upper = 90.)
    obl_fit = 0.
    starmap_all.obl = obl_fit
    # inc_fit = pm.Uniform("inc", lower = 75., upper = 90.)
    inc_fit = 90.
    starmap_all.inc = inc_fit

    prot_fit = st["prot"]
    r_star_fit = st["R"]
    m_star_fit = st["M"]

    star = starry.Primary(
        starmap_all, 
        prot = prot_fit,
        r = r_star_fit,
        m = m_star_fit,
        t0 = 0.,
        theta0 = 0.,
        length_unit = u.Rsun,
        mass_unit = u.Msun,
        time_unit = u.day
    )

    planet_inc = pl['inc']
    planet_r = pl["R"]

    planet = starry.Secondary(
        planetmap,
        r = pl['ror'] * r_star_fit,
        porb = pl['porb'],
        inc = planet_inc,
        m = 0.,
        t0 = 0.,
        ecc = pl['ecc'],
        w = 23.64034616,
        length_unit = u.Rsun,
        mass_unit = u.Msun,
        time_unit = u.day
    )

    system = starry.System(star, planet)
    pos = system.position(t = times_all)
    xp = pm.Deterministic("xp", pos[0][1])
    yp = pm.Deterministic("yp", pos[1][1])
    zp = pm.Deterministic("zp", pos[2][1])
    xs = pm.Deterministic("xs", pos[0][0])
    ys = pm.Deterministic("ys", pos[1][0])
    zs = pm.Deterministic("zs", pos[2][0])
    flux = pm.Deterministic("flux", system.flux(t = times_all, total=True))
    obs = pm.Normal("obs_flux", mu=flux, sd=spec_map_all_err, observed=spec_map_all)

with fit_intr_priors:
    map_soln = pmx.optimize(vars=[intr_cont, intr_ctrst, intr_wav, intr_sigma], start=None)
    # map_soln = pmx.optimize(vars=[veq_fit], start=map_soln)
    # map_soln = pmx.optimize(vars=[intr_sigma], start=map_soln)
    map_soln = pmx.optimize(vars=[spot1_contrast, spot1_radius, spot1_lon, spot1_lat], start=map_soln)
    # map_soln = pmx.optimize(vars=[veq_fit], start=map_soln)

fig, ax = plt.subplots()
ax.plot(map_soln['wav_grid_pad'], map_soln['intr_spec'], label='Intrinsic Line Profile', color='C2')
ax.plot(wav_grid, spec_map_all[0], label='Observed (frame 0)', color='C0')
ax.plot(wav_grid, map_soln['flux'][0], label='Model (frame 0)', color='C1')
ax.set_xlabel("Wavelength")
ax.set_ylabel("Normalized Flux")
ax.set_xlim(wav_grid[0], wav_grid[-1])
ax.legend()
fig.savefig("intrinsic_line_profile_fit.png")

xp = map_soln['xp'] - map_soln['xs']
yp = map_soln['yp'] - map_soln['ys']
zp = map_soln['zp'] - map_soln['zs']

pos_on_disk = np.sqrt(xp**2 + yp**2)

mean_out = np.zeros_like(wav_grid)
mean_out_quiet = np.zeros_like(wav_grid)
j = 0
for i in range(len(times_transit)):
    if i < 12 or i > 27:
        mean_out += map_soln['flux'][i,:]
        # mean_out_quiet += flux_quiet[i,:]
        j += 1
mean_out /= j
mean_out_quiet /= j

fig, ax = plt.subplots()
im = ax.imshow(map_soln['flux'], aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
for i in range(len(times_all)):
    plt.text(wav_grid[0] + 0.1, times_all[i], f"{pos_on_disk[i]:.2f}", va='center', ha='left', color='black', fontsize=6)
ax.axhline(0., color='k', ls='--')
ax.axhline(times_all[12], color='k', ls='--')
ax.axhline(times_all[28], color='k', ls='--')
ax.set_xlabel("Wavelength")
ax.set_ylabel("Time (days)")
plt.colorbar(im, ax = ax, label='Flux')
fig.savefig(f"model_spectral_time_series.png")

# fig, ax = plt.subplots()
# starmap_all.show(ax=ax, projection='moll', grid=False, colorbar=True)
# fig.savefig("model_map.png")
# plt.close(fig)

fig, ax = plt.subplots()
im = ax.imshow(map_soln['flux'] - mean_out, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
ax.axhline(0., color='k', ls='--')
ax.axhline(times_all[12], color='k', ls='--')
ax.axhline(times_all[28], color='k', ls='--')
ax.set_xlabel("Wavelength")
ax.set_ylabel("Time (days)")
plt.colorbar(im, ax = ax, label='Flux')
fig.savefig(f"residual_model_spectral_time_series.png")

fig, ax = plt.subplots()
im = ax.imshow(observed_resids_grid, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
ax.axhline(0., color='k', ls='--')
ax.axhline(times_all[12], color='k', ls='--')
ax.axhline(times_all[28], color='k', ls='--')
ax.set_xlabel("Wavelength")
ax.set_ylabel("Time (days)")
plt.colorbar(im, ax = ax, label='Flux')
fig.savefig(f"observed_residuals_spectral_time_series.png")
plt.close(fig)

for key in ['intr_cont', 'intr_ctrst', 'intr_sigma', 'intr_wav', 'spot_contrast', 'spot_radius', 'spot_lon', 'spot_lat']:
    print(f"{key}: {map_soln[key]}")

snr = 1000.

starmap_all = starry.DopplerMap(ydeg=ndeg, udeg = 2, nt = nt, wav = wav_grid, interpolate=True, lazy=lazy)
starmap_quiet = starry.DopplerMap(ydeg=ndeg, udeg = 2, nt = nt, wav = wav_grid, interpolate=True, lazy=lazy)
starmap_shm = starry.Map(ydeg=ndeg, udeg = 2, lazy=lazy)
planetmap = starry.Map(ydeg=1, amp=0., nt = nt, nw = len(wav_grid), lazy=lazy)

print(f"Recursion limit: {sys.getrecursionlimit()}")
sys.setrecursionlimit(50000)  # Default is 1000, increase to 10000
print(f"Recursion limit: {sys.getrecursionlimit()}")

with pm.Model(**theano_config) as fit_all:

    wav_grid_pad = starmap_all.wav0
    pm.Deterministic("wav_grid_pad", wav_grid_pad)

    # Compute the base profile (fixed, from MAP solution)
    intr_spec_base = gaussian_profile_tt(
        wav_grid_pad,
        map_soln['intr_ctrst'],
        map_soln['intr_wav'],
        map_soln['intr_sigma'],
        map_soln['intr_cont']
    )

    # Extract the prior means for the free region (evaluated as constants at graph build time)
    prior_means = intr_spec_base[135:240]

    # Define the free parameters, centred on the MAP profile values
    line_prof = pm.Normal("line_prof", mu=prior_means, sd=prior_means / snr, shape=105)

    # # Build the full profile with the free region substituted in
    intr_spec = tt.set_subtensor(intr_spec_base[135:240], line_prof)
    # intr_spec = gaussian_profile_tt(wav_grid_pad, map_soln['intr_ctrst'], map_soln['intr_wav'], map_soln['intr_sigma'], map_soln['intr_cont'])
    pm.Deterministic("intr_spec", intr_spec)

    starmap_all.load(spectrum = intr_spec)
    starmap_quiet.load(spectrum = intr_spec)

    spot1_contrast = pm.Uniform("spot_contrast", lower = 0.01, upper = 0.99, testval=map_soln['spot_contrast'])
    spot1_radius = pm.Uniform("spot_radius", lower = 5., upper = 75., testval=map_soln['spot_radius'])  # degrees
    spot1_lon = pm.Uniform("spot_lon", lower = -80, upper = 20., testval=map_soln['spot_lon'])  # degrees
    spot1_lat = pm.Uniform("spot_lat", lower = -60., upper = 60., testval=map_soln['spot_lat'])  # degrees

    starmap_shm.spot(contrast = spot1_contrast, radius = spot1_radius, lon = spot1_lon, lat = spot1_lat)
  
    amp = starmap_shm.y[0]
    y = starmap_shm.y[1:]

    starmap_all.amp = amp
    starmap_all[1:, :] = y
    y =pm.Deterministic("y", starmap_shm.y)

    # veq_fit = pm.Normal("veq", mu = 24., sigma = 2.)
    # veq_fit = pm.Uniform("veq", lower = 20., upper = 28.)
    veq_fit = 24.
    starmap_all.veq = veq_fit * 1000.
    starmap_quiet.veq = veq_fit * 1000.
    # obl_fit = pm.Uniform("obl", lower = -90., upper = 90.)
    obl_fit = 0.
    starmap_all.obl = obl_fit
    starmap_quiet.obl = obl_fit
    # inc_fit = pm.Uniform("inc", lower = 75., upper = 90.)
    inc_fit = 90.
    starmap_all.inc = inc_fit
    starmap_quiet.inc = inc_fit

    prot_fit = st["prot"]
    r_star_fit = st["R"]
    m_star_fit = st["M"]

    star = starry.Primary(
        starmap_all, 
        prot = prot_fit,
        r = r_star_fit,
        m = m_star_fit,
        t0 = 0.,
        theta0 = 0.,
        length_unit = u.Rsun,
        mass_unit = u.Msun,
        time_unit = u.day
    )

    star_quiet = starry.Primary(
        starmap_quiet,
        prot = prot_fit,
        r = r_star_fit, 
        m = m_star_fit, 
        t0 = 0., 
        theta0 = 0., 
        length_unit = u.Rsun, 
        mass_unit = u.Msun, 
        time_unit = u.day 
    )

    planet_inc = pl['inc']
    planet_r = pl["R"]

    planet = starry.Secondary(
        planetmap,
        r = pl['ror'] * r_star_fit,
        porb = pl['porb'],
        inc = planet_inc,
        m = 0.,
        t0 = 0.,
        ecc = pl['ecc'],
        w = 23.64034616,
        length_unit = u.Rsun,
        mass_unit = u.Msun,
        time_unit = u.day
    )

    system = starry.System(star, planet)
    system_quiet = starry.System(star_quiet, planet)
    flux = pm.Deterministic("flux", system.flux(t = times_all, total=True))
    flux_clean = pm.Deterministic("flux_clean", system_quiet.flux(t = times_all, total=True))
    obs = pm.Normal("obs_flux", mu=flux, sd=spec_map_all_err, observed=spec_map_all)

with fit_all:
    
    # map_soln = pmx.optimize(vars=[veq_fit], start=map_soln)
    # map_soln = pmx.optimize(vars=[intr_sigma], start=map_soln)
    start = {**fit_all.test_point, **map_soln}
    map_soln_all = pmx.optimize(vars=[spot1_contrast, spot1_radius, spot1_lon, spot1_lat], start=start)
    map_soln_all = pmx.optimize(vars=[line_prof], start=map_soln_all)
    # map_soln_all = pmx.optimize(vars=[veq_fit, obl_fit, inc_fit], start=map_soln_all)

with fit_all:
    trace_all = pm.sample(1000, tune=1000, chains=4, cores = 1, return_inferencedata=True)

mean_vals = {}
for var in [
    # 'LD1q', 'LD2q', 
    'veq', 'obl', 'inc', 'line_prof', 'spot_contrast', 'spot_radius', 'spot_lon', 'spot_lat', 'flux', 'flux_clean'
    ]:
    samps = trace_all.get_values(varname=var, combine=True)
    mean_vals[var] = dict.fromkeys(['mean', 'std', 'median', 'q1', 'q3'])
    mean_vals[var]['mean'] = np.mean(samps, axis=0)
    mean_vals[var]['std'] = np.std(samps, axis=0)
    mean_vals[var]['median'] = np.median(samps, axis=0)
    mean_vals[var]['q1'] = np.quantile(samps, 0.16, axis=0)
    mean_vals[var]['q3'] = np.quantile(samps, 0.84, axis=0)
    if mean_vals[var]['mean'].ndim == 0:
        print("{0}: {1:.4f} +{2:.4f} -{3:.4f}".format(var, mean_vals[var]['median'], mean_vals[var]['q3'] - mean_vals[var]['median'], mean_vals[var]['median'] - mean_vals[var]['q1']))

fig, ax = plt.subplots()
az.plot_trace(trace_all, var_names=['veq', 'obl', 'inc', 'spot_contrast', 'spot_radius', 'spot_lon', 'spot_lat'], ax = ax)
fig.savefig("trace_plot.png")

az.summary(trace_all, var_names=['veq', 'obl', 'inc', 'spot_contrast', 'spot_radius', 'spot_lon', 'spot_lat'], ax = ax)

trace_all.to_netcdf("fit_all_trace.nc")

fig, ax = plt.subplots()
corner(trace_all, var_names=['veq', 'obl', 'inc', 'spot_contrast', 'spot_radius', 'spot_lon', 'spot_lat'], ax = ax)
fig.savefig("corner_plot.png")


# fig, ax = plt.subplots()
# im = ax.imshow(flux - flux_quiet, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"residuals_from_quiet_model_spectral_time_series_obl{starmap_all.obl}.png")
# plt.close(fig)

# fig, ax = plt.subplots()
# im = ax.imshow(flux_quiet - mean_out_quiet, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"planet_only_spectral_time_series_obl{starmap_all.obl}.png")
# plt.close(fig)

# for key in map_soln.keys():
#     print(f"{key}: {map_soln[key]}")


# with pm.Model(**theano_config) as fit_intr_priors:
# # print(starmap.velocity_unit)
# # model_ccfs = np.zeros((len(time_transit_mask), len(vels)))

# # temps = compute_temps_for_ccf(wav_binned, binned_temp, vels)
# # print(temps.shape)

# # print(f"Recursion limit: {sys.getrecursionlimit()}")
# # sys.setrecursionlimit(50000)  # Default is 1000, increase to 10000
# # # print(f"Recursion limit: {sys.getrecursionlimit()}")

# # vsini_fit = st['vsini']
# # obl_fit = np.pi/2.
# # inc_fit = 90.

# # amp = pm.Normal("amp", mu = 1.0, sigma = 0.1)
# # y = pm.Normal("y", mu = 0.0, sigma=1E-3, shape = ((ndeg + 1) ** 2) - 1)

# # SHT matrix: converts from pixels to Ylms
# # A = starmap.sht_matrix(smoothing=0.075)
# # npix = A.shape[1]

# # # Prior on the map: intensity uniform in [0, 1]
# # p = pm.Uniform("pixels", lower=0.0, upper=1.0, shape=(npix,))
# # amp = pm.Uniform("amp", lower=0.0, upper=1.0)
# # starmap[:, :] = amp * tt.dot(A, p)

#     intr_cont = pm.Uniform("intr_cont", lower = 0.9, upper = 1.1)
#     intr_ctrst = pm.Uniform("intr_ctrst", lower = 0.4, upper = 0.8)
#     intr_sigma = pm.Uniform("intr_sigma", lower = 0.05, upper = 0.25)
#     intr_wav = pm.Uniform("intr_wav", lower = 9999.5, upper = 10000.)

# # intr_cont = 1.0
# # intr_ctrst = 0.5
# # intr_sigma = 0.15
# # intr_wav = 9999.8

# # intr_cont = np.median(np.append(spec_map[0][:10], spec_map[0][-10:]))
# # intr_ctrst = 0.8
# # intr_sigma = 0.1
# # intr_wav = 9999.5
#     wav_grid_pad = starmap_all.wav0
#     # wav_grid_pad_t = tt.as_tensor_variable(wav_grid_pad)
#     pm.Deterministic("wav_grid_pad", wav_grid_pad)

#     intr_spec = gaussian_profile_tt(wav_grid_pad, intr_ctrst, intr_wav, intr_sigma, intr_cont)
#     pm.Deterministic("intr_spec", intr_spec)
#     starmap_all.load(spectrum = intr_spec)
#     starmap_quiet.load(spectrum = intr_spec)
# # starmap_out.load(spectrum = intr_spec)

# # spot1_contrast = pm.Uniform("spot_contrast", lower = 0.01, upper = 0.99)
# # spot1_radius = pm.Uniform("spot_radius", lower = 5., upper = 60.)  # degrees
# # spot1_lon = pm.Uniform("spot_lon", lower = -80, upper = 0.)  # degrees
# # spot1_lat = pm.Uniform("spot_lat", lower = -60., upper = 60.)  # degrees

# # spot1_contrast = 0.5
# # spot1_radius = 30.  # degrees   
# # spot1_lon = -40.  # degrees
# # spot1_lat = 0.  # degrees

# # starmap_shm.spot(contrast = spot1_contrast, radius = spot1_radius, lon = spot1_lon, lat = spot1_lat)

# # spot2_contrast = pm.Uniform("spot2_contrast", lower = 0.01, upper = 0.99)
# # spot2_radius = pm.Uniform("spot2_radius", lower = 5., upper = 60.)  # degrees
# # spot2_lon = pm.Uniform("spot2_lon", lower = -10, upper = 80.)  # degrees
# # spot2_lat = pm.Uniform("spot2_lat", lower = -90., upper = 90.)  # degrees

# # starmap_shm.spot(contrast = spot2_contrast, radius = spot2_radius, lon = spot2_lon, lat = spot2_lat)
# # amp = starmap_shm.y[0]
# # y = starmap_shm.y[1:]

# # starmap_all.amp = amp
# # starmap_all[1:, :] = y
# # y =pm.Deterministic("y", starmap_shm.y)
# # starmap_all.amp = amp
# # starmap_all[1:, :] = y
# # starmap_out.amp = amp
# # starmap_out[1:, :] = y

# # lat = np.linspace(-90, 90, 300)
# # lon = np.linspace(-180, 180, 600)
# # image = np.ones((len(lat), len(lon)))
# # y = lat.reshape(-1, 1)
# # x = lon.reshape(1, -1)
# # image[(x - spot_lon) ** 2 + (y - spot_lat) ** 2 < spot_radius ** 2] = (1 - spot_contrast)
# # starmap.load(image = image, spectrum = intr_spec)


# # alpha_fit = pm.TruncatedNormal("alpha", mu = 0.0, sigma = 0.1, lower = 0., upper = 0.5)
#     # LD1q = pm.TruncatedNormal("LD1q", mu = st['LDq'][0], sigma = st['LDq_unc'][0], lower = 0.0, upper = 1.0)
#     # LD2q = pm.TruncatedNormal("LD2q", mu = st['LDq'][1], sigma = st['LDq_unc'][1], lower = 0.0, upper = 1.0)

#     # LD1_fit = pm.Deterministic("LD1", 2 * LD1q ** 0.5 * LD2q) 
#     # LD2_fit = pm.Deterministic("LD2", LD1q ** 0.5 * (1. - 2 * LD2q))
#     # LD1q = pm.Uniform("LD1q", lower = 0.0, upper = 1.0)
#     # LD2q = pm.Uniform("LD2q", lower = 0.0, upper = 1.0)

#     # LD1_fit = pm.Deterministic("LD1", 2 * LD1q ** 0.5 * LD2q)
#     # LD2_fit = pm.Deterministic("LD2", LD1q ** 0.5 * (1. - 2 * LD2q))

#     # veq_fit = pm.Normal("veq", mu = 24000., sigma = 2000.)
#     veq_fit = 24.
#     starmap_all.veq = veq_fit 

# # prot_fit = pm.Normal("prot", mu = st["prot"], sigma = st["prot_unc"])
# # r_star_fit = pm.Normal("r_star", mu = st["R"], sigma = st["R_unc"])
# # m_star_fit = pm.Normal("m_star", mu = st["M"], sigma = st["M_unc"])
#     prot_fit = st["prot"]
#     r_star_fit = st["R"]
#     m_star_fit = st["M"]

#     # starmap_all[1] = LD1_fit
#     # starmap_all[2] = LD2_fit

# # t0_fit = pm.Normal("t0", mu = pl["t0"], sigma = )
# # starmap.alpha = alpha_fit
# # starmap_all.veq = veq_fit * 1000.
# # starmap_quiet.veq = veq_fit * 1000.
# # # starmap_all.obl = obl_fit
# # starmap_all.inc = inc_fit
# # starmap_quiet.inc = inc_fit
# # starmap_out.veq = veq_fit * 1000.
# # starmap_out.obl = obl_fit
# # starmap_out.inc = inc_fit
# # starmap[1] = LD1_fit
# # starmap[2] = LD2_fit

#     star = starry.Primary(
#         starmap_all, 
#         prot = prot_fit,
#         r = r_star_fit,
#         m = m_star_fit,
#         t0 = 0.,
#         theta0 = 0.,
#         length_unit = u.Rsun,
#         mass_unit = u.Msun,
#         time_unit = u.day
#     )

#     # star_quiet = starry.Primary(
#     #     starmap_quiet,
#     #     prot = prot_fit,
#     #     r = r_star_fit, 
#     #     m = m_star_fit, 
#     #     t0 = 0., 
#     #     theta0 = 0., 
#     #     length_unit = u.Rsun, 
#     #     mass_unit = u.Msun, 
#     #     time_unit = u.day 
#     # )

# # planet_inc = pm.Normal("planet_inc", mu = pl['inc'], sd = pl['inc_unc'])
# # planet_m = pm.Uniform("planet_m", lower = 5., upper = 40.)
# # planet_r = pm.Normal("planet_r", mu = pl["R"], sigma = pl["R_unc"])
#     planet_inc = pl['inc']
#     # planet_m = 10.
#     planet_r = pl["R"]
#     # porb_fit = pm.TruncatedNormal("porb", mu = pl["porb"], sigma = 0.00003, lower = pl["porb"] - 0.0001, upper = pl["porb"] + 0.0001)

#     planet = starry.Secondary(
#         planetmap,
#         r = pl['ror'] * r_star_fit,
#         porb = pl['porb'],
#         inc = planet_inc,
#         m = 0.,
#         t0 = 0.,
#         ecc = pl['ecc'],
#         w = 23.64034616,
#         length_unit = u.Rsun,
#         mass_unit = u.Msun,
#         time_unit = u.day
#     )
# # print(planet.angle_unit)
# # print(planet.time_unit)
# # print(times_all[0])

# # flux = starmap_out.flux(theta = out_times/st['prot'] * 180./np.pi)
#     system = starry.System(star, planet)
#     # system_quiet = starry.System(star_quiet, planet)
#     flux = pm.Deterministic("flux", system.flux(t = times_all, total=True))
#     # flux = pm.Deterministic("flux", starmap_all.flux(theta = times_all/st['prot'] * 180./np.pi))
#     # flux_quiet = system_quiet.flux(t = times_all, total=True)
#     # flux = starmap_all.flux(theta = times_all/st['prot'] * 180./np.pi)
#     # pos = system.position(t = times_all)
#     obs = pm.Normal("obs_flux", mu=flux, sd=spec_map_all_err, observed=spec_map_all)

# with fit_intr_priors:
#     map_soln = pmx.optimize()

# fig, ax = plt.subplots()
# ax.plot(map_soln['wav_grid_pad'], map_soln['intr_spec'], label='Intrinsic Line Profile', color='C2')
# ax.plot(wav_grid, spec_map_all[0,:], label='Observed', color='C0')
# ax.plot(wav_grid, map_soln['flux'], label='Model', color='C1')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Normalized Flux")
# ax.set_xlim(wav_grid[0], wav_grid[-1])
# ax.legend()
# fig.savefig("intrinsic_line_profile_fit.png")

# for key in map_soln.keys():
#     print(f"{key}: {map_soln[key]}")

# with fit_intr_priors:
#     trace_intr_priors = pm.sample(1000, tune=1000, cores=1, chains = 4, return_inferencedata=True)

# mean_vals = {}
# for var in [
#     # 'LD1q', 'LD2q', 
#     'veq', 'intr_cont', 'intr_ctrst', 'intr_sigma', 'intr_wav']:
#     samps = trace_intr_priors.get_values(varname=var, combine=True)
#     mean_vals[var] = dict.fromkeys(['mean', 'std', 'median', 'q1', 'q3'])
#     mean_vals[var]['mean'] = np.mean(samps, axis=0)
#     mean_vals[var]['std'] = np.std(samps, axis=0)
#     mean_vals[var]['median'] = np.median(samps, axis=0)
#     mean_vals[var]['q1'] = np.quantile(samps, 0.16, axis=0)
#     mean_vals[var]['q3'] = np.quantile(samps, 0.84, axis=0)
#     if mean_vals[var]['mean'].ndim == 0:
#         print("{0}: {1:.4f} +{2:.4f} -{3:.4f}".format(var, mean_vals[var]['median'], mean_vals[var]['q3'] - mean_vals[var]['median'], mean_vals[var]['median'] - mean_vals[var]['q1']))


# np.save('/Users/andrew/Desktop/vsini_output.npy', kT_val)


# print("pos shape:", pos.shape)

# fig, axs = plt.subplots(1,3, figsize=(14,8), sharex=True)

# ax = axs[0]
# ax.plot(times_all, pos[0][0], label = 'Star')
# ax.plot(times_all, pos[0][1], label = 'Planet')
# ax.axvline(0., color='k', ls='--')
# ax.axvline(times_all[11], color='k', ls='--')
# ax.axvline(times_all[26], color='k', ls='--')
# ax.set_ylabel('X Position [R_star]')
# ax.legend()
# ax = axs[1]
# ax.plot(times_all, pos[1][0], label = 'Star')
# ax.plot(times_all, pos[1][1], label = 'Planet')
# ax.axvline(0., color='k', ls='--')
# ax.axvline(times_all[11], color='k', ls='--')
# ax.axvline(times_all[26], color='k', ls='--')
# ax.set_ylabel('Y Position [R_star]')
# ax = axs[2]
# ax.plot(times_all, pos[2][0], label = 'Star')
# ax.plot(times_all, pos[2][1], label = 'Planet')
# ax.axvline(0., color='k', ls='--')
# ax.axvline(times_all[11], color='k', ls='--')
# ax.axvline(times_all[26], color='k', ls='--')
# ax.set_ylabel('Z Position [R_star]')
# ax.set_xlabel('Time [days]')
# fig.savefig("model_positions.png")

# xp = pos[0][1]
# yp = pos[1][1]
# zp = pos[2][1]

# pos_on_disk = np.sqrt(xp**2 + yp**2)

# mean_out = np.zeros_like(wav_grid)
# mean_out_quiet = np.zeros_like(wav_grid)
# j = 0
# for i in range(len(times_transit)):
#     if i < 12 or i > 27:
#         mean_out += flux[i,:]
#         mean_out_quiet += flux_quiet[i,:]
#         j += 1
# mean_out /= j
# mean_out_quiet /= j

# fig, ax = plt.subplots()
# im = ax.imshow(flux, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# for i in range(len(times_all)):
#     plt.text(wav_grid[0] + 0.1, times_all[i], f"{pos_on_disk[i]:.2f}", va='center', ha='left', color='black', fontsize=6)
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"model_spectral_time_series_obl{starmap_all.obl}.png")

# fig, ax = plt.subplots()
# starmap_all.show(ax=ax, projection='moll', grid=False, colorbar=True)
# fig.savefig("model_map.png")
# plt.close(fig)

# fig, ax = plt.subplots()
# im = ax.imshow(flux - mean_out, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"residual_model_spectral_time_series_obl{starmap_all.obl}.png")

# fig, ax = plt.subplots()
# im = ax.imshow(flux - flux_quiet, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"residuals_from_quiet_model_spectral_time_series_obl{starmap_all.obl}.png")
# plt.close(fig)

# fig, ax = plt.subplots()
# im = ax.imshow(flux_quiet - mean_out_quiet, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_all[0], times_all[-1]], origin='lower', cmap='viridis')
# ax.axhline(0., color='k', ls='--')
# ax.axhline(times_all[12], color='k', ls='--')
# ax.axhline(times_all[28], color='k', ls='--')
# ax.set_xlabel("Wavelength")
# ax.set_ylabel("Time (days)")
# plt.colorbar(im, ax = ax, label='Flux')
# fig.savefig(f"planet_only_spectral_time_series_obl{starmap_all.obl}.png")
# plt.close(fig)

# for i in range(len(times_all)):
#     # print(f"Time = {times_all[i]:.4f} days")
#     # print(flux[i])
#     fig, ax = plt.subplots()
#     ax.plot(wav_grid, spec_map_all[i,:], label='Observed', color='C0')
#     ax.plot(wav_grid, flux[i,:], label='Model', color='C1')
#     axr = ax.twinx()
#     axr.plot(wav_grid_pad, intr_spec, 'o', label='Intrinsic Line Profile', color='C2')
#     ax.fill_between(wav_grid, spec_map_all[i,:] - spec_map_all_err[i,:], spec_map_all[i,:] + spec_map_all_err[i,:], color='C0', alpha=0.3)
#     ax.axvline(intr_wav - 1, color='k', ls='--', label='Intrinsic Line Center')
#     ax.axvline(intr_wav + 1, color='k', ls='--')
#     ax.set_xlabel("Wavelength")
#     ax.set_ylabel("Normalized Flux")
#     ax.set_xlim(wav_grid[0], wav_grid[-1])
#     ax.set_title(f"Time = {times_all[i]:.4f} days")
#     ax.legend()
#     fig.savefig(f"model_fit_time_{i:02d}.png")
#     plt.close(fig)

# print(len(intr_spec))
# print(len(wav_grid_pad[(wav_grid_pad < intr_wav + 1) & (wav_grid_pad > intr_wav - 1)]))

# fig, ax = plt.subplots()

# plt.plot(times_all, xp, label="x")
# plt.plot(times_all, yp, label="y")
# plt.plot(times_all, zp, label="z")
# plt.ylabel("position [R$_*$]")
# plt.xlabel("time [days]")
# plt.legend()

# fig, ax = plt.subplots()
# for i in range(len(times_all)):
#     ax.plot(xp[i], yp[i], 'ob')
#     ax.text(xp[i], yp[i], f"{i}", fontsize=8, ha='right', va='bottom')
# # ax.scatter(0, 0, marker="*", color="k", s=100, zorder=10)
# circle = mpatches.Circle((0., 0.), radius=1., 
#                           fill=False, edgecolor='black', linewidth=1.5)
# ax.add_patch(circle)

# # Important: set equal aspect ratio so the circle isn't distorted
# ax.set_aspect('equal')

# ax.set_xlabel(r"x [R$_*$]")
# ax.set_ylabel(r"y [R$_*$]")
# fig.savefig("model_positions_2.png")
# plt.close(fig)

# star_show = starry.Primary(
#     starmap_shm, 
#     prot = prot_fit,
#     r = r_star_fit,
#     m = m_star_fit,
#     t0 = 0.,
#     theta0 = 0.,
#     length_unit = u.Rsun,
#     mass_unit = u.Msun,
#     time_unit = u.day
# )

# system_show = starry.System(star_show, planet)
# system_show.show(ax=ax, t = times_all)
# fig.savefig("model_show.gif")
# plt.close(fig)

# pm.Deterministic("flux", flux)
# obs = pm.Normal("obs_flux", mu=flux, sd=spec_map_all_err, observed=spec_map_all)

# pm.Deterministic("flux_full", starmap_all.flux(theta = times_all/st['prot'] * 180./np.pi))


# with pm.Model(**theano_config) as model:
#     # vsini_fit = st['vsini']
#     # obl_fit = 0.
#     # inc_fit = 90.

#     # amp = pm.Normal("amp", mu = 1.0, sigma = 0.1)
#     # y = pm.Normal("y", mu = 0.0, sigma=1E-3, shape = ((ndeg + 1) ** 2) - 1)

#     # SHT matrix: converts from pixels to Ylms
#     # A = starmap.sht_matrix(smoothing=0.075)
#     # npix = A.shape[1]

#     # # Prior on the map: intensity uniform in [0, 1]
#     # p = pm.Uniform("pixels", lower=0.0, upper=1.0, shape=(npix,))
#     # amp = pm.Uniform("amp", lower=0.0, upper=1.0)
#     # starmap[:, :] = amp * tt.dot(A, p)

#     intr_cont = pm.Uniform("intr_cont", lower = 0.9, upper = 1.1)
#     intr_ctrst = pm.Uniform("intr_ctrst", lower = 0.4, upper = 0.8)
#     intr_sigma = pm.Uniform("intr_sigma", lower = 0.05, upper = 0.25)
#     intr_wav = pm.Uniform("intr_wav", lower = 9999.5, upper = 10000.)

#     # intr_cont = np.median(np.append(spec_map[0][:10], spec_map[0][-10:]))
#     # intr_ctrst = 0.8
#     # intr_sigma = 0.1
#     # intr_wav = 9999.5
#     wav_grid_pad = starmap_all.wav0
#     pm.Deterministic("wav_grid_pad", wav_grid_pad)

#     intr_spec = gaussian_profile(wav_grid_pad, intr_ctrst, intr_wav, intr_sigma, intr_cont)
#     pm.Deterministic("intr_spec", intr_spec)
#     starmap_all.load(spectrum = intr_spec)
#     # starmap_out.load(spectrum = intr_spec)

#     spot1_contrast = pm.Uniform("spot_contrast", lower = 0.01, upper = 0.99)
#     spot1_radius = pm.Uniform("spot_radius", lower = 5., upper = 60.)  # degrees
#     spot1_lon = pm.Uniform("spot_lon", lower = -80, upper = 0.)  # degrees
#     spot1_lat = pm.Uniform("spot_lat", lower = -60., upper = 60.)  # degrees

#     starmap_shm.spot(contrast = spot1_contrast, radius = spot1_radius, lon = spot1_lon, lat = spot1_lat)

#     # spot2_contrast = pm.Uniform("spot2_contrast", lower = 0.01, upper = 0.99)
#     # spot2_radius = pm.Uniform("spot2_radius", lower = 5., upper = 60.)  # degrees
#     # spot2_lon = pm.Uniform("spot2_lon", lower = -10, upper = 80.)  # degrees
#     # spot2_lat = pm.Uniform("spot2_lat", lower = -90., upper = 90.)  # degrees

#     # starmap_shm.spot(contrast = spot2_contrast, radius = spot2_radius, lon = spot2_lon, lat = spot2_lat)

#     # y =pm.Deterministic("y", starmap_shm.y)
#     starmap_all.amp = amp
#     starmap_all[1:, :] = y
#     # starmap_out.amp = amp
#     # starmap_out[1:, :] = y

#     # lat = np.linspace(-90, 90, 300)
#     # lon = np.linspace(-180, 180, 600)
#     # image = np.ones((len(lat), len(lon)))
#     # y = lat.reshape(-1, 1)
#     # x = lon.reshape(1, -1)
#     # image[(x - spot_lon) ** 2 + (y - spot_lat) ** 2 < spot_radius ** 2] = (1 - spot_contrast)
#     # starmap.load(image = image, spectrum = intr_spec)


#     # alpha_fit = pm.TruncatedNormal("alpha", mu = 0.0, sigma = 0.1, lower = 0., upper = 0.5)
#     # LD1q = pm.TruncatedNormal("LD1q", mu = st['LDq'][0], sigma = st['LDq_unc'][0], lower = 0.0, upper = 1.0)
#     # LD2q = pm.TruncatedNormal("LD2q", mu = st['LDq'][1], sigma = st['LDq_unc'][1], lower = 0.0, upper = 1.0)

#     # LD1_fit = pm.Deterministic("LD1", 2 * LD1q ** 0.5 * LD2q) 
#     # LD2_fit = pm.Deterministic("LD2", LD1q ** 0.5 * (1. - 2 * LD2q))

#     veq_fit = vsini_fit / np.sin(inc_fit * np.pi/180.)

#     # prot_fit = pm.Normal("prot", mu = st["prot"], sigma = st["prot_unc"])
#     # r_star_fit = pm.Normal("r_star", mu = st["R"], sigma = st["R_unc"])
#     # m_star_fit = pm.Normal("m_star", mu = st["M"], sigma = st["M_unc"])
#     prot_fit = st["prot"]
#     r_star_fit = st["R"]
#     m_star_fit = st["M"]

#     # t0_fit = pm.Normal("t0", mu = pl["t0"], sigma = )
#     # starmap.alpha = alpha_fit
#     starmap_all.veq = veq_fit * 1000.
#     starmap_all.obl = obl_fit
#     starmap_all.inc = inc_fit
#     # starmap_out.veq = veq_fit * 1000.
#     # starmap_out.obl = obl_fit
#     # starmap_out.inc = inc_fit
#     # starmap[1] = LD1_fit
#     # starmap[2] = LD2_fit

#     star = starry.Primary(
#         starmap_all, 
#         prot = prot_fit,
#         r = r_star_fit,
#         m = m_star_fit,
#         t0 = 0.,
#         theta0 = 0.,
#         length_unit = u.Rsun,
#         mass_unit = u.Msun,
#         time_unit = u.day
#     )

#     # planet_inc = pm.Normal("planet_inc", mu = pl['inc'], sd = pl['inc_unc'])
#     # planet_m = pm.Uniform("planet_m", lower = 5., upper = 40.)
#     # planet_r = pm.Normal("planet_r", mu = pl["R"], sigma = pl["R_unc"])
#     planet_inc = pl['inc']
#     planet_m = 10.
#     planet_r = pl["R"]
#     # porb_fit = pm.TruncatedNormal("porb", mu = pl["porb"], sigma = 0.00003, lower = pl["porb"] - 0.0001, upper = pl["porb"] + 0.0001)

#     planet = starry.Secondary(
#         planetmap,
#         r = planet_r,
#         porb = pl['porb'],
#         inc = planet_inc,
#         m = planet_m,
#         t0 = 0.,
#         length_unit = u.Rearth,
#         mass_unit = u.Mearth,
#         time_unit = u.day
#     )

#     # flux = starmap_out.flux(theta = out_times/st['prot'] * 180./np.pi)
#     system = starry.System(star, planet)
#     flux = system.flux(t = times_all, total=True)

#     pm.Deterministic("flux", flux)
#     obs = pm.Normal("obs_flux", mu=flux, sd=spec_map_all_err, observed=spec_map_all)

#     # pm.Deterministic("flux_full", starmap_all.flux(theta = times_all/st['prot'] * 180./np.pi))

# with model:
#     map_soln = pmx.optimize()

# for key in map_soln.keys():
#     print(f"{key}: {map_soln[key]}")

# fig, ax = plt.subplots()
# im = ax.imshow(spec_map_all, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[-1], times_transit[0]], cmap='viridis', origin='lower')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Data')
# fig.savefig("2D_flux_variation_data.png")


# fig, ax = plt.subplots()
# im = ax.imshow(map_soln['flux'], aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[0], times_transit[-1]], cmap='viridis', origin='lower')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Model')
# fig.savefig("2D_flux_variation_model.png")

# mean_model_line = np.zeros_like(wav_grid)
# for i in range(len(data['out_frames'])):
#     mean_model_line += map_soln['flux'][i,:]
# mean_model_line /= (len(data['out_frames']))

# model_residuals_from_mean = map_soln['flux'] - mean_model_line

# fig, ax = plt.subplots()
# im = ax.imshow(model_residuals_from_mean, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[0], times_transit[-1]], cmap='viridis', origin='lower')   
# plt.colorbar(im, ax=ax, label='Flux Residuals from Mean Line')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# fig.savefig("2D_flux_variation_model_residuals.png")

# fig, ax = plt.subplots()
# im = ax.imshow(observed_resids_grid, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[0], times_transit[-1]], cmap='viridis', origin='lower')
# plt.colorbar(im, ax=ax, label='Data Residuals from Mean Line')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# fig.savefig("2D_flux_variation_data_residuals.png")

# fig, ax = plt.subplots()
# im = ax.imshow(spec_map_all - map_soln['flux_full'], aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[0], times_transit[-1]], cmap='viridis', origin='lower' )
# plt.colorbar(im, ax=ax, label='Data - Model')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# fig.savefig("Data_minus_model_2D_map.png")

# fig, ax = plt.subplots()
# ax.plot(map_soln['wav_grid_pad'], map_soln['intr_spec'], 'k-', alpha = 0.5)
# for i in range(int(len(times_all))):
#     ax.plot(wav_grid, map_soln['flux_full'][i,:], color='C0', alpha=0.5)
#     ax.plot(wav_grid, spec_map[i,:], 'r', alpha=0.5)
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Flux')
# ax.set_title('All Line profiles from Model')
# fig.savefig("all_line_profiles_model.png")

# for i in range(int(len(times_all))):
#     fig, axs = plt.subplots(2, 1, sharex=True)
#     ax = axs[0]
#     ax.plot(wav_grid, map_soln['flux_full'][i,:], color='C0', alpha=0.5)
#     ax.plot(wav_grid, spec_map[i,:], 'r', alpha=0.5)
#     ax.set_ylabel('Flux')
#     ax.set_xlim(9998.6, 10001)

#     ax = axs[1]
#     ax.errorbar(wav_grid, (spec_map_all[i] - map_soln['flux_full'][i])/spec_map_all_err[i], yerr = spec_map_all_err[i], fmt = 'ob')
#     ax.plot(wav_grid, np.zeros_like(wav_grid), 'r-')
#     ax.set_xlabel('Wavelength (nm)')
#     ax.set_ylabel('Flux Residuals')
#     ax.set_title(f'Line Profile Residuals - Frame {i}')
#     ax.set_xlim(9998.6, 10001)
#     plt.tight_layout()
#     fig.savefig(f"line_profile_residuals_{i}.png")
#     plt.close(fig)

# starmap_inf = starry.Map(ydeg=ndeg, udeg = 2)
# # fig, ax = plt.subplots()
# starmap_inf.amp = map_soln['amp']
# starmap_inf[1:, :] = map_soln['y']
# starmap_inf.show(theta = times_transit/st['prot'] * 180./np.pi, grid=True, file = "stellar_map.gif", colorbar=True)
# # fig.savefig("stellar_map.png")


# #system
#     sys = starry.System(star, planet)

#     flux_model = sys.flux(t = times_transit, total=True)
#     pm.Deterministic("flux_model", flux_model)

#     obs = pm.Normal("obs_flux", mu=flux_model, sd=spec_map_err, observed=spec_map)



# with model:
#     map_soln = pmx.optimize()

# for key in map_soln.keys():
#     print(f"{key}: {map_soln[key]}")

# print(np.isnan(np.sum(map_soln['flux_model'])))

# for i in range(len(times_transit)):
#     print(f"time: {times_transit[i]} : {map_soln['flux_model'][i]}")

# fig, ax = plt.subplots()
# ax.plot(wav_grid, map_soln['flux_model'][0], 'r-', label='model 0')
# ax.plot(wav_grid, map_soln['flux_model'][1], 'b-', label='model 1')
# ax.plot(wav_grid, spec_map[0], 'ro', label='data 0', markersize=2)
# ax.plot(wav_grid, spec_map[1], 'bo', label='data 1', markersize=2)
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Flux')
# ax.legend()
# fig.savefig("flux_comparison.png")

# fig, ax = plt.subplots()
# im = ax.imshow(map_soln['flux_model'], aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[-1], times_transit[0]], cmap='viridis')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Model')
# fig.savefig("2D_flux_variation_model.png")

# fig, ax = plt.subplots()
# im = ax.imshow(spec_map, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[-1], times_transit[0]], cmap='viridis')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Data')
# fig.savefig("2D_flux_variation_data.png")


# with pm.Model(**theano_config) as model:
#     # vsini_fit = pm.Normal("vsini", mu=st['vsini'], sigma=st['vsini_unc'])
#     vsini_fit = st['vsini']
#     obl_fit = 0.
#     inc_fit = 90.
#     # obl_fit = pm.Uniform("obl", lower = -180, upper = 180.)
#     # inc_fit = pm.Normal("inc", mu = 90., sigma = 7.)

#     # alpha_fit = pm.TruncatedNormal("alpha", mu = 0.0, sigma = 0.1, lower = 0., upper = 0.5)
#     # LD1q = pm.TruncatedNormal("LD1q", mu = st['LDq'][0], sigma = st['LDq_unc'][0], lower = 0.0, upper = 1.0)
#     # LD2q = pm.TruncatedNormal("LD2q", mu = st['LDq'][1], sigma = st['LDq_unc'][1], lower = 0.0, upper = 1.0)

#     # LD1_fit = pm.Deterministic("LD1", 2 * LD1q ** 0.5 * LD2q) 
#     # LD2_fit = pm.Deterministic("LD2", LD1q ** 0.5 * (1. - 2 * LD2q))

#     veq_fit = pm.Deterministic("veq", vsini_fit / tt.sin(inc_fit * np.pi/180.))

#     # prot_fit = pm.Normal("prot", mu = st["prot"], sigma = st["prot_unc"])
#     # r_star_fit = pm.Normal("r_star", mu = st["R"], sigma = st["R_unc"])
#     # m_star_fit = pm.Normal("m_star", mu = st["M"], sigma = st["M_unc"])
#     prot_fit = st["prot"]
#     r_star_fit = st["R"]
#     m_star_fit = st["M"]

#     # t0_fit = pm.Normal("t0", mu = pl["t0"], sigma = )

#     intr_cont = pm.Uniform("intr_cont", lower = 14, upper = 18)
#     intr_ctrst = pm.Uniform("intr_ctrst", lower = 0.05, upper = 0.3)
#     intr_sigma = pm.Uniform("intr_sigma", lower = 0.01, upper = 0.5)
#     intr_wav = pm.Uniform("intr_wav", lower = 9999., upper = 10000.)

#     intr_spec = gaussian_profile_tt(wav_grid_pad, intr_ctrst, intr_wav, intr_sigma, intr_cont)

#     starmap_all.load(spectrum = intr_spec)

#     # starmap.alpha = alpha_fit
#     starmap_all.veq = veq_fit * 1000.
#     starmap_all.obl = obl_fit
#     starmap_all.inc = inc_fit
#     # starmap[1] = LD1_fit
#     # starmap[2] = LD2_fit

#     star = starry.Primary(
#         starmap_all, 
#         prot = prot_fit,
#         r = r_star_fit,
#         m = m_star_fit,
#         t0 = 0.,
#         theta0 = 0.,
#         length_unit = u.Rsun,
#         mass_unit = u.Msun,
#         time_unit = u.day
#     )

#     # planet_inc = pm.Normal("planet_inc", mu = pl['inc'], sd = pl['inc_unc'])
#     # planet_m = pm.Uniform("planet_m", lower = 5., upper = 40.)
#     # planet_r = pm.Normal("planet_r", mu = pl["R"], sigma = pl["R_unc"])
#     planet_inc = pl['inc']
#     planet_m = 10.
#     planet_r = pl["R"]
#     # porb_fit = pm.TruncatedNormal("porb", mu = pl["porb"], sigma = 0.00003, lower = pl["porb"] - 0.0001, upper = pl["porb"] + 0.0001)

#     planet = starry.Secondary(
#         planetmap,
#         r = planet_r,
#         porb = pl['porb'],
#         inc = planet_inc,
#         m = planet_m,
#         t0 = 0.,
#         length_unit = u.Rearth,
#         mass_unit = u.Mearth,
#         time_unit = u.day
#     )

#     #system
#     sys = starry.System(star, planet)

#     flux_model = sys.flux(t = times_transit, total=True)
#     pm.Deterministic("flux_model", flux_model)

#     obs = pm.Normal("obs_flux", mu=flux_model, sd=spec_map_err, observed=spec_map)


# with model:
#     map_soln = pmx.optimize()

# for key in map_soln.keys():
#     print(f"{key}: {map_soln[key]}")

# print(np.isnan(np.sum(map_soln['flux_model'])))

# for i in range(len(times_transit)):
#     print(f"time: {times_transit[i]} : {map_soln['flux_model'][i]}")

# fig, ax = plt.subplots()
# ax.plot(wav_grid, map_soln['flux_model'][0], 'r-', label='model 0')
# ax.plot(wav_grid, map_soln['flux_model'][1], 'b-', label='model 1')
# ax.plot(wav_grid, spec_map[0], 'ro', label='data 0', markersize=2)
# ax.plot(wav_grid, spec_map[1], 'bo', label='data 1', markersize=2)
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Flux')
# ax.legend()
# fig.savefig("flux_comparison.png")

# fig, ax = plt.subplots()
# im = ax.imshow(map_soln['flux_model'], aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[-1], times_transit[0]], cmap='viridis')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Model')
# fig.savefig("2D_flux_variation_model.png")

# fig, ax = plt.subplots()
# im = ax.imshow(spec_map, aspect='auto', extent=[wav_grid[0], wav_grid[-1], times_transit[-1], times_transit[0]], cmap='viridis')
# plt.colorbar(im, ax=ax, label='Flux')
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Time since mid-transit (days)')
# ax.set_title('2D Flux Variation Data')
# fig.savefig("2D_flux_variation_data.png")
