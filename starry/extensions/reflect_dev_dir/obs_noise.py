#### ELT Telescope and Noise Model python package to be with starry reflection

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from spectres import spectres

def wav_grid_from_R(R, wav_min, wav_max):
    N = int(np.log(wav_max/wav_min)/np.log(R/(R-1)))
    wav_max = wav_min * (R / (R - 1)) ** (N)
    wav = np.geomspace(wav_min, wav_max, int(N), endpoint = True)
    return wav

def resample_to_R(R, wav, spec, spec_err = None, wav_min = None, wav_max = None, stack = False):
    if wav_min is None:
        wav_min = wav[0]
    if wav_max is None:
        wav_max = wav[-1]

    wav_grid = wav_grid_from_R(R, wav_min, wav_max)

    fill = None
    if wav[0] > wav_grid[0] and wav[-1] < wav_grid[-1]:
        print("Given wavelength range is smaller than interpolated range on both ends. Wav grid will be truncated.")
        wav_grid = wav_grid_from_R(R, wav[0], wav[-1])
    if wav[0] == wav_grid[0]:
        fill = spec[0]
    if wav[-1] == wav_grid[-1]:
        fill = spec[-1]

    if spec_err is not None:
        spec_resamp, spec_err_resamp = spectres(wav_grid, wav, spec, spec_errs = spec_err, fill = fill)
        if stack:
            return np.column_stack((wav_grid, spec_resamp, spec_err_resamp))
        else:
            return wav_grid, spec_resamp
    else:
        spec_resamp = spectres(wav_grid, wav, spec, fill = fill)
        if stack:
            return np.column_stack((wav_grid, spec_resamp))
        else:
            return wav_grid, spec_resamp
        
class Telescope:
    def __init__(self, 
                 d_spec_aper = np.array([0.87, 0.71]), 
                 inst_eff = 0.10, 
                 emis_bg = 0.2, 
                 diam = 38.5, 
                 cobs = 0.28, 
                 det_pix_size = np.array([10, 15]), 
                 f_cam = 1.5, 
                 pix_bin_fac = 1, 
                 read_noise = np.array([1, 4.5]),           #e-/pix - different for obs < 0.95 um
                 dar_curr = np.array([1, 20]),              #e-/pix/hr - different for obs < 0.95 um
                 AO = 'No AO'             
                 ):

        self.tel_eff = np.array([[0.36, 0.4, 0.45, 0.55, 0.65, 0.8, 1.25, 1.65, 2.16], [.13, 0.28, 0.44, 0.58, 0.64, 0.68, 0.8, 0.83, 0.84]])     #ESO telescope efficiencies (Table 2 in ANDES ETC)
        self.d_spec_aper = d_spec_aper
        self.inst_eff = inst_eff
        self.emis_bg = emis_bg
        self.diam = diam
        self.cobs = cobs
        self.det_pix_size = det_pix_size
        self.f_cam = f_cam
        self.pix_bin_fac = pix_bin_fac
        self.read_noise = read_noise
        self.dar_curr = dar_curr
        if AO != 'No AO' and AO != 'LTAO' and AO != 'GLAO':
            print("Invalid AO system. Please choose from 'LTAO', 'GLAO', or 'No AO'. Defaulting to 'No AO'.")
            self.AO = 'No AO'
        else:
            self.AO = AO

    def telescope_area(self):
        # In cm^2 
        return np.pi * (self.diam / 2) ** 2 * (1 - self.cobs ** 2) * 10 ** 4

    def pix_theta(self):
        # Sky-projected pixel size in arcsec
        # Will be 2D for lambda <= 0.95 um and lambda > 0.95 um
        return 0.036 * (self.f_cam/1.5) ** (-1) * (self.det_pix_size/10) * (self.diam/38.5) ** (-1)

    def n_pix_spec_aper(self):
        # Number of pixels in the spectrometer aperture
        # Will be 2D for lambda <= 0.95 um and lambda > 0.95 um
        return np.pi / 4. * self.d_spec_aper ** 2 / self.pix_theta() ** 2
    
    def n_readout_pix_per_res(self):
        # Number of readout pixels per resolution element
        # Will be 2D for lambda <= 0.95 um and lambda > 0.95 um
        return int(self.n_pix_spec_aper() / self.pix_bin_fac)

    def telescope_eff(self, wav):
        return np.interp(wav, self.tel_eff[0], self.tel_eff[1])
    
    def slit_eff_interp(self, wav):
        if self.AO == 'No AO':
            eso_table_noAO = np.loadtxt('slit_eff_noAO.txt', skiprows=1)
            func = interp.RectBivariateSpline(x = eso_table_noAO[:,0], y = eso_table_noAO[1:,1], z = eso_table_noAO[:,2:])
        elif self.AO == 'LTAO':
            eso_table_LTAO = np.loadtxt('slit_eff_LTAO.txt', skiprows=1)
            func = interp.RectBivariateSpline(x = eso_table_LTAO[:,0], y = eso_table_LTAO[1:,1], z = eso_table_LTAO[:,2:])
        elif self.AO == 'GLAO':
            eso_table_GLAO = np.loadtxt('slit_eff_GLAO.txt', skiprows=1)
            func = interp.RectBivariateSpline(x = eso_table_GLAO[:,0], y = eso_table_GLAO[1:,1], z = eso_table_GLAO[:,2:])
        return func(self.d_spec_aper, wav)/100.

    def __str__(self):
        return f"Telescope with diameter {self.diameter} and focal length {self.focal_length}"
    
class Observation:
    def __init__(self, Telescope, wav, exp_time, rpow, ndit = 1, airmass = 1., T_bg = 283., mag = None, flux = None):
        self.telescope = Telescope
        self.mag = mag
        self.wav = wav
        self.exp_time = exp_time        #in seconds
        self.rpow = rpow
        self.sky_bg = np.array([[0.36, 0.44, 0.55, 0.64, 0.8, 1.05, 1.25, 1.65, 2.16], [22.50, 22.50, 21.8, 21.5, 20.5, 20.5, 20., 19.5, 19.5]])
        if exp_time > 30 * 60 * ndit:
            self.ndit = int(np.floor(exp_time / (30 * 60)) + 1)
            print(f"Exposure time exceeds 30 minutes. Number of dithered exposures will be {self.ndit}.")
        else:
            self.ndit = ndit
        self.airmass = airmass
        self.T_bg = T_bg
        self.alpha_atm_trans = np.array([[0.36, 0.44, 0.55, 0.70, 0.90, 1.0, 1.25, 1.65, 2.16, 2.60], [0.67, 0.83, 0.90,0.98, 0.99, 1., 1., 1., 1., 1.]])
        self.flux_model = flux           #In photons/cm^2/um PER HOUR
        if flux is not None and mag is not None:
            print("Neither flux nor magnitude provided. Observation model will not be able to calculate SNR.")

    def noise_det(self):
        #Detector noise over detector area corresponding to spectrometer aperture (e-)
        return np.sqrt(self.telescope.n_readout_pix_per_res() * (self.ndit * self.telescope.read_noise ** 2 / self.telescope.pix_bin_fac + self.telescope.dar_curr/3600 * self.exp_time))

    def sky_bg_w_airmass(self):
        #Sky background with airmass correction
        bg = np.interp(self.wav, self.sky_bg[0], self.sky_bg[1])
        return bg - 0.4 * (self.airmass - 1)
    
    def atmospheric_trans(self):
        alpha = np.interp(self.wav, self.alpha_atm_trans[0], self.alpha_atm_trans[1])
        return alpha ** self.airmass
    
    def total_eff(self):
        telescope_eff = self.telescope.telescope_eff(self.wav)
        return telescope_eff * self.telescope.inst_eff * self.atmospheric_trans()
    
    def bg_flux_in_spec_aper(self, print_breakdown = False, plot_breakdown = False):
        #Background flux in spectrometer aperture (e-/s)
        sigma_sky = 10 ** ((16.85 - self.sky_bg_w_airmass()) / 2.5) / self.rpow                                                                 #Sky background in photons/cm^2/s/arcsec^2
        sigma_therm = (1.4 * 10 ** 12 * self.telescope.emis_bg * np.exp(-14388/(self.wav * self.T_bg))) / (self.rpow * self.wav ** 3)           #Thermal background in photons/cm^2/s/arcsec^2
        bg_factor = self.total_eff() * self.telescope.telescope_area() * np.pi / 4. * self.telescope.d_spec_aper ** 2     
        sigma_sky = sigma_sky * bg_factor
        sigma_therm = sigma_therm * bg_factor                      
        if print_breakdown:
            print(f"Sky background: {sigma_sky}")
            print(f"Thermal background: {sigma_therm}")
        if plot_breakdown:
            fig, ax = plt.subplots()
            ax.plot(self.wav, sigma_sky.flatten(), label = "Sky background")
            ax.plot(self.wav, sigma_therm.flatten(), label = "Thermal background")
            ax.legend()
            ax.set_ylabel("Background Flux (e-/s)")
            ax.set_xlabel("Wavelength (um)")
            plt.title("Background Flux Breakdown")
            plt.show()
        return sigma_sky + sigma_therm
    
    def bg_noise(self):
        #Background noise in spectrometer aperture and resolution element (e-)
        return np.sqrt(self.bg_flux_in_spec_aper() * self.exp_time)
    
    def noise_per_res_elem_from_nobj(self):
        #Noise per resolution element (e-)
        return np.sqrt(self.noise_det() ** 2 + self.bg_noise() ** 2 + self.nobj())
    
    def nobj_from_mag(self):
        #Number of object photons per resolution element from target in exposure time (e-)
        return self.telescope.slit_eff_interp(self.wav) * self.total_eff() * self.telescope.telescope_area() * self.exp_time / self.rpow * 10 ** ((16.85 - self.mag) / 2.5)
    
    def nobj(self):
        #Number of object photons per resolution element from target in exposure time (e-)
        num_hours = self.exp_time / 3600.
        N = num_hours * self.flux_model / 100 ** 2
        return self.telescope.slit_eff_interp(self.wav) * self.total_eff() * self.telescope.telescope_area() /self.rpow * N
    
    def snr(self, print_breakdown = False, plot_breakdown = False):
        #Signal-to-noise ratio per resolution element
        snr = np.sqrt(self.nobj() / self.noise_per_res_elem_from_nobj())
        return snr

    def snr_from_mag(self, print_breakdown = False, plot_breakdown = False):
        #Signal-to-noise ratio per resolution element from target in exposure time
        term = self.telescope.slit_eff_interp(self.wav) * self.total_eff() * self.telescope.telescope_area() * self.exp_time / self.rpow * 10 ** ((16.85 - self.mag) / 2.5)
        snr = np.sqrt(term ** 2 / (self.noise_det() ** 2 + self.bg_noise() ** 2 + term))
        if print_breakdown:
            print(f"SNR: {snr}")
            print(f"Object photon noise: {100 * term/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)}%")
            print(f"Detector noise: {100 * self.noise_det() ** 2/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)}%")
            print(f"Background noise: {100 * self.bg_noise() ** 2/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)}%")
        if plot_breakdown:
            fig, ax = plt.subplots()
            ax.plot(self.wav, snr.flatten(), label = "SNR per resolution element")
            ax2 = ax.twinx()
            ax2.plot(self.wav, (100 * term/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)).flatten(), label = "% Object photon noise")
            ax2.plot(self.wav, (100 * self.noise_det() ** 2/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)).flatten(), label = "% Detector noise")
            ax2.plot(self.wav, (100 * self.bg_noise() ** 2/(self.noise_det() ** 2 + self.bg_noise() ** 2 + term)).flatten(), label = "% Background noise")
            ax.legend()
            ax2.legend()
            ax.set_ylabel("SNR")
            ax2.set_ylabel("% Contribution to Noise")
            ax.set_xlabel("Wavelength (um)")
            plt.title("SNR and Noise Breakdown")
            plt.show()
        return snr

    def __str__(self):
        return f"Observation with telescope {self.telescope} and wavelength {self.wav}"