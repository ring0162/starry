import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import astropy.units as u
import starry
from scipy import integrate
from astropy.io import fits
import pickle



starry.config.lazy = False
starry.config.quiet = True

#Write txt file data into a numpy array, ignoring the first row
basalt = np.loadtxt('spectra/bayes_paper/albedo-rock-basalt-solid-ASTER.txt', skiprows=26)
granite = np.loadtxt('spectra/bayes_paper/albedo-rock-granite-solid-alkalic-ASTER.txt', skiprows=26)
sand = np.loadtxt('spectra/bayes_paper/albedo-soil-sand-brown_loamy_fine-ASTER.txt', skiprows=26)
grass = np.loadtxt('spectra/bayes_paper/albedo-vegetation-grass-lawn-ASTER.txt', skiprows=26)
trees = np.loadtxt('spectra/bayes_paper/albedo-vegetation-trees-deciduous-ASTER.txt', skiprows=26)
cloud = np.loadtxt('spectra/bayes_paper/albedo-water-cloud-MODIS.txt', skiprows=26)
coast = np.loadtxt('spectra/bayes_paper/albedo-water-coast-USGS+ASTER.txt', skiprows=26)
snow = np.loadtxt('spectra/bayes_paper/albedo-water-fine_snow-ASTER.txt', skiprows=26)
sea = np.loadtxt('spectra/bayes_paper/albedo-water-seawater-USGS+ASTER.txt', skiprows=26)

bacterial_mat = np.loadtxt('spectra/Bacterial_mat_spec.txt', skiprows=1)
bac_mat_wav = np.loadtxt('spectra/Bacterial_mat_wav.txt', skiprows=1)

bacterial_mat = np.column_stack((bac_mat_wav, bacterial_mat))

# Put all spectra onto same wavelength grid

for i in range(len(bacterial_mat[:,1])):
    if bacterial_mat[i, 1] < 0:
        bacterial_mat[i, 1] = sea[i, 1]

basalt[:, 1] = basalt[:, 1] / 100.
granite[:, 1] = granite[:, 1] / 100.
sand[:, 1] = sand[:, 1] / 100.
grass[:, 1] = grass[:, 1] / 100.
trees[:, 1] = trees[:, 1] / 100.
snow[:, 1] = snow[:, 1] / 100.

basalt = np.flip(basalt, axis=0)
granite = np.flip(granite, axis = 0)
sand = np.flip(sand, axis =0)

#Create a dictionary of the spectra
spectra = {'basalt': basalt, 'granite': granite, 'sand': sand, 'grass': grass, 'trees': trees, 'cloud': cloud, 'coast': coast, 'snow': snow, 'sea': sea, 'bacterial_mat': bacterial_mat}

for key in spectra:
    i = np.where(spectra[key][:,0] > 0.4)
    j = np.where(spectra[key][:,0] < 2.5)
    spectra[key] = spectra[key][np.min(i):np.max(j),:]

# Put all spectra onto same wavelength grid TODO: Come back to resolution.

wav = np.linspace(0.4, 2.5, 100)
print(wav)

for key in spectra:
    resamp = np.interp(wav, spectra[key][:,0], spectra[key][:,1])
    spectra[key] = np.column_stack((wav, resamp))
    plt.plot(spectra[key][:,0], spectra[key][:,1], label=key)

# Make Continental Spectra

cont =  (0.3 * spectra['grass'][:,1] + 
        0.3 * spectra['trees'][:,1] +
        0.09 * spectra['granite'][:,1] + 
        0.09 * spectra['basalt'][:,1] +
        0.07 * spectra['sand'][:,1] +
        0.15 * spectra['snow'][:,1])

cont_warm =  (0.3 * spectra['grass'][:,1] + 
        0.3 * spectra['trees'][:,1] +
        0.09 * spectra['granite'][:,1] + 
        0.09 * spectra['basalt'][:,1] +
        0.07 * spectra['sand'][:,1])/0.85

spectra['cont_warm'] = np.column_stack((wav, cont_warm))
spectra['cont'] = np.column_stack((wav, cont))

print("Spectra loaded and resampled.")

# Cloud Function
def add_clouds(map, spec, min_size = 30, max_size = 140, num = 6, plot = False, save_plot = False):

    cloud_pix = np.zeros((num, 3))

    for i in range(num):
        size = np.random.uniform(min_size, max_size)
        
        lon = np.random.uniform(0, 640)
        lat = np.random.uniform(0, 320)

        r_pix = int(size / 640. * 360)

        for x, y in np.ndindex(map.shape):
            if (x - lat)**2 + (y - lon)**2 < r_pix**2:
                map[x, y] = spec

        cloud_pix[i] = (lat, lon, r_pix)
    
    cum_coverage = (1. - np.count_nonzero(map - spec) / (320. * 640.)) * 100.

    if plot:
        print("{} percent cloud coverage".format(cum_coverage))
        plt.imshow(map)
        plt.colorbar()
        plt.show()
    
    if save_plot:
        plt.imshow(map)
        plt.colorbar()
        plt.title('Plot after cloud model with {} percent cover'.format(np.round(cum_coverage)))
        plt.savefig('cloud_cover.png')

    return map, cum_coverage

#### Apr 16: Adapting Working Method for Cont/Ocean/ice/clouds to include coast ###

earth_refl_spec = np.ones((len(wav), 320, 640))
ydeg = 20
specmap_y = np.zeros((len(wav), (ydeg+1)**2))
specmap_y_shift = np.zeros((len(wav), (ydeg+1)**2))
specmap_p = np.zeros((len(wav), 996))
minpix = np.zeros((len(wav)))
maxpix = np.zeros((len(wav)))
amps = np.zeros((len(wav)))

map = starry.Map(ydeg = ydeg, reflected=True)
scalarmap = starry.Map(ydeg = ydeg, reflected=True)

image0 = np.flipud(plt.imread('earth.png'))
image0[image0 < 0.5] = 0.
image0[image0 >= 0.5] = 1.

##Coast Method
coast_ij = np.zeros((image0.shape[0], image0.shape[1]))
for row in range(len(image0[:,0]) - 1):
    for col in range(len(image0[0]) - 1):
        if image0[row, col] != image0[row, col+1]:
            coast_ij[row, col] = 1.
        if image0[row, col] != image0[row + 1, col]:
            coast_ij[row, col] = 0.8


for row in range(len(coast_ij[:,0])):
    for col in range(len(coast_ij[0])):
        if coast_ij[row, col] == 1.:
            image0[row, col] = 0.5
            if col < len(coast_ij[0]) - 2:
                image0[row, col+1] = 0.5
                image0[row, col+2] = 0.5
            if col > 1:
                image0[row, col-1] = 0.5
                image0[row, col-2] = 0.5
        if coast_ij[row, col] == 0.8:
            image0[row, col] = 0.5
            if row < len(coast_ij[:,0]) - 2:
                image0[row+1, col] = 0.5
                image0[row+2, col] = 0.5
            if row > 1:
                image0[row-1, col] = 0.5
                image0[row-2, col] = 0.5
###End coast block

image0, cloud_percent = add_clouds(image0, 0.25, num = 15, save_plot=True)

for wl in range(len(wav)):
    
    map.reset()
    image = image0.copy()

    image[0:25] = spectra['snow'][wl, 1]
    image[293:320] = spectra['snow'][wl, 1]
    image[26:41][image[26:41] == 1.] = spectra['snow'][wl, 1]
    image[277:292][image[277:292] == 1.] = spectra['snow'][wl, 1]
    image[42:276][image[42:276] == 0.] = spectra['sea'][wl, 1]
    image[42:276][image[42:276] == 1.] = spectra['cont_warm'][wl, 1]
    image[42:276][image[42:276] == 0.5] = spectra['coast'][wl, 1]
    image[image == 0.5] = spectra['cloud'][wl, 1]
    
    # for i in range(len(cloud_pix)):
    #     lat = cloud_pix[i, 0]
    #     lon = cloud_pix[i, 1]
    #     r_pix = cloud_pix[i, 2]
    #     for x, y in np.ndindex(image.shape):
    #         if (x - lat)**2 + (y - lon)**2 < r_pix**2:
    #             image[x, y] = spectra['cloud'][wl, 1]

    earth_refl_spec[wl] = image

    map.load(image, smoothing = 0.1)
    specmap_y[wl] = map.y

    _, _, Y2P, P2Y, _, _ = map.get_pixel_transforms()
    p = Y2P.dot(map.y)

    #Pixel stretch to match max/min albedo to max/min pixel value
    maxi = np.max((spectra['cont_warm'][wl, 1], spectra['cloud'][wl, 1], spectra['snow'][wl, 1], spectra['sea'][wl, 1]))
    mini = np.min((spectra['cont_warm'][wl, 1], spectra['cloud'][wl, 1], spectra['snow'][wl, 1], spectra['sea'][wl, 1]))

    p = p / np.max(p) * (maxi - mini) + np.min(p)

    #Pixel shift to set min val 
    
    #shift = np.min(p) - 0.1         # shift the pixel values so that the minimum is 0.1
    shift = np.min(p) - mini      # shift the pixel values so that the minimum is ocean reflectance
    p = p - shift

    specmap_p[wl] = p

    y = P2Y.dot(p)

    specmap_y_shift[wl] = y

    minpix[wl] = np.min(p)
    maxpix[wl] = np.max(p)  
    
    #Times 0.5 added on 27/2 to try to reduce overall spherical albedo to more realistic values
    amps[wl] = y[0] * 0.5

    if wl == 20:
        scalarmap.load(image, smoothing = 0.1)

print("Spectra loaded into starry maps.")

def diagnostic_plots():
    fig, axs = plt.subplots(3, 1, squeeze=False)
    ax_twin = axs[0].twinx()
    axs[0].plot(wav, amps, color = 'red', label = "Average Albedo")
    ax_twin.plot(wav, spectra['cont_warm'][:,1], label = "Continental Albedo")
    ax_twin.plot(wav, spectra['sea'][:,1], label = "Ocean Albedo")
    axs[0].set_title('Average spherical albedo at each wavelength')
    axs[0].set_xlabel('Wavelength (micron)')
    axs[0].set_ylabel('Average Albedo')
    ax_twin.set_ylabel('Reflectance')
    axs[0].legend(bbox_to_anchor=(0., 1.), loc='upper left')
    ax_twin.legend()

    axs[1].plot(wav, maxpix, label = "Maximum pixel value at each wavelength")
    axs[1].plot(wav, minpix, label = "Minimum pixel value at each wavelength")
    axs[1].set_title('Minimum pixel value at each wavelength')
    axs[1].set_xlabel('Wavelength (micron)')
    axs[1].set_ylabel('Pixel value')
    axs[1].legend()

    max_albedo = np.zeros_like(wav)
    min_albedo = np.zeros_like(wav)

    for wl in range(len(wav)):
        max_albedo[wl] = np.max((spectra['cont_warm'][wl, 1], spectra['snow'][wl, 1], spectra['sea'][wl, 1], spectra['cloud'][wl, 1]))
        min_albedo [wl]= np.min((spectra['cont_warm'][wl, 1], spectra['snow'][wl, 1], spectra['sea'][wl, 1], spectra['cloud'][wl, 1]))

    axs[2].plot(wav, maxpix/minpix, label = "Pixel scale factor at each wavelength")
    axs[2].plot(wav, max_albedo/min_albedo, label = "Albedo scale factor at each wavelength")
    axs[2].set_title('Scale factor at each wavelength')
    axs[2].set_xlabel('Wavelength (micron)')
    axs[2].set_ylabel('Scale factor')
    axs[2].legend()

    fig.savefig('diagnostic_plots.png')

diagnostic_plots()
print("Diagnostic plots saved.")

#Create a map with the spectra

specmap = starry.Map(ydeg=ydeg, reflected=True, nw = len(wav))

for wl in range(len(wav)):
    specmap[1:, :, wl] = specmap_y_shift[wl][1:]/specmap_y_shift[wl][0]
    specmap.amp[wl] = specmap_y_shift[wl][0]

print("Spectral map created.")

####THE OBSERVATION FUNCTION###

def earth_refl_spec_obs(
        map,
        scalarmap, 
        num_obs = 30, 
        num_nights = 1,
        texp = 30.,             #Observational units assumed to be in minutes (texp, cadence)
        cadence = 60.,          
        phase_init = 90., 
        theta_init = 0., 
        snr = 30., 
        comp_time = 1.,         #Cadence of starry flux computations to be integrated    
        a = 150E6,              #Orbit/planet scales in units of km (a, R)
        Rp = 6400., 
        Rstar = 695508.,
        prot = 1., 
        porb = 365.25,          #Orbital and rotational periods in units of days (prot, porb)
        obl = 23.5,
        dist = 5.,             #Distance to system in parsecs
        show_plot = False,
        save_plot = True, 
        save_data = True, 
        filename = 'observation_sim'
        ):

    if cadence < texp:
        print('Warning! Cadence is less than exposure time. Correct if you want meaningful results.')
    
    ###########################################################
    # Compute the observation times and number of evaluations #
    ###########################################################    
    obs_per_night = np.zeros(num_nights)
    for night in range(num_nights):
        obs_per_night[night] = round(num_obs / num_nights)
        #print(obs_per_night[night])
        if obs_per_night[night] * cadence > (12 * 60.):
            print('Warning! Observation time per night is greater than 12 hours. Conisder breaking into multiple nights for ground based observations.')
         
    # obs_per_night[-1] = num_obs - np.sum(obs_per_night[:-1])
    # if obs_per_night[-1] * cadence > (12 * 60.):
    #     print('Warning! Observation time per night is greater than 12 hours. Conisder breaking into multiple nights for ground based observations.')

    obs_time = (num_nights - 1) * 24. * 60. + obs_per_night[-1] * cadence

    num_eval = int(obs_time/comp_time)

    t_obs = np.zeros((num_nights, int(obs_per_night[0])))
    for night in range(num_nights):
        t_obs[night] = np.arange(night * 24. * 60., night * 24. * 60. + obs_per_night[night] * cadence, cadence)
        #print(t_obs[night])
    
    # t_obs[-1] = np.arange((num_nights - 1) * 24. * 60., (num_nights - 1) * 24. * 60. + obs_per_night[-1] * cadence, cadence)

    #obs_per_night = round(num_obs / num_nights)
    #print("obs per night:", obs_per_night)
    #obs_time = (num_nights - 1) * 24. * 60. + (num_obs - (num_nights-1) * obs_per_night) * cadence
    #print("obs time:", obs_time)
    #print("24 hours = ", 24. * 60., " minutes.")
    #num_eval = int(obs_time / comp_time)

    # t_obs = np.arange(0, obs_per_night * cadence, cadence)
    # if num_nights > 2:
    #     for night in range(num_nights - 2):
    #             start = (night+1) * 24 * 60.
    #             t_obs = np.concatenate((t_obs,np.arange(start, start + obs_per_night * cadence, cadence)))
    # start = (num_nights-1) * 24 * 60.
    # if num_nights != 1:
    #     t_obs = np.concatenate((t_obs, np.arange(start, start + (num_obs - (num_nights-1) * obs_per_night) * cadence, cadence)))
  
    #print(t_obs)
    #print(len(t_obs))
   

    # if obs_per_night * cadence > (12 * 60.) or (num_obs - (num_nights-1) * obs_per_night) * cadence > (12 * 60.):
    #     print('Warning! Observation time per night is greater than 12 hours. Conisder breaking into multiple nights for ground based observations.')

    #########################################################
    # Compute planet's orbital position at each observation #
    #########################################################
    
    phase_final = phase_init + obs_time / (porb * 24 * 60.) * 360.
    obs_phase = np.linspace(phase_init, phase_final, num_eval)

    r = a/Rp # Planet - star distance in units of Earth radii
    #Phase = 0 is secondary eclipse, phase = 180 is transit
    x = r * np.sin(np.deg2rad(obs_phase))
    z = r * np.cos(np.deg2rad(obs_phase))
    y = np.zeros_like(x)

    ang_sep = x * Rp/(dist * 30856775814671.914)


    ################################################################
    # Compute the rotation angle of the planet at each observation #
    ################################################################

    theta_final = theta_init + obs_time / (prot * 24 * 60.) * 360.
    theta = np.linspace(theta_init, theta_final, num_eval)

    #Compute the flux throughout the observation
    map.obl = obl
    flux = map.flux(theta = theta, xs = x, ys = y, zs = z, rs = Rstar/Rp)

    #############################################################
    # Integrate over the exposure time to get the observed flux #
    #############################################################
    int_flux = np.zeros((num_nights, int(obs_per_night[0]), len(map.wav)))

    obs_params = {
        'x': np.zeros((num_nights, int(obs_per_night[0]))), 
        'z': np.zeros((num_nights, int(obs_per_night[0]))), 
        'theta': np.zeros((num_nights, int(obs_per_night[0]))),
        'ang_sep': np.zeros((num_nights, int(obs_per_night[0]))),
        'obs_phase': np.zeros((num_nights, int(obs_per_night[0])))
        }
    
    for night in range(num_nights):
        for i in range(len(t_obs[night])):
            start = int(t_obs[night][i] / comp_time)
            #print(start)
            end = start + int(texp / comp_time)
            #print(end)
            int_flux[night][i] = integrate.simpson(flux[start:end], dx = comp_time, axis = 0)

            obs_params['x'][night][i] = x[start]
            obs_params['z'][night][i] = z[start]
            obs_params['theta'][night][i] = theta[start]
            obs_params['ang_sep'][night][i] = ang_sep[start]
            obs_params['obs_phase'][night][i] = obs_phase[start]


    #########################
    # Add noise to the flux #
    #########################

    signal = np.max(int_flux) - np.min(int_flux)
    scale = signal / snr
    noise = np.random.normal(0, scale, size = (num_nights, int(obs_per_night[0]), len(map.wav)))
    int_flux_noise = int_flux + noise

    R = 700 / ((wav[-1] - wav[0])/len(wav)) # Resolution

    print("This observation lasts for {} days at R = {} (at 700 nm)."
        .format(
            np.round(obs_time/(60*24), decimals=2), 
            np.round(R, decimals = 2)
            ))
    
    print("The planet moves from phase angle between {} and {} degrees and at a rotation angle between {} and {} degrees."
          .format(
            obs_phase[0], 
            np.round(obs_phase[-1], decimals=1), 
            theta[0], 
            np.round(theta[-1]%360., decimals=1)
    ))

    print("The planet completes {} rotations and {} orbits."
          .format(
            np.round((theta[-1] - theta[0])/360., decimals=2),
            np.round((obs_phase[-1] - obs_phase[0])/360., decimals=2)
          ))
    
    ##########################
    # Plot the observed flux #
    ##########################
    
    elt_angles = (2*700E-9/38.5, 3*700E-9/38.5)
    elt_labels = ('2lambda/D', '3lambda/D')

    
    # Create a figure and axes
    fig, axs = plt.subplots(num_nights, 5, figsize=(37.5,5 * num_nights))
    cmap = plt.get_cmap('viridis')

    for night in range(num_nights):
        # Get the current axis
        ax = axs[night][0]
        # Create a color mapping
        norm = plt.Normalize(vmin=np.min(int_flux[night]*1E9), vmax=np.max(int_flux[night]*1E9))
        # Plot the flux as an image with color mapping
        im = ax.imshow(int_flux[night]*1E9, cmap=cmap, norm=norm, aspect='auto', origin = 'lower',
                        extent=(wav[0], wav[-1], t_obs[night][0] - night * 24 * 60., t_obs[night][-1] - night * 24 * 60.))
        # Add a color bar to show the mapping
        cbar = ax.figure.colorbar(im, ax=ax)
        # Add labels and titles
        ax.set_ylabel('Obs Time (Night {})'.format(night + 1))
        cbar.ax.set_ylabel('Flux (ppb)')
        ax.set_xlabel('Wavelength (micron)')

        ax = axs[night][1]
        norm = plt.Normalize(vmin=np.min(int_flux_noise[night]*1E9), vmax=np.max(int_flux_noise[night]*1E9))
        im = ax.imshow(int_flux_noise[night]*1E9, cmap=cmap, norm=norm, aspect='auto', origin = 'lower',
                        extent=(wav[0], wav[-1], t_obs[night][0] - night * 24 * 60., t_obs[night][-1] - night * 24 * 60.))
        cbar = ax.figure.colorbar(im, ax=ax)
        
        ax.set_xlabel('Wavelength (micron)')
        ax.set_ylabel('Obs Time (Night {})'.format(night + 1))
        cbar.ax.set_ylabel('Flux (ppb)')

        ax = axs[night][2]
        scalarmap.show(
            theta = obs_params['theta'][night][0], 
            xs = obs_params['x'][night][0], 
            ys = np.zeros_like(obs_params['x'][night][0]), 
            zs = obs_params['z'][night][0], 
            ax = ax)
        
        ax = axs[night][3]
        scalarmap.show(
            theta = obs_params['theta'][night][-1], 
            xs = obs_params['x'][night][-1], 
            ys = np.zeros_like(obs_params['x'][night][-1]), 
            zs = obs_params['z'][night][-1], 
            ax = ax)
        
        ax = axs[night][4]
        ax.plot(t_obs[night], obs_params['ang_sep'][night])
        ax.hlines(elt_angles[0], t_obs[night][0], t_obs[night][-1], color = 'red', linestyle = '--', label = elt_labels[0])
        ax.hlines(elt_angles[1], t_obs[night][0], t_obs[night][-1], color = 'blue', linestyle = '--', label = elt_labels[1])
        ax.set_xlabel('Obs Time (Night {})'.format(night + 1))
        ax.set_ylabel('Angular Separation')
        ax.legend()
        ax_b = ax.twinx()
        ax_b.plot(t_obs[night], obs_params['obs_phase'][night], color = 'green')
        ax_b.set_ylabel('Phase Angle')

    axs[0][0].set_title('Simulated Reflection Spectrum (No Noise)')
    axs[0][1].set_title('Simulated Reflection Spectrum (With Noise)')
    axs[0][2].set_title('Illumination of Planet - Start of night')
    axs[0][3].set_title('Illumination of Planet - End of night')
    axs[0][4].set_title('Angular Separation')

    if save_plot:
        plt.savefig(filename + '_plots.png')
    if show_plot: 
        plt.show()
    if save_data:
        results = [int_flux, int_flux_noise, flux, obs_phase, theta, obs_time, t_obs, obs_params]
        pickle.dump(results, open(filename + '_results.pkl', 'wb'))

    ### DONE
    ### Add in day/night break into cadence 
    ### Need texp AND cadence
    ### Clouds

    ### TO INVESTIGATE
    ### consider weather (bad nights)...best case scenario? Worst case?
    ### If you signal multiples of the same phase, do you get better or worse than randomly integrated over phase?
    ### Add in rondomization of rotational period, obliquity, planetary properties rather than Earth
    ### Should you take a hemispherical shot? Or take a 10 degree shot over and over again and build up the signal at that phase?
    ### HZ M DWARF PLANETS ARE (probably) TIDALLY LOCKED! - How will signal vary with fixed Earth longitude?
    
    ### TO DO
    ### Then add Earth from NASA PSG?
    ### Build in random continents
    ### How much ocean vs continent on Mars x billion years ago? How much ocean vs continent on Venus? 
    ### Can make an estimate range to produce representative guess at continental converage?
    ### Need to build angular distance at different phases into model - higher angle = easier to resolve, lower angle = bigger reflected surface
    ### See ESPRESSO ETC for S/N calculator

    return int_flux, int_flux_noise, flux, obs_phase, theta, obs_time, t_obs, obs_params

###Values for Prox b
R_prox_cent_b = 1.07 * 6400.
a_prox_cent_b = 0.04856 * 150E6
P_prox_cent_b = 11.1868
dist_prox_cent_b = 1.30197
R_prox_star = 0.141 * 695508

###Call the observation function!

int_flux, int_flux_noise, flux, obs_phase, theta, obs_time, t_obs, params = earth_refl_spec_obs(
    specmap,
    scalarmap, 
    num_obs = 150, 
    num_nights = 5, 
    cadence = 15., 
    texp = 15., 
    phase_init = 20., 
    theta_init = 320., 
    snr = 5.,
    Rp = R_prox_cent_b,
    Rstar = R_prox_star,
    a = a_prox_cent_b,
    porb = P_prox_cent_b,
    prot = P_prox_cent_b,
    dist = dist_prox_cent_b, 
    filename = 'proxb'
)

print("Done.")