import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import re
import subprocess

# -----------------------------
# global constants in cgs units
# -----------------------------
M_Sun = 1.9891e33               # [M_sun] = g
G = 6.67259e-8                  # [G] = cm3 g-1 s-2 
au = 14959787066000.            # [Astronomical unit] = cm
mp = 1.6726219e-24              # [proton mass] = g
pi = 3.141592653589793116       # [pi]
R_Sun = 6.961e10                # [R_sun] = cm
pc = 3.085678e18                # [pc] = cm
kB = 1.380658e-16               # [kB] = erg K-1
c  = 2.99792458e10              # [c] = cm s-1
h  = 6.6261e-27                 # [h = Planck constant] = g cm2 s-1


# special color scale for 1D plots:
c20 = [(31, 119, 180), (255, 127, 14), (174, 199, 232), (255, 187, 120),    
       (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
       (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
       (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
       (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
for i in range(len(c20)):
    r, g, b = c20[i]    
    c20[i] = (r / 255., g / 255., b / 255.)

    
# -------------------------
# read input parameter file
# -------------------------
verbose = 'No'
params = open("params.dat",'r')
lines_params = params.readlines()
params.close()                 

par = []                       # allocating a dictionary
var = []                       # allocating a dictionary
regex = re.compile(',')

for line in lines_params:      
    try:
        line.split()[0][0]     # check if blank line (GWF)
    except:
        continue
    if (line.split()[0][0]=='#'): # check if line starts with a # (comment)
        continue
    else:
        name, value = line.split()[0:2]
    try:
        float(value)           # first trying with float
    except ValueError:         # if it is not float
        try:
            int(value)         # we try with integer
        except ValueError:     # if it is not integer nor float
            # we try with array with several values separated by a comma (,)
            if(regex.search(value) != None):  
                if name == 'dustfluids':
                    value = [int(x) for x in value.split(',')]
            else:
                value = '"' + value + '"'   # if none of the above tests work, we know value is a string
    par.append(value)
    var.append(name)

for i in range(len(par)):
    exec(var[i]+"="+str(par[i]))

# verbose mode option
if verbose == 'Yes':
    for i in range(len(par)):
        print(var[i]+"="+str(par[i]))


# -------------------------        
# Below we work out specific parameters and/or error messages
# -------------------------

# was simulation carried out with Fargo3D?
hydro2D = 'Yes'
fargo3d = 'No'
if os.path.isfile(dir+'/variables.par') == True:
    fargo3d = 'Yes'

if fargo3d == 'No':
    dustfluids = 'No'
    

# Check if Fargo3D simulation was carried out in 2D or in 3D by
# fetching NZ in the variables.par file:
if fargo3d == 'Yes':
    command = 'awk " /^NZ/ " '+dir+'/*.par'
    if sys.version_info[0] < 3:
        buf = subprocess.check_output(command, shell=True)
    else:
        buf = subprocess.getoutput(command)
    Nz = float(buf.split()[1])
    if Nz > 1:
        hydro2D = 'No'

# Define parameters for use by RADMC-3D
# Case where only dust transfer is calculated:
if RTdust_or_gas == 'dust':
    incl_dust  = 1
    incl_lines = 0
    linenlam = 1
    if not('dust_interpolation' in open('params.dat').read()):
        dust_interpolation = 'T'
    if not('dustsublimation' in open('params.dat').read()):
        dustsublimation = 'No'
    
# Case where only line transfer is calculated:
if RTdust_or_gas == 'gas':
    incl_lines = 1
    incl_dust  = 0
    polarized_scat = 'No'
    # currently in our RT gas calculations, the gas temperature has to
    # be that of the hydro simulation (it cannot be equal to the dust
    # temperature computed by a RT Monte-Carlo calculation)
    Tdust_eq_Thydro = 'Yes'
    
# Case where both dust and line transfers are calculated
if RTdust_or_gas == 'both':
    incl_lines = 1
    incl_dust  = 1
    if not('dust_interpolation' in open('params.dat').read()):
        dust_interpolation = 'T'
    if not('dustsublimation' in open('params.dat').read()):
        dustsublimation = 'No'

if recalc_radmc == 'Yes':
    recalc_rawfits = 'Yes'
    recalc_fluxmap = 'Yes'
if recalc_rawfits == 'Yes':
    recalc_fluxmap = 'Yes'

nb_photons = int(nb_photons)
nb_photons_scat = int(nb_photons_scat)
    
if incl_lines == 1 and moment_order == 1 and inclination == 0.0:
    sys.exit("To get an actual velocity map you need a non-zero disc inclination. Abort!")

if polarized_scat == 'Yes' and scat_mode != 5:
    scat_mode = 5
    print("To get a polarized intensity image you need scat_mode = 5 in params.dat! I've switched scat_mode to 5, please check this is what you aim for!")
    
#if ( (fargo3d == 'No' and polarized_scat == 'Yes') or (fargo3d == 'Yes' and dustfluids == 'No') ):
if polarized_scat == 'Yes':
    z_expansion = 'G'  # Gaussian vertical expansion of dust surface density

if ncol%2 == 1:
    ncol += 1
    print('the number of columns needs to be an even number, I have thus increased ncol by one.')

if axi_intensity == 'Yes':
    deproj_polar = 'Yes'
    
if override_units == 'Yes':
    if new_unit_length == 0.0:
        sys.exit('override_units set to yes but new_unit_length is not defined in params.dat, I must exit!')
    if new_unit_mass == 0.0:
        sys.exit('override_units set to yes but new_unit_mass is not defined in params.dat, I must exit!')

# x-axis flip means that we apply a mirror-symetry x -> -x to the 2D simulation plane,
# which in practice is done by adding 180 degrees to the disc's inclination wrt the line of sight
inclination_input = inclination
if xaxisflip == 'Yes':
    inclination = inclination + 180.0

# this is how the beam position angle should be modified to be understood as
# measuring East from North. You can check this by setting check_beam to Yes
# in the params.dat parameter file
bpaangle = -90.0-bpaangle


# Dust global parameters in Fargo3D simulations
if fargo3d == 'Yes' and (RTdust_or_gas == 'dust' or RTdust_or_gas == 'both'):
    command = 'awk " /^DUSTINTERNALRHO/ " '+dir+'/variables.par'
    # check which version of python we're using
    if sys.version_info[0] < 3:   # python 2.X
        buf = subprocess.check_output(command, shell=True)
    else:                         # python 3.X
        buf = subprocess.getoutput(command)
    dust_internal_density = float(buf.split()[1])   # in g / cm^3
    # case where dust fluids are simulated
    if dustfluids != 'No':
        # find out how many dust fluids there are:
        input_file = dir+'/dustsizes.dat'
        dust_id, dust_size, dust_gas_ratio = np.loadtxt(input_file,unpack=True)
        nbin = len(np.atleast_1d(dust_id))   # in case only a single dust fluid is used
        if nbin != 1:
            amin = dust_size[dustfluids[0]-1]
            amax = dust_size[dustfluids[1]-1]
            nbin = dustfluids[1]-dustfluids[0]+1
            bins = dust_size[dustfluids[0]-1:dustfluids[1]]
        else: # case only a single dust fluid is used
            amin = dust_size
            amax = amin
            nbin = 1
            bins = [dust_size]
            dust_id = [dust_id]
            dust_size = [dust_size]
            dust_gas_ratio = [dust_gas_ratio]
        '''
        # old text below could be used when there is no feedback on gas 
        # and that we want to remormalize stuff?
        amin = dust_size.min()
        amax = dust_size.max()
        bins = np.logspace(np.log10(amin), np.log10(amax), nbin)   
        ratio = np.sum(dust_gas_ratio)
        command = 'awk " /^DUSTSLOPEDIST/ " '+dir+'/variables.par'
        # check which version of python we're using
        if sys.version_info[0] < 3:   # python 2.X
            buf = subprocess.check_output(command, shell=True)
        else:                         # python 3.X
            buf = subprocess.getoutput(command)
        pindex = -float(buf.split()[1])
        '''
    else:
        bins = np.logspace(np.log10(amin), np.log10(amax), nbin+1) 
else:
    bins = np.logspace(np.log10(amin), np.log10(amax), nbin+1) 
        
# label for the name of the image file created by RADMC3D
if RTdust_or_gas == 'dust':
    label = dir+'_o'+str(on)+'_p'+str(pindex)+'_r'+str(ratio)+'_a'+str(amin)+'_'+str(amax)+'_nb'+str(nbin)+'_mode'+str(scat_mode)+'_np'+str('{:.0e}'.format(nb_photons))+'_nc'+str(ncol)+'_z'+str(z_expansion)+'_xf'+str(xaxisflip)+'_Td'+str(Tdust_eq_Thydro)
    
if RTdust_or_gas == 'gas':
    if widthkms == 0.0:
        label = dir+'_o'+str(on)+'_gas'+str(gasspecies)+'_iline'+str(iline)+'_lmode'+str(lines_mode)+'_ab'+str('{:.0e}'.format(abundance))+'_vkms'+str(vkms)+'_turbvel'+str(turbvel)+'_nc'+str(ncol)+'_xf'+str(xaxisflip)+'_Td'+str(Tdust_eq_Thydro)
    else:
        label = dir+'_o'+str(on)+'_gas'+str(gasspecies)+'_iline'+str(iline)+'_lmode'+str(lines_mode)+'_ab'+str('{:.0e}'.format(abundance))+'_widthkms'+str(widthkms)+'_turbvel'+str(turbvel)+'_nc'+str(ncol)+'_xf'+str(xaxisflip)+'_dustandgas'

if RTdust_or_gas == 'both':
    label = dir+'_o'+str(on)+'_gas'+str(gasspecies)+'_iline'+str(iline)+'_lmode'+str(lines_mode)+'_ab'+str('{:.0e}'.format(abundance))+'_widthkms'+str(widthkms)+'_nc'+str(ncol)+'_xf'+str(xaxisflip)+'_nb'+str(nbin)+'_mode'+str(scat_mode)+'_dustandgas'
    
# name of .fits file where data is output
# Note that M.label will disentangle dust and gas line calculations
if plot_tau == 'No':
    if brightness_temp == 'Yes':
        image = 'imageTb_'
    else:
        image = 'image_'
    if (RTdust_or_gas == 'gas' or RTdust_or_gas == 'both'):
        # no need for lambda in file's name for gas RT calculations...
        outfile = image+str(label)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)  
    if RTdust_or_gas == 'dust':
        outfile = image+str(label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)
        
else:
    if (RTdust_or_gas == 'gas' or RTdust_or_gas == 'both'):
        # no need for lambda in file's name for gas RT calculations...
        outfile = 'tau_'+str(label)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)  
    if RTdust_or_gas == 'dust':
        outfile = 'tau_'+str(label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)
        
if secondorder == 'Yes':
    outfile += '_so'
if ('bin_small_dust' in open('params.dat').read()) and (bin_small_dust == 'Yes'):
    outfile += '_bin0'
if ('dustdens_eq_gasdens' in open('params.dat').read()) and (dustdens_eq_gasdens == 'Yes'):
    outfile += '_ddeqgd'
if add_noise == 'Yes':
    outfile += '_wn'+str(noise_dev_std)
    
outputfitsfile_wholedatacube = outfile+'_all.fits'

if (RTdust_or_gas == 'gas' or RTdust_or_gas == 'both'):
    outfile += '_moment'+str(moment_order)

if (RTdust_or_gas == 'both' and subtract_continuum == 'Yes'):
    outfile += '_contsub'
   
outputfitsfile = outfile+'.fits'

if os.path.isfile(outputfitsfile) == False and recalc_rawfits == 'No':
    print('file '+outputfitsfile+' is missing, I need to activate recalc_rawfits')
    recalc_rawfits = 'Yes'

if recalc_radmc == 'Yes' and os.path.isfile('image.fits') == True:
    os.system('rm -f image.fits')
    if verbose == 'Yes':
        print('image.fits already existing: I need to erase it!...')
    
if not('bin_small_dust' in open('params.dat').read()):
    bin_small_dust = 'No'
if not('dustdens_eq_gasdens' in open('params.dat').read()):
    dustdens_eq_gasdens = 'No'

    
# Set of parameters that we only need to read or specify even if
# radmc3d is not called get the aspect ratio and flaring index used in
# the numerical simulation note that this works both for Dusty
# FARGO-ADSG and FARGO3D simulations
command = 'gawk " BEGIN{IGNORECASE=1} /^AspectRatio/ " '+dir+'/*.par'
# check which version of python we're using
if sys.version_info[0] < 3:   # python 2.X
    buf = subprocess.check_output(command, shell=True)
else:                         # python 3.X
    buf = subprocess.getoutput(command)
aspectratio = float(buf.split()[1])

# get the flaring index used in the numerical simulation note that
# this works both for Dusty FARGO-ADSG and FARGO3D simulations
command = 'gawk " BEGIN{IGNORECASE=1} /^FlaringIndex/ " '+dir+'/*.par'
if sys.version_info[0] < 3:
    buf = subprocess.check_output(command, shell=True)
else:
    buf = subprocess.getoutput(command)
flaringindex = float(buf.split()[1])
    
# get the alpha viscosity used in the numerical simulation
if fargo3d == 'No':
    try:
        command = 'awk " /^AlphaViscosity/ " '+dir+'/*.par'
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        alphaviscosity = float(buf.split()[1])
# if no alphaviscosity, then try to see if a constant
# kinematic viscosity has been used in the simulation
    except IndexError:
        command = 'awk " /^Viscosity/ " '+dir+'/*.par'
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        viscosity = float(buf.split()[1])
        # simply set constant alpha value as nu / h^2 (ok at code's unit of length)
        alphaviscosity = viscosity * (aspectratio**(-2.0))
if fargo3d == 'Yes':
    try:
        command = 'awk " /^ALPHA/ " '+dir+'/*.par'
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        alphaviscosity = float(buf.split()[1])
        if alphaviscosity == 0.0:
            command = 'awk " /^NU/ " '+dir+'/*.par'
            if sys.version_info[0] < 3:
                buf = subprocess.check_output(command, shell=True)
            else:
                buf = subprocess.getoutput(command)
            viscosity = float(buf.split()[1])
            if viscosity != 0.0:
                alphaviscosity = viscosity * (aspectratio**(-2.0))
# if no alphaviscosity, then try to see if a constant
# kinematic viscosity has been used in the simulation
    except IndexError:
        command = 'awk " /^NU/ " '+dir+'/*.par'
        if sys.version_info[0] < 3:
            buf = subprocess.check_output(command, shell=True)
        else:
            buf = subprocess.getoutput(command)
        viscosity = float(buf.split()[1])
        # simply set constant alpha value as nu / h^2 (ok at code's unit of length)
        alphaviscosity = viscosity * (aspectratio**(-2.0))
            
# get the grid's radial spacing
if fargo3d == 'No':
    command = 'awk " /^RadialSpacing/ " '+dir+'/*.par'
    if sys.version_info[0] < 3:
        buf = subprocess.check_output(command, shell=True)
    else:
        buf = subprocess.getoutput(command)
    radialspacing = str(buf.split()[1])

# Get gas surface density field from hydro simulation, with
# the aim to inherit from the parameters attached to the mesh
# structure (rmed, Nrad etc.)
from field import *
from mesh import *
gas  = Field(field='gasdens'+str(on)+'.dat', directory=dir)

if (minmaxaxis == '#'):
    minmaxaxis = 1000.0   # arbitrarilly large number

# Color map
if not('mycolormap' in open('params.dat').read()):
    mycolormap = 'nipy_spectral'
if ((RTdust_or_gas == 'gas' or RTdust_or_gas == 'both') and moment_order == 1):
    mycolormap = 'RdBu_r'
