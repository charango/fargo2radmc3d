# =================================================================== 
#                        FARGO to RADMC-3D
#
# Program post-processes FARGO2D/3D outputs as inputs for RADMC-3D.
# It can run with either Python 2.X or Python 3.X.
# 
# code written by Clement Baruteau (CB), Sebastian Perez (SP) and
# Marcelo Barraza (MB) with substantial contributions from Simon
# Casassus (SC) and Gaylor Wafflard-Fernandez (GWF)
#
# =================================================================== 

# =================================
#            TO DO LIST
# =================================
# - write gas T and vel in binary format for gas RT calc
# - add text banner with 'banner' command
# - results from actual 3D simulations from Fargo3D (check
# fargo2python): check latitude expression in mesh.py
# - work out case where RTdust_or_gas = 'both'
# - check again that all goes well without x-axisflip!
# =================================


# =======================
# IMPORT GLOBAL VARIABLES
# =======================
print('--------- Reading and interpreting params.dat parameter file ----------')
import par


# =====================================
# DUST RADIATIVE TRANSFERT CALCULATIONS
# =====================================
if par.RTdust_or_gas == 'dust':

    # 1. We first compute the dust temperature in case RADMC-3D does not
    # do it via a Monte Carlo run. We call function
    # compute_gas_temperature for this purpose:
    if par.Tdust_eq_Thydro == 'Yes':
        from gas_temperature import *
        if par.recalc_dust_temperature == 'Yes':
            print('--------- Setting dust temperature to gas hydro temperature ----------')
            compute_gas_temperature()
        else:
            print('--------- I did not compute dust temperature (recalc_dust_temperature = No in params.dat file) ----------')
        if par.plot_dust_quantities == 'Yes':
            print('--------- Plotting dust "hydro" temperature ----------')
            plot_gas_temperature()
    
    # 2. We then compute the dust mass volume density
    if par.recalc_dust_density == 'Yes':
        from dust_density import *
        print('--------- Computing dust surface density ----------')
        compute_dust_mass_volume_density()
    else:
        print('--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------')
    # Plot dust density if dust sublimation not taken into account. To
    # take dust sublimation into account, we first need the dust
    # temperature to be computed.
    if par.plot_dust_quantities == 'Yes':
        from dust_density import *
        print('--------- Plotting dust density ----------')
        if par.dustsublimation == 'Yes':
            plot_dust_density('before')
        else:
            plot_dust_density('')
            
    # 3. Calculation of dust opacities follows
    from dust_opacities import *
    if par.recalc_opac == 'Yes':
        print('--------- Computing dust opacities ----------')
        compute_dust_opacities()
    else:
        print('--------- I did not compute dust opacities (recalc_opac = No in params.dat file) ------ ')
    if par.plot_dust_quantities == 'Yes':
        from dust_opacities import *
        print('--------- Plotting dust opacities ----------')
        plot_opacities(species=par.species,amin=par.amin,amax=par.amax,nbin=par.nbin,lbda1=par.wavelength*1e3)
    # Write dustopac.inp file even if we don't (re)calculate dust opacities
    write_dustopac(par.species,par.nbin)

    
# ====================================
# GAS RADIATIVE TRANSFERT CALCULATIONS
# ====================================
if par.RTdust_or_gas == 'gas':

    # start by computing gas temperature and then mass volume density
    if par.recalc_gas_quantities == 'Yes':

        from gas_density import *
        from gas_temperature import *
        from gas_velocity import *
        
        print('--------- Computing gas temperature ----------')
        compute_gas_temperature()
        print('--------- Computing gas mass volume density ----------')
        compute_gas_mass_volume_density()
        print('--------- Writing microturbulence file ----------')
        write_gas_microturb()
        print('--------- Computing gas velocity ----------')
        compute_gas_velocity()
    else:
        print('--------- I did not compute the gas density, temperature nor velocity (recalc_gas_quantities = No in params.dat file) ----------')

    
# ===============
# CALL TO RADMC3D 
# ===============

# write radmc3d script, even if recalc_radmc is set to No -- this can
# be useful if radmc3d mctherm / ray_tracing are run on a different
# platform
from radmc_inputs import *
write_radmc3d_script()

if par.recalc_radmc == 'Yes':
    
    # Write other parameter files required by RADMC3D
    print('--------- Printing auxiliary files ----------')
    write_wavelength()
    write_stars(Rstar = par.rstar, Tstar = par.teff)
    write_AMRgrid(par.gas, Plot=False)
    if par.RTdust_or_gas == 'gas':
        write_lines(str(par.gasspecies),par.lines_mode)        
    # rto_style = 3 means that RADMC3D will write binary output files
    # setthreads corresponds to the number of threads (cores) over which radmc3d runs
    write_radmc3dinp(incl_dust=par.incl_dust, incl_lines=par.incl_lines, lines_mode=par.lines_mode, nphot_scat=par.nb_photons_scat, nphot=par.nb_photons, rto_style=3, tgas_eq_tdust=par.tgas_eq_tdust, modified_random_walk=1, scattering_mode_max=par.scat_mode, setthreads=par.nbcores)

    if par.RTdust_or_gas == 'dust' and par.Tdust_eq_Thydro == 'No':
        print('--------- Running thermal MC calculation ----------')
        os.system('radmc3d mctherm')

        # Plot dust temperature after call to radmc3d thermal MC calculation
        if par.plot_dust_quantities == 'Yes':
            from dust_temperature import *
            print('--------- Plotting dust temperature ----------')
            if par.dustsublimation == 'No':
                plot_dust_temperature('')
            else:
                plot_dust_temperature('before')

        # Recompute dust density to take sublimation into account
        if par.dustsublimation == 'Yes':
            from dust_density import *
            print('--------- Computing again dust density ----------')
            recompute_dust_mass_volume_density()
            if par.plot_dust_quantities == 'Yes':
                print('--------- Plotting again dust density ----------')
                plot_dust_density('after')

            print('--------- Running again thermal MC calculation ----------')
            os.system('radmc3d mctherm')

            if par.plot_dust_quantities == 'Yes':
                print('--------- Plotting again dust temperature ----------')
                plot_dust_temperature('after')
                
    # Now run RADMC3D
    print('--------- Now executing RADMC3D ----------')
    os.system('./script_radmc')
    
    
# ============================
# POST-PROCESS RADMC3D RESULTS
# ============================

# First export image created by RADMC-3D into a fits file
if (par.recalc_rawfits == 'Yes'):
    print('--------- Saving results in fits file ----------')
    from radmc_to_fits import *
    outfile = exportfits()
    
# Then convolve raw flux with beam and produce final image
if par.recalc_fluxmap == 'Yes':
    print('--------- Convolving and writing final image ----------')
    from final_image import *
    produce_final_image()


print('--------- Done! ----------')
