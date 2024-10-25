import numpy as np
#import psutil
import sys
import subprocess

import matplotlib
import matplotlib.pyplot as plt

# import global variables
import par

from field import *
from mesh import *

avgstokes  = np.zeros(par.nbin)     # average Stokes number of particles per bin size
    

# =========================
# Compute dust mass surface density for each size bin
# =========================
def compute_dust_mass_surface_density():

    # extra useful quantities (code units)
    Rinf = par.gas.redge[0:len(par.gas.redge)-1]   # inner radial interface between grid cells
    Rsup = par.gas.redge[1:len(par.gas.redge)]     # outer radial interface
    surface  = np.zeros(par.gas.data.shape) # 2D array containing surface of each grid cell
    surf = np.pi * (Rsup*Rsup - Rinf*Rinf) / par.gas.nsec # surface of each grid cell (code units)
    for th in range(par.gas.nsec):
        surface[:,th] = surf
    if par.fargo3d == 'No':
        Pinf = par.gas.pedge[0:len(par.gas.redge)-1]   # inner azimuthal interface between grid cells
        Psup = par.gas.pedge[1:len(par.gas.redge)]     # outer azimuthal interface
        if par.radialspacing != 'L':
            delta_r = Rsup[0]-Rinf[0]
        else:
            delta_r = np.log(Rsup[0]/Rinf[0])
        delta_phi = Psup[0]-Pinf[0]
        
    # Mass of gas in units of the star's mass
    Mgas = np.sum(par.gas.data*surface)
    
    # allocate array
    dust = np.zeros((par.gas.nsec*par.gas.nrad*par.nbin))
    nparticles = np.zeros(par.nbin)     # number of particles per bin size

    '''
    # check out memory available on your architecture
    mem_gib = float(psutil.virtual_memory().total)
    mem_array_gib = par.gas.nrad*par.gas.nsec*par.gas.ncol*par.nbin*8.0
    if (mem_array_gib/mem_gib > 0.5):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Beware that the memory requested for allocating the dust mass volume density or the temperature arrays')
        print('is very close to the amount of memory available on your architecture...')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    '''
    
    # -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # CASE 1: Dusty FARGO-ADSG simulation (Lagrangian particles) case where the
    # dust's surface density is inferred from the spatial distribution of
    # Lagrangian super-particules. This is the case when dust continuum
    # emission is computed. For polarized scattered light prediction from Dusty
    # FARGO-ADSG simulation we will simply use the gas surface density as input
    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    if (par.fargo3d == 'No' and par.polarized_scat == 'No'):
        
        # read information on the dust particles
        (rad, azi, vr, vt, Stokes, a) = np.loadtxt(par.dir+'/dustsystat'+str(par.on)+'.dat', unpack=True)

        # ------------------
        # Populate dust bins
        # ------------------
        for m in range (len(a)):   # CB: sum over dust particles
            
            r = rad[m]
            # radial index of the cell where the particle is
            if par.radialspacing == 'L':
                ip = int(np.log(r/par.gas.redge.min())/np.log(par.gas.redge.max()/par.gas.redge.min()) * par.gas.nrad)
            else:
                ip = int((r-par.gas.redge.min())/(par.gas.redge.max()-par.gas.redge.min()) * par.gas.nrad)
            if (ip < 0 or ip >= par.gas.nrad):
                sys.exit('pb with ip = ', ip, ' in compute_dust_mass_surface_density step: I must exit!')
            
            t = azi[m]
            # azimuthal index of the cell where the particle is
            # (general expression since grid spacing in azimuth is always arithmetic)
            jp = int((t-par.gas.pedge.min())/(par.gas.pedge.max()-par.gas.pedge.min()) * par.gas.nsec)
            if (jp < 0 or jp >= par.gas.nsec):
                sys.exit('pb with jp = ', jp, ' in compute_dust_mass_surface_density step: I must exit!')
                
            # particle size
            pcsize = a[m]
            
            # find out which bin particle belongs to
            ibin = int(np.log(pcsize/par.bins.min())/np.log(par.bins.max()/par.bins.min()) * par.nbin)
            if (ibin >= 0 and ibin < par.nbin):
                nparticles[ibin] += 1
                avgstokes[ibin] += Stokes[m]

                '''
                k = ibin*par.gas.nsec*par.gas.nrad + jp*par.gas.nrad + ip
                # this basically means we're injecting the whole
                # particle mass in the cell where it is located, how
                # about doing a more fancy interpolation scheme?
                dust[k] +=1
                '''
                
                # fill dust array based on particles mass
                # interpolation scheme. The algorithm is the same as
                # that in Dusty FARGO-ADSG (see Dsys.c)
                if par.dust_interpolation == 'N':
                    if m==0:
                        print('Nearest-grid point interpolation done for the dust surface density')
                    myip = ip
                    myjp = jp
                    imin = myip
                    imax = myip
                    jmin = myjp
                    jmax = myjp
                    
                if par.dust_interpolation == 'C':
                    if m==0:
                        print('Cloud-in-cell interpolation done for the dust surface density')
                    if (r < par.gas.rmed[ip]):
                        myip = ip-1
                    else:
                        myip = ip
                    if (t < par.gas.pmed[jp]):
                        myjp = jp-1
                    else:
                        myjp = jp
                    imin = myip
                    imax = myip+1
                    jmin = myjp
                    jmax = myjp+1

                # is now the default interpolation scheme
                if par.dust_interpolation == 'T':
                    if m==0:
                        print('Triangular-shaped cloud interpolation done for the dust surface density')
                    myip = ip
                    myjp = jp
                    imin = myip-1
                    imax = myip+1
                    jmin = myjp-1
                    jmax = myjp+1
                    
                for j in range(jmin,jmax+1):  # j varies from jmin to jmax included
                    if ( (j >= 0) and (j < par.gas.nsec) ):
                        myj = j
                        myazimuth = par.gas.pmed[j]
                    else:
                        if (j < 0):
                            myj = j+par.gas.nsec
                            myazimuth = par.gas.pmed[0]+j*delta_phi
                        if (j >= par.gas.nsec):
                            myj = j-par.gas.nsec
                            myazimuth = par.gas.pmed[par.gas.nsec-1]+(j+1-par.gas.nsec)*delta_phi
                    dphi = np.abs(t-myazimuth)
                    
                    for i in range(imin,imax+1):  # i varies from imin to imax included
                        if ( (i < 0) or (i > par.gas.nrad-1 ) ):
                            dr = 0.0
                        else:
                            if par.radialspacing == 'L':
                                dr = np.abs(np.log(r/par.gas.rmed[i]))
                            else:
                                dr = np.abs(r-par.gas.rmed[i])

                        # Nearest Grid Point (NGP) interpolation
                        if par.dust_interpolation == 'N':
                            if (dr < 0.5*delta_r):
                                wr = 1.0
                            else:
                                wr = 0.0
                            wt = 1.0
                        # Cloud-In-Cell (CIC) a.k.a. bilinear interpolation
                        if par.dust_interpolation == 'C':
                            wr = 1.0-dr/delta_r
                            wt = 1.0-dphi/delta_phi
                        # Triangular-Shaped Cloud (TSC) a.k.a. quadratic spline interpolation
                        # is now the default interpolation scheme
                        if par.dust_interpolation == 'T':
                            if (dr < 0.5*delta_r):
                                wr = 0.75-dr*dr/delta_r/delta_r
                            if ( (dr >= 0.5*delta_r) and (dr <= 1.5*delta_r) ):
                                wr = 0.5*(1.5-dr/delta_r)*(1.5-dr/delta_r)
                            if (dr > 1.5*delta_r):
                                wr = 0.0
                            if (dphi < 0.5*delta_phi): 
                                wt = 0.75-dphi*dphi/delta_phi/delta_phi
                            if ( (dphi >= 0.5*delta_phi) and (dphi <= 1.5*delta_phi) ):
                                wt = 0.5*(1.5-dphi/delta_phi)*(1.5-dphi/delta_phi)
                            if (dphi > 1.5*delta_phi):
                                wt = 0.0
                                
                        if ( (wr<0.0) or (wr>1.0) or (wt<0.0) or (wt>1.0) ):
                            print("interpolation problem with particle index",m," with wr, wt, i, j=", wr, wt, i, j)
                                
                        if ( (i >= 0) and (i <= par.gas.nrad-1) ):
                            k = ibin*par.gas.nsec*par.gas.nrad + myj*par.gas.nrad + i
                            dust[k] += (wr*wt)
                                
                                
        for ibin in range(par.nbin):
            if nparticles[ibin] == 0:
                nparticles[ibin] = 1
            avgstokes[ibin] /= nparticles[ibin]
            if par.verbose == 'Yes':
                print(str(nparticles[ibin])+' grains between '+str(par.bins[ibin])+' and '+str(par.bins[ibin+1])+' meters')

        # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
        dustcube = dust.reshape(par.nbin, par.gas.nsec, par.gas.nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
        del dust

        if par.verbose == 'Yes':
            print('Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*par.gas.cumass))
        
        # ------------------
        # finally compute dust surface density for each size bin
        # ------------------
        frac = np.zeros(par.nbin)
        for ibin in range(par.nbin):
            
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (par.bins[ibin+1]**(4.0-par.pindex) - par.bins[ibin]**(4.0-par.pindex)) / (par.amax**(4.0-par.pindex) - par.amin**(4.0-par.pindex))

            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = par.ratio * Mgas * frac[ibin]
            if par.verbose == 'Yes':
                print('Dust mass [in units of Mstar] in species ', ibin, ' = ', M_i_dust, ' with npc = ', nparticles[ibin])

            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
            dustcube[ibin,:,:] *= M_i_dust / surface / nparticles[ibin]

            # conversion in g/cm^2
            dustcube[ibin,:,:] *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
            
        # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
        if par.bin_small_dust == 'Yes':
            frac[0] *= 1e3
            if par.verbose == 'Yes':
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
                print("Mass fraction of bin 0 changed to: ",str(frac[0]))
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            imin = np.argmin(np.abs(par.gas.rmed-1.4))  # radial index corresponding to 0.3"
            imax = np.argmin(np.abs(par.gas.rmed-2.8))  # radial index corresponding to 0.6"
            dustcube[0,imin:imax,:] = par.gas.data[imin:imax,:] * par.ratio * frac[0] * (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec

        if par.verbose == 'Yes':
            print('Total dust mass [g] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.))
            print('Total dust mass [Mgas] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(Mgas*par.gas.cumass*1e3))
            print('Total dust mass [Mstar] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(par.gas.cumass*1e3))
    
        # Total dust surface density
        dust_surface_density = np.sum(dustcube,axis=0)
        if par.verbose == 'Yes':
            print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

    # -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # CASE 2: FARGO 3D simulation carried out in 2D: case where the
    # dust's surface density is directly read from the simulation
    # outputs. This can be the case for the calculation of the
    # continuum emission as well as of polarized scattered light (in
    # which case, we require dustfluids to be set to i,j in the input
    # parameter file, where i and j denote the indices of the first
    # and last dust fluids to be considered. If dustfluids is set to
    # No, then the gas surface density will be used as input.
    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    if (par.fargo3d == 'Yes' and par.dustfluids != 'No'):

        dustcube = dust.reshape(par.nbin, par.gas.nsec, par.gas.nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
        del dust

        # ------------------
        # dust surface density in each size bin directly from the simulation
        # outputs
        # ------------------
        for ibin in range(par.nbin):

            index = int(par.dust_id[par.dustfluids[0]-1+ibin])
            fileread = 'dust'+str(index)+'dens'+str(par.on)+'.dat'
            
            # read dust surface density for each dust fluid in code units
            dustcube[ibin,:,:]  = Field(field=fileread, directory=par.dir).data
        
            # conversion in g/cm^2
            dustcube[ibin,:,:] *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
            
            # decrease dust surface density inside mask radius
            # NB: mask_radius is in arcseconds
            rmask_in_code_units = par.mask_radius*par.distance*par.au/par.gas.culength/1e2
            for i in range(par.gas.nrad):
                if (par.gas.rmed[i] < rmask_in_code_units):
                    dustcube[ibin,i,:] = 0.0 # *= ( (par.gas.rmed[i]/rmask_in_code_units)**(10.0) ) 

            # case we introduce 'by hand' a cavity in the dust (Baruteau et al. 2021)
            if ('cavity_pol_int' in open('params.dat').read()) and (par.cavity_pol_int == 'Yes'):
                agraincm = 1e2*par.dust_size[par.dustfluids[0]-1+ibin]
                rhopart  = 2.7  # g/cm^3 (small grains)
                mymin = dustcube[ibin,:,:].min()
                for i in range(par.gas.nrad):
                    for j in range(par.gas.nsec):
                        # local Stokes number at R,phi
                        st = 0.5*np.pi*agraincm*rhopart/(par.gas.data[i,j]*par.gas.cumass*1e3/(par.gas.culength*1e2)**2.)
                        if st > 1e-4:  # 1e-4 or 1e-3 looks like a good threshold
                            dustcube[ibin,i,j] = mymin
                    
                    
        if par.verbose == 'Yes':
            print('Total dust mass [g] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.))
            print('Total dust mass [Mgas] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(Mgas*par.gas.cumass*1e3))
            print('Total dust mass [Mstar] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(par.gas.cumass*1e3))
    
        # Total dust surface density
        dust_surface_density = np.sum(dustcube,axis=0)
        if par.verbose == 'Yes':
            print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())


    # -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # CASE 3: case where for polarized intensity maps we use the gas surface
    # density as input.  This is the default case for Dusty FARGO-ADSG
    # simulations, but can also be used for FARGO3D simulations carried out in
    # 2D for which dust fluids were included.
    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    if ( (par.fargo3d == 'No' and par.polarized_scat == 'Yes') or (par.fargo3d == 'Yes' and par.dustfluids == 'No') ):

        dustcube = dust.reshape(par.nbin, par.gas.nsec, par.gas.nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
        del dust
        
        frac = np.zeros(par.nbin)
        buf = 0.0
    
        # ------------------
        # Populate dust bins
        # ------------------
        for ibin in range(par.nbin):
            
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (par.bins[ibin+1]**(4.0-par.pindex) - par.bins[ibin]**(4.0-par.pindex)) / (par.amax**(4.0-par.pindex) - par.amin**(4.0-par.pindex))
            
            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = par.ratio * Mgas * frac[ibin]
            buf += M_i_dust
            if par.verbose == 'Yes':
                print('Dust mass [in units of Mstar] in species ', ibin, ' = ', M_i_dust)

            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi) for every size bin
            dustcube[ibin,:,:] = par.ratio * par.gas.data * frac[ibin]

            # case we introduce 'by hand' a cavity in the dust (Baruteau et al. 2021)
            if ('cavity_pol_int' in open('params.dat').read()) and (par.cavity_pol_int == 'Yes'):
                agraincm = 10.0**(0.5*(np.log10(1e2*par.bins[ibin]) + np.log10(1e2*par.bins[ibin+1])))
                rhopart  = 2.7  # g/cm^3 (small grains)
                mymin = dustcube[ibin,:,:].min()
                for i in range(par.gas.nrad):
                    for j in range(par.gas.nsec):
                        # local Stokes number at R,phi
                        st = 0.5*np.pi*agraincm*rhopart/(par.gas.data[i,j]*par.gas.cumass*1e3/(par.gas.culength*1e2)**2.)
                        if st > 1e-4:  # 1e-4 or 1e-3 looks like a good threshold
                            dustcube[ibin,i,j] = mymin

            # conversion in g/cm^2
            dustcube[ibin,:,:] *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
            
            # decrease dust surface density beyond truncation radius by R^-2
            # NB: truncation_radius is in arcseconds
            rcut_in_code_units  = par.truncation_radius*par.distance*par.au/par.gas.culength/1e2
            rmask_in_code_units = par.mask_radius*par.distance*par.au/par.gas.culength/1e2
            for i in range(par.gas.nrad):
                if (rcut_in_code_units > 0 and par.gas.rmed[i] > rcut_in_code_units):
                    dustcube[ibin,i,:] *= ( (par.gas.rmed[i]/rcut_in_code_units)**(-2.0) )  #same as Baruteau+ 2019 (sharp: -10.0 power)
                if (par.gas.rmed[i] < rmask_in_code_units):
                    dustcube[ibin,i,:] *= ( (par.gas.rmed[i]/rmask_in_code_units)**(6.0) )  #CUIDADIN!
                    #dustcube[ibin,i,:] = 0.0
                    
        if par.verbose == 'Yes':
            print('Total dust mass [g] = ', buf*par.gas.cumass*1e3)
    
        # Total dust surface density
        dust_surface_density = np.sum(dustcube,axis=0)
        if par.verbose == 'Yes':
            print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

    # finally return dustcube
    return dustcube
        
        
# =========================
# Compute dust mass volume density for each size bin (vertical
#    expansion assuming hydrostatic equilibrium for 2D hydro
#    simulations)
# =========================
def compute_dust_mass_volume_density():

    # ==========================
    # 3D simulation with fargo3d
    # ==========================
    if par.hydro2D == 'No':

        # Allocate array
        dustcube = np.zeros((par.nbin, par.gas.ncol, par.gas.nrad, par.gas.nsec))

        # ------------------
        # dust surface density in each size bin directly from the simulation
        # outputs
        # ------------------
        for ibin in range(par.nbin):

            index = int(par.dust_id[par.dustfluids[0]-1+ibin])
            fileread = 'dust'+str(index)+'dens'+str(par.on)+'.dat'
            
            # read dust mass volume density for each dust fluid in code units
            dustcube[ibin,:,:,:] = Field(field=fileread, directory=par.dir).data
        
            # conversion in g/cm^3
            dustcube[ibin,:,:,:] *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**3.)  # dimensions: nbin, ncol, nrad, nsec

            # decrease dust mass volume density inside mask radius
            # NB: mask_radius is in arcseconds
            rmask_in_code_units = par.mask_radius*par.distance*par.au/par.gas.culength/1e2
            for i in range(par.gas.nrad):
                if (par.gas.rmed[i] < rmask_in_code_units):
                    dustcube[ibin,:,i,:] = 0.0 # *= ( (par.gas.rmed[i]/rmask_in_code_units)**(10.0) ) 
            
        print('--------- computing dust mass volume density ----------')
        DUSTOUT = open('dust_density.binp','wb')        # binary format
        # requested header
        # hdr[0] = format number
        # hdr[1] = data precision (8 means double)
        # hdr[2] = nb of grid cells
        # hdr[3] = nb of dust bins
        hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol, par.nbin], dtype=int)
        hdr.tofile(DUSTOUT)
        
        # array (ncol, nbin, nrad, nsec)
        # CB: valentin's smart way!
        rhodustcube = dustcube.transpose(1,0,2,3)  

    # ==========================
    # 2D simulation with dusty fargo adsg or fargo3d
    # ==========================
    else:   

        # extra useful quantities (code units)
        Rinf = par.gas.redge[0:len(par.gas.redge)-1]
        Rsup = par.gas.redge[1:len(par.gas.redge)]
        surface  = np.zeros(par.gas.data.shape) # 2D array containing surface of each grid cell
        surf = np.pi * (Rsup*Rsup - Rinf*Rinf) / par.gas.nsec # surface of each grid cell (code units)
        for th in range(par.gas.nsec):
            surface[:,th] = surf
        
        # Mass of gas in units of the star's mass
        Mgas = np.sum(par.gas.data*surface)

        # since hydro simulation is 2D here, we first need to compute the dust's
        # surface density before getting its mass volume density
        print('--------- computing dust mass surface density ----------')
        dustcube = compute_dust_mass_surface_density()
            
        print('--------- computing dust mass volume density ----------')
        DUSTOUT = open('dust_density.binp','wb')        # binary format
        # requested header
        # hdr[0] = format number
        # hdr[1] = data precision (8 means double)
        # hdr[2] = nb of grid cells
        # hdr[3] = nb of dust bins
        hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol, par.nbin], dtype=int)
        hdr.tofile(DUSTOUT)
        
        # array (ncol, nbin, nrad, nsec)
        rhodustcube = np.zeros((par.gas.ncol,par.nbin,par.gas.nrad,par.gas.nsec))

        # dust aspect ratio as function of ibin and r (or actually, R, cylindrical radius)
        hd = np.zeros((par.nbin,par.gas.nrad))

        # work out averaged Stokes number per size bin with fargo3d
        # Epstein regime assumed so far in the code -> St ~ pi/2 (s x rho_int) / Sigma_gas
        # where Sigma_gas should denote the azimuthally-averaged gas surface density here
        # so that St is understood as a 2D array (nbin, nrad)
        if par.fargo3d == 'Yes':
            Stokes_fargo3d = np.zeros((par.nbin,par.gas.nrad))
            axirhogas = np.sum(par.gas.data,axis=1)/par.gas.nsec  # in code units
            axirhogas *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # in g/cm^2
            
            for ibin in range(par.nbin):
                if par.dustfluids != 'No':
                    mysize = par.dust_size[par.dustfluids[0]-1+ibin]
                else:
                    mysize = par.bins[ibin]
                Stokes_fargo3d[ibin,:] = 0.5*np.pi*(mysize*1e2)*par.dust_internal_density/axirhogas  # since dust size is in meters...
                #print('max Stokes number = ', Stokes_fargo3d[ibin,:].max())
                
        # For the vertical expansion of the dust mass volume density, we
        # need to define a 2D array for the dust's aspect ratio:
        for ibin in range(par.nbin):
            if par.fargo3d == 'No':
                St = avgstokes[ibin]    # avg stokes number for that bin
            if par.fargo3d == 'Yes' and par.dustfluids != 'No':
                St = Stokes_fargo3d[ibin]

            # gas aspect ratio (par.gas.rmed[i] = R in code units)
            hgas = par.aspectratio * (par.gas.rmed)**(par.flaringindex)
            
            # vertical extension depends on grain Stokes number:
            # T = theoretical: hd/hgas = sqrt(alpha/(St+alpha))
            # T2 = theoretical: hd/hgas = sqrt(Dz/(St+Dz)) with Dz = 10xalpha here is the coefficient for
            # vertical diffusion at midplane, which can differ from alpha
            # F = extrapolation from the simulations by Fromang & Nelson 09
            # G = Gaussian = same as gas (case of well-coupled dust for polarized intensity images)
            if par.z_expansion == 'F':
                hd[ibin,:] = 0.7 * hgas * ((St+1./St)/1000.)**(0.2)     
            if par.z_expansion == 'T':
                hd[ibin,:] = hgas * np.sqrt(par.alphaviscosity/(par.alphaviscosity+St))
            if par.z_expansion == 'T2':
                hd[ibin,:] = hgas * np.sqrt(10.0*par.alphaviscosity/(10.0*par.alphaviscosity+St))
            if par.z_expansion == 'G':
                hd[ibin,:] = hgas
        
        # dust aspect ratio as function of ibin, r and phi (2D array for each size bin)
        hd2D = np.zeros((par.nbin,par.gas.nrad,par.gas.nsec))
        for th in range(par.gas.nsec):
            hd2D[:,:,th] = hd    # nbin, nrad, nsec

        # grid radius function of ibin, r and phi (2D array for each size bin)
        r2D = np.zeros((par.nbin,par.gas.nrad,par.gas.nsec))
        for ibin in range(par.nbin):
            for th in range(par.gas.nsec):
                r2D[ibin,:,th] = par.gas.rmed

        # work out exponential and normalization factors exp(-z^2 / 2H_d^2)
        # with z = r cos(theta) and H_d = h_d x R = h_d x r sin(theta)
        # r = spherical radius, R = cylindrical radius
        for j in range(par.gas.ncol):
            rhodustcube[j,:,:,:] = dustcube * np.exp( -0.5*(np.cos(par.gas.tmed[j]) / hd2D)**2.0 )      # ncol, nbin, nrad, nsec
            rhodustcube[j,:,:,:] /= ( np.sqrt(2.*np.pi) * r2D * hd2D  * par.gas.culength*1e2 )          # quantity is now in g / cm^3

        # Renormalize dust's mass volume density such that the sum over the 3D grid's volume of
        # the dust's mass volume density x the volume of each grid cell does give us the right
        # total dust mass, which equals ratio x Mgas. Do that every time except fargo3D simulations
        # carried out in 2D used dust fluids
        if par.dustfluids == 'No':
            
            rhofield = np.sum(rhodustcube, axis=1)  # sum over dust bins
            Redge,Cedge,Aedge = np.meshgrid(par.gas.redge, par.gas.tedge, par.gas.pedge)   # ncol+1, nrad+1, Nsec+1
            r2 = Redge*Redge
            jacob  = r2[:-1,:-1,:-1] * np.sin(Cedge[:-1,:-1,:-1])
            dphi   = Aedge[:-1,:-1,1:] - Aedge[:-1,:-1,:-1]     # same as 2pi/nsec
            dr     = Redge[:-1,1:,:-1] - Redge[:-1,:-1,:-1]     # same as Rsup-Rinf
            dtheta = Cedge[1:,:-1,:-1] - Cedge[:-1,:-1,:-1]

            # volume of a cell in cm^3
            vol = jacob * dr * dphi * dtheta * ((par.gas.culength*1e2)**3)       # ncol, nrad, Nsec
            total_mass = np.sum(rhofield*vol)
            normalization_factor =  par.ratio * Mgas * (par.gas.cumass*1e3) / total_mass
            rhodustcube = rhodustcube*normalization_factor
            if par.verbose == 'Yes':
                print('total dust mass after vertical expansion [g] = ', np.sum(np.sum(rhodustcube, axis=1)*vol), ' as normalization factor = ', normalization_factor)


    # =======================
    # Simple dust sublimation model in case dust temperature set to
    # that of the hydro simulation:
    if par.dustsublimation == 'Yes' and par.Tdust_eq_Thydro == 'Yes':
        buf = np.fromfile('dust_temperature.bdat', dtype='float64')
        buf = buf[4:]
        buf = buf.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

        # Let's reswap axes! -> ncol, nbin, nrad, nsec
        buf = np.swapaxes(buf, 0, 1)  # nsec nbin ncol nrad
        buf = np.swapaxes(buf, 0, 2)  # ncol nbin nsec nrad
        buf = np.swapaxes(buf, 2, 3)  # ncol nbin nrad nsec
        gas_temp = buf[:,0,:,:]       # ncol nrad nsec

        axitemp = np.sum(gas_temp,axis=2)/par.gas.nsec
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                # if azimuthally-averaged dust temperature exceeds
                # 1500K, then drop azimuthally-averaged dust density
                # by 5 orders of magnitude:                
                if axitemp[j,i] > 1500.0:
                    rhodustcube[j,:,i,:] *= 1e-5

        del buf, gas_temp, axitemp
                    
    # print max of dust's mass volume density at each colatitude
    if par.verbose == 'Yes':
        for j in range(par.gas.ncol):
            print('max(rho_dustcube) [g cm-3] for colatitude index j = ', j, ' = ', rhodustcube[j,:,:,:].max())
                        
    # If writing data in an ascii file the ordering should be: nbin, nsec, ncol, nrad.
    # We therefore need to swap axes of array rhodustcube
    # before dumping it in a binary file! just like mastermind game!
    rhodustcube = np.swapaxes(rhodustcube, 2, 3)  # ncol nbin nsec nrad
    rhodustcube = np.swapaxes(rhodustcube, 0, 1)  # nbin ncol nsec nrad
    rhodustcube = np.swapaxes(rhodustcube, 1, 2)  # nbin nsec ncol nrad
    rhodustcube.tofile(DUSTOUT)
    DUSTOUT.close()

    # free RAM memory
    if par.hydro2D == 'Yes':
        del hd2D, r2D
    del rhodustcube, dustcube
    

# =========================
# compute again dust density in case dust sublimation is taken into
# account
# =========================
def recompute_dust_mass_volume_density():

    # Start by reading dust temperature. Note that the .bdat file can
    # be produced by a thermal MC run with RADMC-3D after having
    # already computed the dust density!
    Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
    Temp = Temp[4:]
    Temp = Temp.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

    # Then read again dust_density.binp file
    dens = np.fromfile('dust_density.binp', dtype='float64')
    rhodustcube = dens[4:]
    rhodustcube = rhodustcube.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

    axitemp = np.sum(Temp,axis=1)/par.gas.nsec  # nbin ncol nrad
    for ibin in range(par.nbin):
        print('sublimation: dust species in bin', ibin, 'out of ',par.nbin-1)
        for i in range(par.gas.nrad):
            for j in range(par.gas.ncol):
                # if azimuthally-averaged dust temperature exceeds
                # 1500K, then drop azimuthally-averaged dust density
                # by 5 orders of magnitude:                
                if axitemp[ibin,j,i] > 1500.0:
                    rhodustcube[ibin,:,j,i] *= 1e-5  

    DUSTOUT = open('dust_density.binp','wb')        # binary format
    hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol, par.nbin], dtype=int)
    hdr.tofile(DUSTOUT)
    rhodustcube.tofile(DUSTOUT)
    DUSTOUT.close()

    del dens,rhodustcube,Temp
    
        
# =========================
# Display dust density on RADMC 3D grid
# =========================
def plot_dust_density(mystring):

    dens = np.fromfile('dust_density.binp', dtype='float64')
    rhodustcube = dens[4:]
    rhodustcube = rhodustcube.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

    # Let's reswap axes! -> ncol, nbin, nrad, nsec
    rhodustcube = np.swapaxes(rhodustcube, 0, 1)  # nsec nbin ncol nrad
    rhodustcube = np.swapaxes(rhodustcube, 0, 2)  # ncol nbin nsec nrad
    rhodustcube = np.swapaxes(rhodustcube, 2, 3)  # ncol nbin nrad nsec
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
        
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial')
    fontcolor='white'
    
    # plot azimuthally-averaged dust density vs. radius and colatitude for smallest and largest bin sizes

    # azimuthally-averaged dust density:
    axidens_smallest = np.sum(rhodustcube[:,0,:,:],axis=2)/par.gas.nsec  # (nol,nrad)
    axidens_largest  = np.sum(rhodustcube[:,par.nbin-1,:,:],axis=2)/par.gas.nsec  # (nol,nrad)

    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.tedge)
    R = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
    Z = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au

    # midplane dust mass volume density:
    midplane_dens_smallest = rhodustcube[par.gas.ncol//2-1,0,:,:]           # (nrad,nsec)
    midplane_dens_largest  = rhodustcube[par.gas.ncol//2-1,par.nbin-1,:,:]  # (nrad,nsec)

    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.pedge)
    X = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au
    Y = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au

        
    print('--------- a) plotting azimuthally-averaged dust density (R,z) ----------')

    # --- smallest bin size ---
    fig = plt.figure(figsize=(8.,8.))
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)
    
    ax.set_xlabel('Radius [au]')
    ax.set_ylabel('Altitude [au]')
    ax.set_ylim(Z.min(),Z.max())
    ax.set_xlim(R.min(),R.max())

    if axidens_smallest.max()/axidens_smallest.min() > 1e3:
        mynorm = matplotlib.colors.LogNorm(vmin=1e-3*axidens_smallest.max(),vmax=axidens_smallest.max())
    else:
        mynorm = matplotlib.colors.LogNorm(vmin=axidens_smallest.min(),vmax=axidens_smallest.max())
    #mynorm = matplotlib.colors.LogNorm(vmin=axidens_smallest.min(),vmax=axidens_smallest.max())
        
    CF = ax.pcolormesh(R,Z,axidens_smallest,cmap='nipy_spectral',norm=mynorm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

    cax.xaxis.set_label_position('top')
    cax.set_xlabel('dust density '+r'[g cm$^{-3}$]')
    cax.xaxis.labelpad = 8

    if par.dustsublimation == 'No' or par.Tdust_eq_Thydro == 'Yes':
        fileout = 'dust_density_smallest_Rz.pdf'
    if par.dustsublimation == 'Yes' and par.Tdust_eq_Thydro == 'No':
        if 'before' in mystring:
            fileout = 'dust_density_smallest_Rz_before_subl.pdf'
        if 'after' in mystring:
            fileout = 'dust_density_smallest_Rz_after_subl.pdf'
    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig) 
        
    # --- repeat for largest bin size ---
    fig = plt.figure(figsize=(8.,8.))
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)

    ax.set_xlabel('Radius [au]')
    ax.set_ylabel('Altitude [au]')
    ax.set_ylim(Z.min(),Z.max())
    ax.set_xlim(R.min(),R.max())

    if axidens_largest.max()/axidens_largest.min() > 1e3:
        mynorm = matplotlib.colors.LogNorm(vmin=1e-3*axidens_largest.max(),vmax=axidens_largest.max())
    else:
        mynorm = matplotlib.colors.LogNorm(vmin=axidens_largest.min(),vmax=axidens_largest.max())
    #mynorm = matplotlib.colors.LogNorm(vmin=axidens_largest.min(),vmax=axidens_largest.max())
    
    CF = ax.pcolormesh(R,Z,axidens_largest,cmap='nipy_spectral',norm=mynorm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

    cax.xaxis.set_label_position('top')
    cax.set_xlabel('dust density '+r'[g cm$^{-3}$]')
    cax.xaxis.labelpad = 8

    if par.dustsublimation == 'No' or par.Tdust_eq_Thydro == 'Yes':
        fileout = 'dust_density_largest_Rz.pdf'
    if par.dustsublimation == 'Yes' and par.Tdust_eq_Thydro == 'No':
        if 'before' in mystring:
            fileout = 'dust_density_largest_Rz_before_subl.pdf'
        if 'after' in mystring:
            fileout = 'dust_density_largest_Rz_after_subl.pdf'

    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig) 
    
    print('--------- b) plotting dust density (x,y) ----------')

    # --- smallest bin size ---
    fig = plt.figure(figsize=(8.,8.))
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)
    
    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlim(X.min(),X.max())
    
    if midplane_dens_smallest.max()/midplane_dens_smallest.min() > 1e3:
        mynorm = matplotlib.colors.LogNorm(vmin=1e-3*midplane_dens_smallest.max(),vmax=midplane_dens_smallest.max())
    else:
        mynorm = matplotlib.colors.LogNorm(vmin=midplane_dens_smallest.min(),vmax=midplane_dens_smallest.max())

    midplane_dens_smallest = np.transpose(midplane_dens_smallest)
    CF = ax.pcolormesh(X,Y,midplane_dens_smallest,cmap='nipy_spectral',norm=mynorm,rasterized=True)
    #CF = ax.pcolormesh(X,Y,midplane_dens_smallest,cmap='nipy_spectral',norm=mynorm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    
    cax.xaxis.set_label_position('top')
    cax.set_xlabel('midplane dust density '+r'[g cm$^{-3}$]')
    cax.xaxis.labelpad = 8

    if par.dustsublimation == 'No' or par.Tdust_eq_Thydro == 'Yes':
        fileout = 'dust_density_smallest_midplane.pdf'
    if par.dustsublimation == 'Yes' and par.Tdust_eq_Thydro == 'No':
        if 'before' in mystring:
            fileout = 'dust_density_smallest_midplane_before_subl.pdf'
        if 'after' in mystring:
            fileout = 'dust_density_smallest_midplane_after_subl.pdf'

    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig)

    # --- repeat for largest bin size ---
    fig = plt.figure(figsize=(8.,8.))
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)

    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlim(X.min(),X.max())
    
    if midplane_dens_largest.max()/midplane_dens_largest.min() > 1e3:
        mynorm = matplotlib.colors.LogNorm(vmin=1e-3*midplane_dens_largest.max(),vmax=midplane_dens_largest.max())
    else:
        mynorm = matplotlib.colors.LogNorm(vmin=midplane_dens_largest.min(),vmax=midplane_dens_largest.max())
        
    midplane_dens_largest = np.transpose(midplane_dens_largest)
    CF = ax.pcolormesh(X,Y,midplane_dens_largest,cmap='nipy_spectral',norm=mynorm,rasterized=True)
    #CF = ax.pcolormesh(X,Y,midplane_dens_largest,cmap='nipy_spectral',norm=mynorm)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    
    cax.xaxis.set_label_position('top')
    cax.set_xlabel('midplane dust density '+r'[g cm$^{-3}$]')
    cax.xaxis.labelpad = 8

    if par.dustsublimation == 'No' or par.Tdust_eq_Thydro == 'Yes':
        fileout = 'dust_density_largest_midplane.pdf'
    if par.dustsublimation == 'Yes' and par.Tdust_eq_Thydro == 'No':
        if 'before' in mystring:
            fileout = 'dust_density_largest_midplane_before_subl.pdf'
        if 'after' in mystring:
            fileout = 'dust_density_largest_midplane_after_subl.pdf'

    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig)

        
    # free RAM memory
    del rhodustcube



# =========================
# Display dust-to-gas density ratio on RADMC 3D grid
# =========================
def plot_dust_to_gas_density():

    dens = np.fromfile('dust_density.binp', dtype='float64')
    rhodustcube = dens[4:]
    rhodustcube = rhodustcube.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

    # total dust density obtained by summing over dust bins
    total_dust_density = np.sum(rhodustcube,axis=0)  # nsec ncol nrad
    
    # now read number density of gas species
    file = 'numberdens_'+str(par.gasspecies)+'.binp'
    gas = np.fromfile(file, dtype='float64')      
    rhogascube = gas[3:]
    rhogascube = rhogascube*2.3*par.mp/par.abundance      # back to g / cm^3
    total_gas_density = rhogascube.reshape(par.gas.nsec,par.gas.ncol,par.gas.nrad) # nsec ncol nrad

    # define dust-to-gas density ratio
    dust_to_gas_density_ratio = total_dust_density/total_gas_density    # nsec ncol nrad
    midplane_dust_to_gas_density_ratio = dust_to_gas_density_ratio[:,par.gas.ncol//2-1,:] # nsec nrad  
    
    # X and Y arrays
    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.pedge)
    X = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au
    Y = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial')
    fontcolor='white'
    
    print('--------- plotting midplane dust-to-gas density ratio   ----------')

    fig = plt.figure(figsize=(8.,8.))
    plt.subplots_adjust(left=0.17, right=0.92, top=0.88, bottom=0.1)
    ax = plt.gca()
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.tick_params(axis='x', which='minor', top=True)
    ax.tick_params(axis='y', which='minor', right=True)
    
    ax.set_xlabel('x [au]')
    ax.set_ylabel('y [au]')
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlim(X.min(),X.max())

    # Min and max currently being hardcoded...
    mymin = 1e-2*midplane_dust_to_gas_density_ratio.max()   # 0.01
    mymax = midplane_dust_to_gas_density_ratio.max()        # 10.0
    mynorm = matplotlib.colors.LogNorm(vmin=mymin,vmax=mymax)
    
    CF = ax.pcolormesh(X,Y,midplane_dust_to_gas_density_ratio,cmap='nipy_spectral',norm=mynorm,rasterized=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')

    cax.xaxis.set_label_position('top')
    cax.set_xlabel('midplane dust-to-gas density ratio')
    cax.xaxis.labelpad = 8
        
    fileout = 'midplane_dg_ratio.pdf'
    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig)  # close figure as we reopen figure at every output number

    del rhodustcube
