import numpy as np
#import psutil
import sys
import subprocess

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
    Rinf = par.gas.redge[0:len(par.gas.redge)-1]
    Rsup = par.gas.redge[1:len(par.gas.redge)]
    surface  = np.zeros(par.gas.data.shape) # 2D array containing surface of each grid cell
    surf = np.pi * (Rsup*Rsup - Rinf*Rinf) / par.gas.nsec # surface of each grid cell (code units)
    for th in range(par.gas.nsec):
        surface[:,th] = surf

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

    # -------------------------
    # a) Case with no polarized scattering: we infer the dust's surface
    # density from the results of the gas+dust hydrodynamical simulation
    # -------------------------

    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    # CASE 1: FARGO2D simulation (Lagrangian particles)
    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    if (par.polarized_scat == 'No' and par.fargo3d == 'No'):
        
        # read information on the dust particles
        (rad, azi, vr, vt, Stokes, a) = np.loadtxt(par.dir+'/dustsystat'+str(par.on)+'.dat', unpack=True)

        # ------------------
        # Populate dust bins
        # ------------------
        for m in range (len(a)):   # CB: sum over dust particles
            
            r = rad[m]
            # radial index of the cell where the particle is
            if par.radialspacing == 'L':
                i = int(np.log(r/par.gas.redge.min())/np.log(par.gas.redge.max()/par.gas.redge.min()) * par.gas.nrad)
            else:
                i = np.argmin(np.abs(par.gas.redge-r))
            if (i < 0 or i >= par.gas.nrad):
                sys.exit('pb with i = ', i, ' in compute_dust_mass_surface_density step: I must exit!')
            
            t = azi[m]
            # azimuthal index of the cell where the particle is
            # (general expression since grid spacing in azimuth is always arithmetic)
            j = int((t-par.gas.pedge.min())/(par.gas.pedge.max()-par.gas.pedge.min()) * par.gas.nsec)
            if (j < 0 or j >= par.gas.nsec):
                sys.exit('pb with j = ', j, ' in compute_dust_mass_surface_density step: I must exit!')
                
            # particle size
            pcsize = a[m]
            
            # find out which bin particle belongs to
            ibin = int(np.log(pcsize/par.bins.min())/np.log(par.bins.max()/par.bins.min()) * par.nbin)
            if (ibin >= 0 and ibin < par.nbin):
                k = ibin*par.gas.nsec*par.gas.nrad + j*par.gas.nrad + i
                dust[k] +=1
                nparticles[ibin] += 1
                avgstokes[ibin] += Stokes[m]
                
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
            frac[0] *= 5e3
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
    # CASE 2: FARGO3D simulation in 2D (dust fluids)
    # -  -  -  -  -  -  -  -  -  -  -  -  -  - 
    if (par.polarized_scat == 'No' and par.fargo3d == 'Yes'):

        dustcube = dust.reshape(par.nbin, par.gas.nsec, par.gas.nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
        del dust

        # ------------------
        # dust surface density in each size bin directly from the simulation
        # outputs
        # ------------------
        for ibin in range(len(dust_id)):

            fileread = 'dust'+str(int(dust_id[ibin]))+'dens'+str(par.on)+'.dat'

            # read dust surface density for each dust fluid in code units
            dustcube[ibin,:,:]  = Field(field=fileread, directory=par.dir).data
        
            # conversion in g/cm^2
            dustcube[ibin,:,:] *= (par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec

            # decrease dust surface density inside mask radius
            # NB: mask_radius is in arcseconds
            rmask_in_code_units = par.mask_radius*par.distance*par.au/par.gas.culength/1e2
            for i in range(par.gas.nrad):
                if (par.gas.rmed[i] < par.rmask_in_code_units):
                    dustcube[ibin,i,:] = 0.0 # *= ( (par.gas.rmed[i]/rmask_in_code_units)**(10.0) ) 

        if par.verbose == 'Yes':
            print('Total dust mass [g] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.))
            print('Total dust mass [Mgas] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(Mgas*par.gas.cumass*1e3))
            print('Total dust mass [Mstar] = ', np.sum(dustcube[:,:,:]*surface*(par.gas.culength*1e2)**2.)/(par.gas.cumass*1e3))
    
        # Total dust surface density
        dust_surface_density = np.sum(dustcube,axis=0)
        if par.verbose == 'Yes':
            print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())


    # -------------------------
    # b) Case with polarized scattering: we say that the dust is perfectly
    # coupled to the gas. Same procedure whether simulation was carried out with
    # FARGO2D or FARGO3D (in 2D).
    # -------------------------
    if par.polarized_scat == 'Yes':

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
                for i in range(par.gas.nrad):
                    for j in range(par.gas.nsec):
                        # local Stokes number at R,phi
                        st = 0.5*np.pi*agraincm*rhopart/(par.gas.data[i,j]*par.gas.cumass*1e3/(par.gas.culength*1e2)**2.)
                        if st > 1e-4:  # 1e-4 or 1e-3 looks like a good threshold
                            dustcube[ibin,i,j] = 0.0

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

    # extra useful quantities (code units)
    Rinf = par.gas.redge[0:len(par.gas.redge)-1]
    Rsup = par.gas.redge[1:len(par.gas.redge)]
    surface  = np.zeros(par.gas.data.shape) # 2D array containing surface of each grid cell
    surf = np.pi * (Rsup*Rsup - Rinf*Rinf) / par.gas.nsec # surface of each grid cell (code units)
    for th in range(par.gas.nsec):
        surface[:,th] = surf
    
    # Mass of gas in units of the star's mass
    Mgas = np.sum(par.gas.data*surface)

    # if hydro simulation is 2D, we first need to compute the dust's
    # surface density before getting its mass volume density
    if par.hydro2D == 'Yes':
        dustcube = compute_dust_mass_surface_density()
        print('--------- computing dust mass surface density ----------')
        
    print('--------- computing dust mass volume density ----------')
    DUSTOUT = open('dust_density.inp','w')
    DUSTOUT.write('1 \n')                           # iformat  
    DUSTOUT.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells
    DUSTOUT.write(str(int(par.nbin))+' \n')             # nbin size bins 
    
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
            Stokes_fargo3d[ibin,:] = 0.5*np.pi*(dust_size[ibin]*1e2)*dust_internal_density/axirhogas  # since dust size is in meters...

    # For the vertical expansion of the dust mass volume density, we
    # need to define a 2D array for the dust's aspect ratio:
    for ibin in range(par.nbin):
        if par.polarized_scat == 'No':
            if par.fargo3d == 'No':
                St = avgstokes[ibin]                              # avg stokes number for that bin
            else:
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
    # total dust mass, which equals ratio x Mgas.
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
    
    # finally write mass volume densities for all size bins
    for ibin in range(par.nbin):
        print('dust species in bin', ibin, 'out of ',par.nbin-1)
        for k in range(par.gas.nsec):
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    DUSTOUT.write(str(rhodustcube[j,ibin,i,k])+' \n')

    # print max of dust's mass volume density at each colatitude
    if par.verbose == 'Yes':
        for j in range(par.gas.ncol):
            print('max(rho_dustcube) [g cm-3] for colatitude index j = ', j, ' = ', rhodustcube[j,:,:,:].max())

    DUSTOUT.close()

    # free RAM memory
    del rhodustcube, dustcube, hd2D, r2D
