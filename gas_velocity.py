# import global variables
import par

import numpy as np

from mesh import *
from field import *

import matplotlib
import matplotlib.pyplot as plt

# =========================
# Microturbulent line broadening
# =========================
def write_gas_microturb():

    MTURB = open('microturbulence.binp','wb')
    # requested header
    # hdr[0] = format number
    # hdr[1] = data precision (8 means double)
    # hdr[2] = nb of grid cells
    hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol], dtype=int)
    hdr.tofile(MTURB)

    # Default case: uniform microturbulence set by 'turbvel' parameter in params.dat
    microturb = np.ones((par.gas.ncol,par.gas.nrad,par.gas.nsec))*par.turbvel*1.0e2  # ncol, nrad, nsec in cm/s

    if par.turbvel == 'cavity':  # Baruteau et al. 2021 model
        for k in range(par.gas.nsec):
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    if par.gas.rmed[i] < 1.0:
                        myalpha = 3e-2   # inside cavity
                    else:
                        myalpha = 3e-4   # outside cavity
                    # v_turb ~ sqrt(alpha) x isothermal sound speed
                    microturb[j,i,k] = np.sqrt(myalpha * par.kB * gas_temp[j,i,k] / 2.3 / par.mp)  
        if par.verbose:
            print('min and max of microturbulent velocity in cm/s = ',microturb.min(),microturb.max())

    # If writing data in an ascii file the ordering should be: nsec, ncol, nrad.
    # We therefore need to swap axes of array microturb
    # before dumping it in a binary file! just like mastermind game!
    microturb = np.swapaxes(microturb, 0, 1)  # nrad ncol nsec
    microturb = np.swapaxes(microturb, 0, 2)  # nsec ncol nrad
    microturb.tofile(MTURB)
    MTURB.close()


# =========================
# Compute gas velocity field on RADMC 3D grid
# =========================
def compute_gas_velocity():

    # 3D simulation carried out with Fargo 3D (spherical coordinates already)
    if par.fargo3d == 'Yes' and par.hydro2D == 'No':

        vtheta3D  = Field(field='gasvz'+str(par.on)+'.dat', directory=par.dir).data  # code units
        vtheta3D *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s
        
        vrad3D    = Field(field='gasvy'+str(par.on)+'.dat', directory=par.dir).data  # code units
        vrad3D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        vphi3D    = Field(field='gasvx'+str(par.on)+'.dat', directory=par.dir).data  # code units

        f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        omegaframe = omega[par.on]
        
        for theta in range(par.gas.ncol):
            for phi in range(par.gas.nsec):
                vphi3D[theta,:,phi] += par.gas.rmed*omegaframe
        vphi3D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

    else:
        # arrays allocation
        vrad3D_cyl   = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D_cyl   = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))   # zeros!
        vtheta3D     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!
        vrad3D       = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!
        vphi3D       = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))   # zeros!

        if par.fargo3d == 'No':
            vrad2D   = Field(field='gasvrad'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D   = Field(field='gasvtheta'+str(par.on)+'.dat', directory=par.dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        else:
            vrad2D    = Field(field='gasvy'+str(par.on)+'.dat', directory=par.dir).data  # code units
            vphi2D    = Field(field='gasvx'+str(par.on)+'.dat', directory=par.dir).data  # code units
            f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
            
        vrad2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        omegaframe = omega[par.on]
        print('omegaframe = ', omegaframe)
        
        for phi in range(par.gas.nsec):
            vphi2D[:,phi] += par.gas.rmed*omegaframe
        vphi2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        # Make gas velocity axisymmetric (testing purposes)
        if ('axisymgas' in open('params.dat').read()) and (par.axisymgas == 'Yes'):
            axivrad2D = np.sum(vrad2D,axis=1)/par.gas.nsec
            axivphi2D = np.sum(vphi2D,axis=1)/par.gas.nsec  
            for i in range(par.gas.nrad):
                vrad2D[i,:] = axivrad2D[i]
                vphi2D[i,:] = axivphi2D[i]
            
        # Vertical expansion for vrad and vphi (vtheta being assumed zero)
        for z in range(par.gas.nver):
            vrad3D_cyl[z,:,:] = vrad2D
            vphi3D_cyl[z,:,:] = vphi2D

        # Now, sweep through the spherical grid
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                R = par.gas.rmed[i]*np.sin(par.gas.tmed[j])  # cylindrical radius
                z = par.gas.rmed[i]*np.cos(par.gas.tmed[j])  # vertical altitude
                icyl = np.argmin(np.abs(par.gas.rmed-R))
                if R < par.gas.rmed[icyl] and icyl > 0:
                    icyl-=1
                jcyl = np.argmin(np.abs(par.gas.zmed-z))
                if z < par.gas.zmed[jcyl] and jcyl > 0:
                    jcyl-=1
                if (icyl < par.gas.nrad-1 and jcyl < par.gas.nver-1):
                    vrad3D[j,i,:] = ( vrad3D_cyl[jcyl,icyl,:]*(par.gas.rmed[icyl+1]-R)*(par.gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1,icyl,:]*(par.gas.rmed[icyl+1]-R)*(z-par.gas.zmed[jcyl]) + vrad3D_cyl[jcyl,icyl+1,:]*(R-par.gas.rmed[icyl])*(par.gas.zmed[jcyl+1]-z) + vrad3D_cyl[jcyl+1,icyl+1,:]*(R-par.gas.rmed[icyl])*(z-par.gas.zmed[jcyl]) ) / ( (par.gas.rmed[icyl+1]-par.gas.rmed[icyl]) * (par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]) )
                    vphi3D[j,i,:] = ( vphi3D_cyl[jcyl,icyl,:]*(par.gas.rmed[icyl+1]-R)*(par.gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1,icyl,:]*(par.gas.rmed[icyl+1]-R)*(z-par.gas.zmed[jcyl]) + vphi3D_cyl[jcyl,icyl+1,:]*(R-par.gas.rmed[icyl])*(par.gas.zmed[jcyl+1]-z) + vphi3D_cyl[jcyl+1,icyl+1,:]*(R-par.gas.rmed[icyl])*(z-par.gas.zmed[jcyl]) ) / ( (par.gas.rmed[icyl+1]-par.gas.rmed[icyl]) * (par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]) )
                else:
                # simple nearest-grid point interpolation...
                    vrad3D[j,i,:] = vrad3D_cyl[jcyl,icyl,:]   
                    vphi3D[j,i,:] = vphi3D_cyl[jcyl,icyl,:] 

                    
    print('--------- writing gas_velocity.inp file ----------')

    # Define gas velocity array that contains all three components
    gasvel = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec,3))
    gasvel[:,:,:,0] = vrad3D
    gasvel[:,:,:,1] = vtheta3D
    gasvel[:,:,:,2] = vphi3D

    VELOUT = open('gas_velocity.binp','wb')
    # requested header
    # hdr[0] = format number
    # hdr[1] = data precision (8 means double)
    # hdr[2] = nb of grid cells
    hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol], dtype=int)
    hdr.tofile(VELOUT)
    
    # If writing data in an ascii file the ordering should be: nsec, ncol, nrad
    # We therefore need to swap axes of array gasvel
    # before dumping it in a binary file! just like mastermind game!
    gasvel = np.swapaxes(gasvel, 0, 1)  # nrad ncol nsec
    gasvel = np.swapaxes(gasvel, 0, 2)  # nsec ncol nrad
    gasvel.tofile(VELOUT)
    VELOUT.close()
    
    '''
    VELOUT = open('gas_velocity.inp','w')
    VELOUT.write('1 \n')                           # iformat
    VELOUT.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells
    for k in range(par.gas.nsec):
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                GASVEL.write(str(vrad3D[j,i,k])+' '+str(vtheta3D[j,i,k])+' '+str(vphi3D[j,i,k])+' \n')
    VELOUT.close()
    '''

    # finally output plots of the gas temperature
    if par.plot_gas_quantities == 'Yes':
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.ticker as ticker
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
        
        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rc('font', family='Arial')
        fontcolor='white'

        # midplane radial velocity:
        vrmid = vrad3D[par.gas.ncol//2-1,:,:] # nrad, nsec
        vrmid /= 1e5 # in km/s

        # midplane azimuthal velocity:
        vtmid = vphi3D[par.gas.ncol//2-1,:,:] # nrad, nsec
        vtmid /= 1e5 # in km/s
            
        radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.pedge)
        X = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au
        Y = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
        
        # common plot features
        if par.gasspecies == 'co':
            strgas = r'$^{12}$CO'
        elif par.gasspecies == '13co':
            strgas = r'$^{13}$CO'
        elif par.gasspecies == 'c17o':
            strgas = r'C$^{17}$O'
        elif par.gasspecies == 'c18o':
            strgas = r'C$^{18}$O'
        else:
            strgas = str(par.gasspecies).upper()  # capital letters
            
        print('--------- a) plotting midplane radial velocity (x,y) ----------')

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
            
        mynorm = matplotlib.colors.Normalize(vmin=vrmid.min(),vmax=vrmid.max())
        vrmid = np.transpose(vrmid)
        CF = ax.pcolormesh(X,Y,vrmid,cmap='nipy_spectral',norm=mynorm,rasterized=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')

        cax.xaxis.set_label_position('top')
        cax.set_xlabel(strgas+' midplane radial velocity '+r'[km s$^{-1}$]')
        cax.xaxis.labelpad = 8
        
        fileout = 'vrad_midplane.pdf'
        plt.savefig('./'+fileout, dpi=160)
        plt.close(fig)  # close figure as we reopen figure at every output number


        print('--------- b) plotting midplane azimuthal velocity (x,y) ----------')

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
            
        mynorm = matplotlib.colors.Normalize(vmin=vtmid.min(),vmax=vtmid.max())
        vtmid = np.transpose(vtmid)
        CF = ax.pcolormesh(X,Y,vtmid,cmap='nipy_spectral',norm=mynorm,rasterized=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')

        cax.xaxis.set_label_position('top')
        cax.set_xlabel(strgas+' midplane azimuthal velocity '+r'[km s$^{-1}$]')
        cax.xaxis.labelpad = 8
        
        fileout = 'vphi_midplane.pdf'
        plt.savefig('./'+fileout, dpi=160)
        plt.close(fig)  # close figure as we reopen figure at every output number

        
    del vrad3D, vphi3D, vtheta3D, vrad3D_cyl, vphi3D_cyl
