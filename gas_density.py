# import global variables
import par

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

from mesh import *
from field import *


# =========================
# Compute gas mass volume density on RADMC 3D grid
# =========================
def compute_gas_mass_volume_density():

    # array allocation, right now this is assuming gas data is only 2D...
    gascube = par.gas.data*(par.gas.cumass*1e3)/((par.gas.culength*1e2)**2.)  # nrad, nsec, quantity is in g / cm^2
    
    # Artificially make a cavity devoid of gas (test)
    if ('cavity_gas' in open('params.dat').read()) and (par.cavity_gas == 'Yes'):
        imin = np.argmin(np.abs(par.gas.rmed-1.3))
        for i in range(par.gas.nrad):
            if i < imin:
                for j in range(par.gas.nsec):
                    gascube[i,j] *= ((par.gas.rmed[i]/par.gas.rmed[imin])**(6.0))
                    
    GASOUT = open('numberdens_%s.inp'%par.gasspecies,'w')
    GASOUT.write('1 \n')                                  # iformat
    GASOUT.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')   # n cells

    if par.lines_mode > 1:
        GASOUTH2 = open('numberdens_h2.inp','w')
        GASOUTH2.write('1 \n')                                 # iformat
        GASOUTH2.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')  # n cells

    # array (ncol, nrad, nsec)
    rhogascubeh2     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))      # H2
    rhogascubeh2_cyl = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))
    rhogascube       = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))      # gas species (eg, CO...)
    rhogascube_cyl   = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))

    # gas aspect ratio as function of r (or actually, R, cylindrical radius)
    hgas = par.aspectratio * (par.gas.rmed)**(par.flaringindex)
    hg2D = np.zeros((par.gas.nrad,par.gas.nsec))
    r2D  = np.zeros((par.gas.nrad,par.gas.nsec))
    for th in range(par.gas.nsec):
        hg2D[:,th] = hgas     # nrad, nsec
        r2D[:,th] = par.gas.rmed  # nrad, nsec

    # work out vertical expansion. First, for the array in cylindrical coordinates
    for j in range(par.gas.nver):
        rhogascubeh2_cyl[j,:,:] = gascube * np.exp( -0.5*(par.gas.zmed[j]/hg2D/r2D)**2.0 )  # nver, nrad, nsec
        rhogascubeh2_cyl[j,:,:] /= ( np.sqrt(2.*np.pi) * r2D * hg2D  * par.gas.culength*1e2 * 2.3*par.mp )   # quantity is now in cm^-3

    # multiply by constant abundance ratio
    rhogascube_cyl = rhogascubeh2_cyl*par.abundance

    # Simple model for photodissociation: drop
    # number density by 5 orders of magnitude if 1/2
    # erfc(z/sqrt(2)H) x Sigma_gas / mu m_p < 1e21 cm^-2
    if (par.photodissociation == 'Yes' or par.freezeout == 'Yes'):
        for k in range(par.gas.nsec):
            for j in range(par.gas.nver):
                for i in range(par.gas.nrad):
                    # all relevant quantities below are in in cgs
                    chip = 0.5 *  math.erfc(par.gas.zmed[j]/np.sqrt(2.0)/hgas[i]/par.gas.rmed[i]) * gascube[i,k] / (2.3*par.mp)
                    chim = 0.5 * math.erfc(-par.gas.zmed[j]/np.sqrt(2.0)/hgas[i]/par.gas.rmed[i]) * gascube[i,k] / (2.3*par.mp)
                    if (par.photodissociation == 'Yes' and (chip < 1e21 or chim < 1e21)):
                        rhogascube_cyl[j,i,k] *= 1e-5
                # Simple modelling of freezeout: drop CO number density
                    if par.freezeout == 'Yes' and gas_temp_cyl[j,i,k] < 19.0:
                        rhogascube_cyl[j,i,k] *= 1e-5

    # then, sweep through the spherical grid
    for j in range(par.gas.ncol):
        for i in range(par.gas.nrad):
            
            R = par.gas.rmed[i]*np.sin(par.gas.tmed[j])  # cylindrical radius
            icyl = np.argmin(np.abs(par.gas.rmed-R))
            if R < par.gas.rmed[icyl] and icyl > 0:
                icyl-=1
            
            z = par.gas.rmed[i]*np.cos(par.gas.tmed[j])  # vertical altitude            
            jcyl = np.argmin(np.abs(par.gas.zmed-z))
            if z < par.gas.zmed[jcyl] and jcyl > 0:
                jcyl-=1

            # bilinear interpolation
            if (icyl < par.gas.nrad-1 and jcyl < par.gas.nver-1 and icyl > 0):
                dr = par.gas.rmed[icyl+1]-par.gas.rmed[icyl]
                dz = par.gas.zmed[jcyl+1]-par.gas.zmed[jcyl]
                
                xij     = (par.gas.rmed[icyl+1]-R) * (par.gas.zmed[jcyl+1]-z) / (dr*dz)
                if xij < 0 or xij > 1:
                    print('beware that xij < 0 or xij > 1:',i,j,xij,par.gas.rmed[icyl+1]-R,dr,par.gas.zmed[jcyl+1]-z,dz)
                
                xijp1   = (par.gas.rmed[icyl+1]-R) * (z-par.gas.zmed[jcyl])   / (dr*dz)
                if xijp1 < 0 or xijp1 > 1:
                    print('beware that xijp1 < 0 or xijp1 > 1:',i,j,xijp1,par.gas.rmed[icyl+1]-R,dr,z-par.gas.zmed[jcyl],dz)

                xip1j   = (R-par.gas.rmed[icyl])   * (par.gas.zmed[jcyl+1]-z) / (dr*dz)
                if xip1j < 0 or xip1j > 1:
                    print('beware that xip1j < 0 or xip1j > 1:',i,j,xip1j,R-par.gas.rmed[icyl],dr,par.gas.zmed[jcyl+1]-z,dz)

                xip1jp1 = (R-par.gas.rmed[icyl])   * (z-par.gas.zmed[jcyl])   / (dr*dz)
                if xip1jp1 < 0 or xip1jp1 > 1:
                    print('beware that xip1jp1 < 0 or xip1jp1 > 1:',i,j,xip1jp1,R-par.gas.rmed[icyl],dr,z-par.gas.zmed[jcyl],dz)
 
                rhogascube[j,i,:] = rhogascube_cyl[jcyl,icyl,:]*xij +\
                  rhogascube_cyl[jcyl+1,icyl,:]*xijp1 +\
                  rhogascube_cyl[jcyl,icyl+1,:]*xip1j +\
                  rhogascube_cyl[jcyl+1,icyl+1,:]*xip1jp1

                rhogascubeh2[j,i,:] = rhogascubeh2_cyl[jcyl,icyl,:]*xij +\
                  rhogascubeh2_cyl[jcyl+1,icyl,:]*xijp1 +\
                  rhogascubeh2_cyl[jcyl,icyl+1,:]*xip1j +\
                  rhogascubeh2_cyl[jcyl+1,icyl+1,:]*xip1jp1
            
            else:
            # simple nearest-grid point interpolation...
                rhogascube[j,i,:] = rhogascube_cyl[jcyl,icyl,:]    
                rhogascubeh2[j,i,:] = rhogascubeh2_cyl[jcyl,icyl,:]


    print('--------- writing numberdens.inp file ----------')
    for k in range(par.gas.nsec):
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                GASOUT.write(str(rhogascube[j,i,k])+' \n')
                if par.lines_mode > 1:
                    GASOUTH2.write(str(rhogascubeh2[j,i,k])+' \n')
    GASOUT.close()
    if par.lines_mode > 1:
        GASOUTH2.close()

        
    # plot azimuthally-averaged density vs. radius and colatitude
    if par.plot_gas_quantities == 'Yes':
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.ticker as ticker
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
        
        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rc('font', family='Arial')
        fontcolor='white'

        print('--------- plotting density(R,theta) ----------')

        axidens   = np.zeros((par.gas.ncol,par.gas.nrad))
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                for k in range(par.gas.nsec):
                    axidens[j,i]   += rhogascube[j,i,k]
                axidens[j,i]   /= (par.gas.nsec+0.0)
                
        fig = plt.figure(figsize=(8.,8.))
        plt.subplots_adjust(left=0.14, right=0.94, top=0.88, bottom=0.11)
        ax = plt.gca()
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.tick_params(axis='x', which='minor', top=True)
        ax.tick_params(axis='y', which='minor', right=True)

        radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.tedge)
        X = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
        Y = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Altitude [au]')

        ax.set_ylim(Y.min(),Y.max())
        ax.set_xlim(X.min(),X.max())
        
        mynorm = matplotlib.colors.LogNorm()
        vmin = axidens.min()
        vmax = axidens.max()

        CF = ax.pcolormesh(X,Y,axidens,cmap=par.mycolormap,vmin=vmin,vmax=vmax,norm=mynorm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

        # title on top
        cax.xaxis.set_label_position('top')
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
            
        cax.set_xlabel(strgas+' number density '+r'[cm$^{-3}$]')
        cax.xaxis.labelpad = 8
        
        fileout = 'number_density.pdf'
        plt.savefig('./'+fileout, dpi=160)

        
    # print max of gas mass volume density at each colatitude
    if par.verbose == 'Yes':
        for j in range(par.gas.ncol):
            print('max(rho_dustcube) for gas species at colatitude index j = ', j, ' = ', rhogascube[j,:,:].max())

    # free RAM memory
    del rhogascube,rhogascubeh2,rhogascube_cyl,rhogascubeh2_cyl
    
