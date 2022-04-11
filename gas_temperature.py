# import global variables
import par

from mesh import *
from field import *

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt


# =========================
# Compute gas temperature on RADMC 3D grid
# =========================
def compute_gas_temperature():
    
    # Set dust / gas temperature if Tdust_eq_Thydro == 'Yes' (currently default case)
    if par.Tdust_eq_Thydro == 'Yes':

        if par.RTdust_or_gas == 'gas':
            print('--------- writing temperature.inp file (no mctherm) ----------')
            TEMPOUT = open('gas_temperature.inp','w')
            TEMPOUT.write('1 \n')                           # iformat
            TEMPOUT.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells

        if par.RTdust_or_gas == 'dust':
            print('--------- Writing temperature file (no mctherm) ----------')
            os.system('rm -f dust_temperature.bdat')        # avoid confusion!...
            TEMPOUT = open('dust_temperature.dat','w')       
            TEMPOUT.write('1 \n')                           # iformat
            TEMPOUT.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells
            TEMPOUT.write(str(int(par.nbin))+' \n')             # nbin size bins 

        gas_temp     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))
        gas_temp_cyl = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))

        thydro = par.aspectratio*par.aspectratio*par.gas.cutemp*par.gas.rmed**(-1.0+2.0*par.flaringindex)

        for k in range(par.gas.nsec):
            for j in range(par.gas.nver):
                gas_temp_cyl[j,:,k] = thydro  # only function of R (cylindrical radius)

        # Now, sweep through the spherical grid
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
                        print('beware that xij < 0 or xij > 1 in gas_temperature.py:',i,j,icyl,jcyl,xij,par.gas.rmed[icyl],R,par.gas.rmed[icyl+1],dr,par.gas.zmed[jcyl],z,par.gas.zmed[jcyl+1],dz)
                    
                    xijp1   = (par.gas.rmed[icyl+1]-R) * (z-par.gas.zmed[jcyl])   / (dr*dz)
                    if xijp1 < 0 or xijp1 > 1:
                        print('beware that xijp1 < 0 or xijp1 > 1 in gas_temperature.py:',i,j,icyl,jcyl,xijp1,par.gas.rmed[icyl+1]-R,dr,z-par.gas.zmed[jcyl],dz)
                
                    xip1j   = (R-par.gas.rmed[icyl])   * (par.gas.zmed[jcyl+1]-z) / (dr*dz)
                    if xip1j < 0 or xip1j > 1:
                        print('beware that xip1j < 0 or xip1j > 1 in gas_temperature.py:',i,j,icyl,jcyl,xip1j,R-par.gas.rmed[icyl],dr,par.gas.zmed[jcyl+1]-z,dz)

                    xip1jp1 = (R-par.gas.rmed[icyl])   * (z-par.gas.zmed[jcyl])   / (dr*dz)
                    if xip1jp1 < 0 or xip1jp1 > 1:
                        print('beware that xip1jp1 < 0 or xip1jp1 > 1 in gas_temperature.py:',i,j,icyl,jcyl,xip1jp1,R-par.gas.rmed[icyl],dr,z-par.gas.zmed[jcyl],dz)

                    gas_temp[j,i,:] = gas_temp_cyl[jcyl,icyl,:]*xij +\
                      gas_temp_cyl[jcyl+1,icyl,:]*xijp1 +\
                      gas_temp_cyl[jcyl,icyl+1,:]*xip1j +\
                      gas_temp_cyl[jcyl+1,icyl+1,:]*xip1jp1
                        
                else:
                # simple nearest-grid point interpolation...
                    gas_temp[j,i,:] = gas_temp_cyl[jcyl,icyl,:]   

        # Finally write temperature file
        if par.RTdust_or_gas == 'gas':
            for k in range(par.gas.nsec):
                for j in range(par.gas.ncol):
                    for i in range(par.gas.nrad):
                        TEMPOUT.write(str(gas_temp[j,i,k])+' \n')
            
        if par.RTdust_or_gas == 'dust':
            # write dust temperature for all size bins
            for ibin in range(par.nbin):
                print('writing temperature of dust species in bin', ibin, 'out of ',par.nbin-1)
                for k in range(par.gas.nsec):
                    for j in range(par.gas.ncol):
                        for i in range(par.gas.nrad):
                            TEMPOUT.write(str(gas_temp[j,i,k])+' \n')

        TEMPOUT.close()
        
        # finally output plots of the gas temperature
        if par.plot_gas_quantities == 'Yes' or par.plot_dust_quantities == 'Yes':
        
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import matplotlib.ticker as ticker
            from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
        
            matplotlib.rcParams.update({'font.size': 20})
            matplotlib.rc('font', family='Arial')
            fontcolor='white'

            # azimuthally-averaged gas temperature:
            axitemp = np.sum(gas_temp,axis=2)/par.gas.nsec
            
            radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.tedge)
            R = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
            Z = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au

            # surface gas temperature:
            surftemp = gas_temp[par.gas.ncol-1,:,:]
            
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

            
            print('--------- a) plotting azimuthally-averaged gas temperature (R,z) ----------')

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

            
            mynorm = matplotlib.colors.Normalize(vmin=axitemp.min(),vmax=axitemp.max())
            if axitemp.max() / axitemp.min() > 10:
                mynorm = matplotlib.colors.LogNorm(vmin=axitemp.min(),vmax=axitemp.max())

            CF = ax.pcolormesh(R,Z,axitemp,cmap='nipy_spectral',norm=mynorm,rasterized=True)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2.5%", pad=0.12)
            cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
            cax.xaxis.tick_top()
            cax.xaxis.set_tick_params(labelsize=20, direction='out')

            cax.xaxis.set_label_position('top')
            if par.RTdust_or_gas == 'gas':
                cax.set_xlabel(strgas+' temperature '+r'[K]')
                fileout = 'gas_temperature_Rz.pdf'
            if par.RTdust_or_gas == 'dust':
                cax.set_xlabel('dust temperature '+r'[K]')
                fileout = 'dust_temperature_Rz.pdf'
            cax.xaxis.labelpad = 8
            
            plt.savefig('./'+fileout, dpi=160)
            plt.close(fig)  # close figure as we reopen figure at every output number
            
            
            print('--------- b) plotting surface gas temperature (x,y) ----------')

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
            
            mynorm = matplotlib.colors.Normalize(vmin=surftemp.min(),vmax=surftemp.max())
            if surftemp.max() / surftemp.min() > 10:
                mynorm = matplotlib.colors.LogNorm(vmin=surftemp.min(),vmax=surftemp.max())
            
            surftemp = np.transpose(surftemp)
            CF = ax.pcolormesh(X,Y,surftemp,cmap='nipy_spectral',norm=mynorm,rasterized=True)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="2.5%", pad=0.12)
            cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
            cax.xaxis.tick_top()
            cax.xaxis.set_tick_params(labelsize=20, direction='out')

            cax.xaxis.set_label_position('top')
            if par.RTdust_or_gas == 'gas':
                cax.set_xlabel(strgas+' surface temperature '+r'[K]')
                fileout = 'gas_temperature_surface.pdf'
            if par.RTdust_or_gas == 'dust':
                cax.set_xlabel('dust surface temperature '+r'[K]')
                fileout = 'dust_temperature_surface.pdf'                
            cax.xaxis.labelpad = 8
            
            plt.savefig('./'+fileout, dpi=160)
            plt.close(fig)  # close figure as we reopen figure at every output number
            
        del gas_temp  
