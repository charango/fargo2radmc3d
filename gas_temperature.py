# import global variables
import par

from mesh import *
from field import *

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt


# =========================
# Compute temperature of hydrodynamical simulation on RADMC 3D grid
# called if (par.Tdust_eq_Thydro == 'Yes' or (par.Tdust_eq_Thydro ==
# 'No' and par.Tdust_eq_Tgas == 'No') ):
# =========================
def compute_hydro_temperature():    

    if ( (par.RTdust_or_gas == 'gas') or (par.RTdust_or_gas == 'both' and par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No') ):
        print('--------- writing gas_temperature.binp file (no mctherm) ----------')
        
        TEMPOUT = open('gas_temperature.binp','wb')
        # requested header
        # hdr[0] = format number
        # hdr[1] = data precision (8 means double)
        # hdr[2] = nb of grid cells
        hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol], dtype=int)
        hdr.tofile(TEMPOUT)

        TEMPOUTCYL = open('gas_tempcyl.binp','wb')
        hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.nver], dtype=int)
        hdr.tofile(TEMPOUTCYL)
            
    if ( (par.RTdust_or_gas == 'dust' or par.RTdust_or_gas == 'both') and par.Tdust_eq_Thydro == 'Yes'):        
        print('--------- Writing dust_temperature.bdat file (no mctherm) ----------')

        TEMPOUT = open('dust_temperature.bdat','wb')
        # requested header
        # hdr[0] = format number
        # hdr[1] = data precision (8 means double)
        # hdr[2] = nb of grid cells
        # hdr[3] = nb of dust bins
        hdr = np.array([1, 8, par.gas.nrad*par.gas.nsec*par.gas.ncol, par.nbin], dtype=int)
        hdr.tofile(TEMPOUT)

    # --------------------------
    # 3D simulation with fargo3d
    # --------------------------
    if par.hydro2D == 'No':

        # Allocate arrays
        gas_temp     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))

        if "ISOTHERMAL" in open(par.dir+'summary'+str(on)+'.dat',"r").read():
            c_s = Field(field='gasenergy'+str(par.on)+'.dat', directory=par.dir).data  # in code units; gasenergyX.dat countains the values of the sound speed
            thydro = c_s*c_s            # in code units, the sound velocity c_s equals c_s = sqrt(T)
        else:
            e = Field(field='gasenergy'+str(par.on)+'.dat', directory=par.dir).data    # in code units; gasenergyX.dat contains the thermal energy per unit volume
            rho = Field(field='gasdens'+str(par.on)+'.dat', directory=par.dir).data
            command = par.awk_command+' " /^GAMMA/ " '+par.dir+'*.par'
            if sys.version_info[0] < 3:
                buf = subprocess.check_output(command, shell=True)
            else:
                buf = subprocess.getoutput(command)
            gamma = float(buf.split()[1])    
            thydro = (gamma-1.0)*e/rho

        thydro *= par.gas.cutemp    # (ncol,nrad,nsec) in K
        gas_temp = thydro

    # --------------------------
    # 2D simulation with dusty fargo adsg or fargo3d
    # --------------------------
    else:
            
        gas_temp     = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))
        gas_temp_cyl = np.zeros((par.gas.nver,par.gas.nrad,par.gas.nsec))

        # Test if energy equation was used or not. For that, we just
        # need to check if a file TemperatureXX.dat was written in
        # Fargo run's directory:
        # Cuidadin: won't apply to FARGO-3D runs!
        input_file = par.dir+'/Temperature'+str(par.on)+'.dat'

        if os.path.isfile(input_file) == False:
            thydro = par.aspectratio*par.aspectratio*par.gas.cutemp*par.gas.rmed**(-1.0+2.0*par.flaringindex)
            for k in range(par.gas.nsec):
                for j in range(par.gas.nver):
                    gas_temp_cyl[j,:,k] = thydro  # only function of R (cylindrical radius)
        # case existing gas Temperature file:
        else:
            thydro = Field(field='Temperature'+str(par.on)+'.dat', directory=par.dir).data  # in code units
            thydro *= par.gas.cutemp  # (nrad,nsec) in K
            for j in range(par.gas.nver):
                gas_temp_cyl[j,:,:] = thydro    # function of R and phi

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
    if ( (par.RTdust_or_gas == 'gas') or (par.RTdust_or_gas == 'both' and par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No') ):
        # If writing data in an ascii file the ordering should be: nsec, ncol, nrad.
        # We therefore need to swap axes of array gas_temp
        # before dumping it in a binary file! just like mastermind game!
        gas_temp = np.swapaxes(gas_temp, 0, 1)  # nrad ncol nsec
        gas_temp = np.swapaxes(gas_temp, 0, 2)  # nsec ncol nrad
        gas_temp.tofile(TEMPOUT)
        TEMPOUT.close()
        if par.hydro2D == 'Yes':
            gas_temp_cyl.tofile(TEMPOUTCYL)         # nver nrad nsec
            TEMPOUTCYL.close()
            del gas_temp_cyl
        del gas_temp

    if ( (par.RTdust_or_gas == 'dust' or par.RTdust_or_gas == 'both') and par.Tdust_eq_Thydro == 'Yes'):
        # Define 4D dust temperature array
        dust_temp = np.zeros((par.gas.ncol,par.nbin,par.gas.nrad,par.gas.nsec))
        for ibin in range(par.nbin):
            dust_temp[:,ibin,:,:] = gas_temp
        # If writing data in an ascii file the ordering should be: nbin, nsec, ncol, nrad.
        # We therefore need to swap axes of array dust_temp
        # before dumping it in a binary file! just like mastermind game!
        dust_temp = np.swapaxes(dust_temp, 2, 3)  # ncol nbin nsec nrad
        dust_temp = np.swapaxes(dust_temp, 0, 1)  # nbin ncol nsec nrad
        dust_temp = np.swapaxes(dust_temp, 1, 2)  # nbin nsec ncol nrad
        dust_temp.tofile(TEMPOUT)
        TEMPOUT.close()
        del dust_temp    
        
        
# =========================
# Display gas temperature on RADMC 3D grid
# =========================
def plot_gas_temperature():

    if ( (par.RTdust_or_gas == 'dust' or par.RTdust_or_gas == 'both') and par.Tdust_eq_Thydro == 'Yes'):
        buf = np.fromfile('dust_temperature.bdat', dtype='float64')
        buf = buf[4:]
        buf = buf.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad

        # Let's reswap axes! -> ncol, nbin, nrad, nsec
        buf = np.swapaxes(buf, 0, 1)  # nsec nbin ncol nrad
        buf = np.swapaxes(buf, 0, 2)  # ncol nbin nsec nrad
        buf = np.swapaxes(buf, 2, 3)  # ncol nbin nrad nsec
        gas_temp = buf[:,0,:,:]
        del buf

    if ( (par.RTdust_or_gas == 'gas') or (par.RTdust_or_gas == 'both' and par.Tdust_eq_Thydro == 'No' and par.Tdust_eq_Tgas == 'No') ):
        gas_temp = np.fromfile('gas_temperature.binp', dtype='float64')
        gas_temp = gas_temp[3:]
        gas_temp = gas_temp.reshape(par.gas.nsec,par.gas.ncol,par.gas.nrad) # nsec ncol nrad

        # Let's reswap axes! -> ncol, nrad, nsec
        gas_temp = np.swapaxes(gas_temp, 0, 1)  # ncol nsec nrad
        gas_temp = np.swapaxes(gas_temp, 1, 2)  # ncol nrad nsec
        
        
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
    if (par.RTdust_or_gas == 'gas' or (par.RTdust_or_gas == 'both' and par.Tdust_eq_Thydro == 'No')):
        cax.set_xlabel(strgas+' temperature '+r'[K]')
        fileout = 'gas_temperature_Rz.pdf'
    if (par.RTdust_or_gas == 'dust' or par.RTdust_or_gas == 'both'):
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
    if (par.RTdust_or_gas == 'gas' or (par.RTdust_or_gas == 'both' and par.Tdust_eq_Thydro == 'No')):
        cax.set_xlabel(strgas+' surface temperature '+r'[K]')
        fileout = 'gas_temperature_surface.pdf'
    if (par.RTdust_or_gas == 'dust' or par.RTdust_or_gas == 'both'):
        cax.set_xlabel('dust surface temperature '+r'[K]')
        fileout = 'dust_temperature_surface.pdf'                
    cax.xaxis.labelpad = 8
            
    plt.savefig('./'+fileout, dpi=160)
    plt.close(fig)  # close figure as we reopen figure at every output number
        
    del gas_temp
