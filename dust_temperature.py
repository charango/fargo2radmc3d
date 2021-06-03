# import global variables
import par

from mesh import *
from field import *

import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt


# =========================
# Display dust temperature on RADMC 3D grid
# =========================
def plot_dust_temperature():

    Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
    Temp = Temp[4:]
    Temp = Temp.reshape(par.nbin,par.gas.nsec,par.gas.ncol,par.gas.nrad) # nbin nsec ncol nrad
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)
        
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial')
    fontcolor='white'

    print('--------- Plotting azimuthally-averaged temperature (R,z) for all dust size bins ----------')

    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.tedge)
    R = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
    Z = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au

    axitemp = np.zeros((par.nbin,par.gas.ncol,par.gas.nrad))   # nbin ncol nrad
    axitemp = np.sum(Temp,axis=1) / par.gas.nsec
    
    mynorm = matplotlib.colors.Normalize(vmin=axitemp.min(),vmax=axitemp.max())
    
    # Loop over size bins:
    for l in range(par.nbin):

        fig = plt.figure(figsize=(8.,8.))
        plt.subplots_adjust(left=0.14, right=0.94, top=0.88, bottom=0.11)
        ax = plt.gca()
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.tick_params(axis='x', which='minor', top=True)
        ax.tick_params(axis='y', which='minor', right=True)
        
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Altitude [au]')
        ax.set_ylim(Z.min(),Z.max())
        ax.set_xlim(R.min(),R.max())
        
        CF = ax.pcolormesh(R,Z,axitemp[l,:,:],cmap=par.mycolormap,norm=mynorm,rasterized=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')

        # title on top
        cax.xaxis.set_label_position('top')
        cax.set_xlabel('dust temperature '+r'[K]')
        cax.xaxis.labelpad = 8

        # show dust size in bottom-left corner:
        strsize = 's='+'{:0.2e}'.format(par.bins[l])+'m' # round to 2 decimals
        xstr = 1.02*R.min() 
        ystr = 0.98*Z.max() 
        ax.text(xstr,ystr,strsize, fontsize=20, color = 'black',weight='bold',horizontalalignment='left', verticalalignment='top')
        
        fileout = 'dustRz_temperature_'+str(l).zfill(2)+'.pdf'
        plt.savefig('./'+fileout, dpi=160)
        plt.close(fig)  # close figure as we reopen figure at every output number

    print('--------- Plotting surface temperature (x,y) for all dust size bins ----------')

    radius_matrix, theta_matrix = np.meshgrid(par.gas.redge,par.gas.pedge)
    X = radius_matrix * np.sin(theta_matrix) *par.gas.culength/1.5e11 # in au
    Y = radius_matrix * np.cos(theta_matrix) *par.gas.culength/1.5e11 # in au

    surftemp = Temp[:,:,par.gas.ncol-1,:]  # nbin nsec nrad

    mynorm = matplotlib.colors.Normalize(vmin=surftemp.min(),vmax=surftemp.max())
    
    # Loop over size bins:
    for l in range(par.nbin):

        fig = plt.figure(figsize=(8.,8.))
        plt.subplots_adjust(left=0.14, right=0.94, top=0.88, bottom=0.11)
        ax = plt.gca()
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.tick_params(axis='x', which='minor', top=True)
        ax.tick_params(axis='y', which='minor', right=True)
        
        ax.set_xlabel('x [au]')
        ax.set_ylabel('y [au]')
        ax.set_ylim(Y.min(),Y.max())
        ax.set_xlim(X.min(),X.max())
               
        CF = ax.pcolormesh(X,Y,surftemp[l,:,:],cmap=par.mycolormap,norm=mynorm,rasterized=True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CF, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')

        # title on top
        cax.xaxis.set_label_position('top')
        cax.set_xlabel('dust temperature '+r'[K]')
        cax.xaxis.labelpad = 8

        # show dust size in bottom-left corner:
        strsize = 's='+'{:0.2e}'.format(par.bins[l])+'m' # round to 2 decimals
        xstr = 0.98*X.min() 
        ystr = 0.98*Y.max() 
        ax.text(xstr,ystr,strsize, fontsize=20, color = 'black',weight='bold',horizontalalignment='left', verticalalignment='top')
        
        fileout = 'dustsurface_temperature_'+str(l).zfill(2)+'.pdf'
        plt.savefig('./'+fileout, dpi=160)
        plt.close(fig)  # close figure as we reopen figure at every output number

