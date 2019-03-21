# =================================================================== 
#                        FARGO2D to RADMC3D
# code written by Clement Baruteau (CB), Sebastian Perez (SP) and Marcelo Barraza (MB)
# with substantial contributions from Simon Casassus (SC) and Gaylor Wafflard-Fernandez (GWF)
# =================================================================== 
# 
# present program can run with either Python 2.X or Python 3.X.
#
# Setup FARGO2D outputs for input into RADMC3D (v0.41, Dullemond et
# al). Python based.
# 
# Considerations: 
#    - Polar coordinate system based on input simulation grid.
#      
#        X == azimuthal angle
#        Y == cylindrical radius
# 
#     Order in RADMC3D
#        x y z = r (spherical) theta (colatitude) phi (azimuth)
#     Order in FARGO2D
#        x y = azimuth radius (cylindrical)
# 
# =================================================================== 

# =========================================
#            TO DO LIST
# =========================================
# - check again that all goes well without x-axisflip!
# =========================================


# -----------------------------
# Requires librairies
# -----------------------------
import numpy as np
import matplotlib
#matplotlib.use('Agg')         # SP
import matplotlib.pyplot as plt
import pylab
import math 
import copy
import sys
import os
import subprocess
from astropy.io import fits
import matplotlib
import re
from astropy.convolution import convolve, convolve_fft
from matplotlib.colors import LinearSegmentedColormap
import psutil
import os.path
from scipy import ndimage
from copy import deepcopy
from astropy.wcs import WCS
import scipy as sp
from scipy.ndimage import map_coordinates
from pylab import *
import matplotlib.colors as colors
from makedustopac import *


# -----------------------------
# global constants in cgs units
# -----------------------------
M_Sun = 1.9891e33               # [M_sun] = g
G = 6.67259e-8                  # [G] = cm3 g-1 s-2 
au = 14959787066000.            # [Astronomical unit] = cm
m_H = 1.673534e-24              # [Hydrogen mass] = g
pi = 3.141592653589793116       # [pi]
R_Sun = 6.961e10                # [R_sun] = cm
pc = 3.085678e18                # [pc] = cm
kB = 1.380658e-16               # [kB] = erg K-1
c  = 2.99792458e10              # [c] = cm s-1
h  = 6.6261e-27                 # [h = Planck constant] = g cm2 s-1

# -------------------------------------------------------------------  
# building mesh arrays for theta, r, phi (x, y, z)
# -------------------------------------------------------------------  
class Mesh():
    # based on P. Benitez Llambay routine
    """
    Mesh class, keeps all mesh data.
    Input: directory [string] -> place where domain files are
    """
    def __init__(self, directory=""):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'
                
        # -----
        # grid
        # -----
        self.nx = self.nrad     # spherical radius
        self.ny = ncol          # colatitude (ncol is a global variable)
        self.nz = self.nsec     # azimuth
        print('RADMC grid:')
        print('number of grid cells in radius     = ', self.nx)
        print('number of grid cells in colatitude = ', self.ny)
        print('number of grid cells in azimuth    = ', self.nz)

        # -----
        # radius (x) #X-Edge
        # -----
        # CB: we build the array of spherical radii used by RADMC3D from the set of cylindrical radii
        # adopted in the FARGO 2D simulation
        try:
            domain_x = np.loadtxt(directory+"used_rad.dat")  # radial interfaces of grid cells
        except IOError:
            print('IOError')
        xm = domain_x      
        
        # -----
        # colatitude (Y) #Y-Edge
        # -----
        #
        # CB: note that we can't do mirror symmetry in RADMC3D when scattering_mode_max = 2
        # ie anisotropic scattering is assumed. We need to define the grid's colatitude on
        # both sides about the disc midplane (defined where theta = pi/2)
        #
        # thmin is set as pi/2+atan(zmax_over_H*h) with h the gas aspect ratio
        # zmax_over_H = z_max_grid / pressure scale height, value set in params.dat
        thmin = np.pi/2. - math.atan(zmax_over_H*aspectratio)
        thmax = np.pi/2.
        if polarized_scat == 'No':
            # first define array of colatitudes above midplane
            ymp = np.linspace(np.log10(thmin),np.log10(thmax),self.ny//2+1)
            # refine towards the midplane
            ym_lower = -1.0*10**(ymp)+thmin+thmax
        else:
            # first define array of colatitudes above midplane
            ymp = np.linspace(thmin,thmax,self.ny//2+1)
            # no refinement towards the midplane
            ym_lower = -ymp+thmin+thmax
        # then define array of colatitudes below midplane
        ym_upper = np.pi-ym_lower[1:self.ny//2+1]
        # and finally concatenate
        ym = np.concatenate([ym_lower[::-1],ym_upper])
        
        # -----
        # azimuth (z) # Z-Edge
        # -----
        zm = np.linspace(0.,2.*pi,self.nsec+1)
            
        self.xm = xm            # X-Edge r
        self.ym = ym            # Y-Edge theta
        self.zm = zm            # Z-Edge phi
        
        self.xmed = 0.5*(xm[:-1] + xm[1:]) # X-Center r 
        self.ymed = 0.5*(ym[:-1] + ym[1:]) # Y-Center theta
        self.zmed = 0.5*(zm[:-1] + zm[1:]) # Z-Center phi


# -------------------------------------------------------------------  
# reading fields 
# can be density, energy, velocities, etc
# -------------------------------------------------------------------  
class Field(Mesh):
    # based on P. Benitez Llambay routine
    """
    Field class, it stores all the mesh, parameters and scalar data 
    for a scalar field.
    Input: field [string] -> filename of the field
           staggered='c' [string] -> staggered direction of the field. 
                                      Possible values: 'x', 'y', 'xy', 'yx'
           directory='' [string] -> where filename is
           dtype='float64' (numpy dtype) -> 'float64', 'float32', 
                                             depends if FARGO_OPT+=-DFLOAT is activated
    """
    def __init__(self, field, staggered='c', directory='', dtype='float64'):
        if len(directory) > 1:
            if directory[-1] != '/':
                directory += '/'

        # get nrad and nsec (number of cells in radial and azimuthal directions)
        buf, buf, buf, buf, buf, buf, nrad, nsec = np.loadtxt(directory+"dims.dat",
                                                      unpack=True)
        nrad = int(nrad)
        nsec = int(nsec)
        self.nrad = nrad
        self.nsec = nsec

        Mesh.__init__(self, directory)    # all Mesh attributes inside Field
            
        # units.dat contains units of mass [kg], length [m], time [s], and temperature [k] 
        cumass, culength, cutime, cutemp = np.loadtxt(directory+"units.dat",
                                                      unpack=True)
        self.cumass = cumass
        self.culength = culength
        self.cutime = cutime 
        self.cutemp = cutemp

        # now, staggering:
        if staggered.count('x')>0:
            self.x = self.xm[:-1] # do not dump last element
        else:
            self.x = self.xmed
        if staggered.count('y')>0:
            self.y = self.ym[:-1]
        else:
            self.y = self.ymed
            
        self.data = self.__open_field(directory+field,dtype) # scalar data is here.

    def __open_field(self, f, dtype):
        """
        Reading the data
        """
        field = np.fromfile(f, dtype=dtype)
        return field.reshape(self.nx,self.nz) # 2D


# ---------------------
# define RT model class
# ---------------------
class RTmodel():
    def __init__(self, distance = 140, label='', npix=256, Lambda=800,
                 incl=30.0, posang=0.0, phi=0.0,
                 line = '12co', imolspec=1, iline=3, linenlam=80, widthkms=4,):
        
        # disk parameters
        self.distance = distance * pc
        self.label = label
        # RT pars
        self.Lambda = Lambda
        self.line = line
        self.npix   = npix
        self.incl   = incl
        self.posang = posang
        self.phi = float(phi)
        # line emission pars
        self.imolspec = imolspec
        self.iline    = iline 
        self.widthkms = widthkms 
        self.linenlam = linenlam


# -------------------------
# functions calling RADMC3D
# -------------------------
def run_mctherm():
    os.system('radmc3d mctherm')
   
def run_raytracing(M):
    command='radmc3d image lambda '+str(M.Lambda)+' npix '+str(M.npix)+' incl '+str(M.incl)+' posang '+str(M.posang)+' phi '+str(M.phi)
    if plot_tau == 'Yes':
        command='radmc3d image tracetau lambda '+str(M.Lambda)+' npix '+str(M.npix)+' incl '+str(M.incl)+' posang '+str(M.posang)+' phi '+str(M.phi)
    if polarized_scat == 'Yes':
        command=command+' stokes'
    if secondorder == 'Yes':
        command=command+' secondorder'
    print(command)
    os.system(command)

def write_radmc3d_script():
    command ='radmc3d image lambda '+str(wavelength*1e3)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)
    if plot_tau == 'Yes':
        command ='radmc3d image tracetau lambda '+str(wavelength*1e3)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)
    if polarized_scat == 'Yes':
        command=command+' stokes'
    if secondorder == 'Yes':
        command=command+' secondorder'
    SCRIPT = open('script_radmc','w')
    SCRIPT.write('radmc3d mctherm; '+command)
    SCRIPT.close()
    os.system('chmod a+x script_radmc')
    

# -------------------------
# Convert result of RADMC3D calculation into fits file
# -------------------------
def exportfits(M):
    # name of .fits file where data is output
    if plot_tau == 'No':
        outfile = 'image_'+str(M.label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(M.phi)+'_PA'+str(posangle)
    else:
        outfile = 'tau_'+str(M.label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(M.phi)+'_PA'+str(posangle)
    if secondorder == 'Yes':
        outfile = outfile+'_so'
    outfile = outfile+'.fits'
    LOG = open('fluxlog.txt','a')
    LOG.write(outfile+"\n")

    # input file produced by radmc3D
    infile = 'image.out'
    # read header info:
    f = open(infile,'r')
    iformat = int(f.readline())
    # nb of pixels
    im_nx, im_ny = tuple(np.array(f.readline().split(),dtype=int))  
    # nb of wavelengths
    nlam = int(f.readline())    # always = 1, no multiple wavelengths allowed
    # pixel size in each direction in cm
    pixsize_x, pixsize_y = tuple(np.array(f.readline().split(),dtype=float))
    # read wavelength in microns
    lbda = float(f.readline())
    f.readline()                # empty line

    # load image data
    if polarized_scat == 'No':
        images = np.loadtxt(infile, skiprows=6)
        im = images.reshape(im_ny,im_nx)
        naxis = 2
    if polarized_scat == 'Yes':
        naxis = 3
        images = np.zeros((5*im_ny*im_nx))
        im = images.reshape(5,im_ny,im_nx)
        for j in range(im_ny):
            for i in range(im_nx):
                line = f.readline()
                dat = line.split()
                im[0,j,i] = float(dat[0]) # I
                im[1,j,i] = float(dat[1]) # Q
                im[2,j,i] = float(dat[2]) # U
                im[3,j,i] = float(dat[3]) # V
                im[4,j,i] = math.sqrt(float(dat[1])**2.0+float(dat[2])**2.0) # P
                if (j == im_ny-1) and (i == im_nx-1):
                    f.readline()     # empty line
            
    # close image file
    f.close()

    # calculate physical scales
    distance = M.distance          # distance is in cm here
    pixsize_x_deg = 180.0*pixsize_x / distance / pi
    pixsize_y_deg = 180.0*pixsize_y / distance / pi

    # surface of a pixel in radian squared
    pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (pi/180.)**2
    # 1 Jansky converted in cgs x pixel surface in cm^2
    fluxfactor = 1.e23 * pixsurf_ster

    # Fits header
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32
    hdu.header['NAXIS'] = 2 # naxis
    hdu.header['NAXIS1'] = im_nx
    hdu.header['NAXIS2'] = im_ny
    #hdu.header['NAXIS3'] = 1
    #hdu.header['NAXIS4'] = 1
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    #hdu.header['CTYPE3'] = 'FREQ'
    #hdu.header['CTYPE4'] = 'STOKES'
    hdu.header['CRVAL1'] = 8.261472379700E+01 # float(0.0)
    hdu.header['CRVAL2'] = 2.533239051468E+01 # float(0.0)
    #hdu.header['CRVAL3'] = 33.0E+09
    #hdu.header['CRVAL4'] = 1.0E+00
    hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
    hdu.header['CDELT2'] = float(pixsize_y_deg)
    #hdu.header['CDELT3'] = 8.05E+09
    #hdu.header['CDELT4'] = 1.0E+00
    hdu.header['CUNIT1'] = 'deg     '
    hdu.header['CUNIT2'] = 'deg     '
    #hdu.header['CUNIT3'] = 'Hz     '
    #hdu.header['CUNIT4'] = ' '
    hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
    hdu.header['CRPIX2'] = float((im_ny+1.)/2.)
    hdu.header['BUNIT'] = 'JY/PIXEL'
    hdu.header['BTYPE'] = 'Intensity'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0
    del hdu.header['EXTEND']

    # keep track of all parameters in params.dat file
    #for i in range(len(lines_params)):
    #    hdu.header[var[i]] = par[i]
    LOG.write('pixsize '+str(pixsize_x_deg*3600.)+"\n")

    # conversion of the intensity from erg/s/cm^2/steradian to Jy/pix
    if plot_tau == 'No':
        im = im*fluxfactor

    hdu.data = im.astype('float32')

    if plot_tau == 'No':
        print("Total flux [Jy] = "+str(np.sum(hdu.data)))
        LOG.write('flux '+str(np.sum(hdu.data))+"\n")
    LOG.close()
    
    hdu.writeto(outfile, output_verify='fix', overwrite=True)
    return outfile


# --------------------------------------------------
# function return kernel for convolution by an elliptical beam
# --------------------------------------------------
def Gauss_filter(img, stdev_x, stdev_y, PA, Plot=False):
    ''' 
    img: image array
    stdev_x: float
    BMAJ sigma dev.
    stdev_y: float
    BMIN sigma dev.
    PA: East of North in degrees
    '''

    image = img 
    (nx0, ny0) = image.shape
        
    nx = np.minimum(nx0, int(8.*stdev_x))
    # pixel centering
    nx = nx + ((nx+1) % 2)
    ny = nx
        
    x = np.arange(nx)+1
    y = np.arange(ny)+1
    X, Y = np.meshgrid(x,y)
    X0 = np.floor(nx//2)+1
    Y0 = np.floor(ny//2)+1

    data = image
    
    theta = np.pi * PA / 180.
    A = 1
    a = np.cos(theta)**2/(2*stdev_x**2) + np.sin(theta)**2/(2*stdev_y**2)
    b = np.sin(2*theta)/(4*stdev_x**2) - np.sin(2*theta)/(4*stdev_y**2)
    c = np.sin(theta)**2/(2*stdev_x**2) + np.cos(theta)**2/(2*stdev_y**2)

    Z = A*np.exp(-(a*(X-X0)**2-2*b*(X-X0)*(Y-Y0)+c*(Y-Y0)**2))
    Z /= np.sum(Z)
    
    if Plot==True:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm                
        plt.imshow(Z, cmap=cm.jet)
        plt.show()
                
    result = convolve_fft(data, Z, boundary='fill', fill_value=0.0)

    if Plot==True:
        plt.imshow(result, cmap=cm.magma)
        plt.show()

    return result


# -------------------------------------------------------------------  
# writing dustopac
# -------------------------------------------------------------------  
def write_dustopac(species=['ac_opct', 'Draine_Si'],nbin=20):
    print('writing dustopac.inp')
    hline = "-----------------------------------------------------------------------------\n"
    OPACOUT = open('dustopac.inp','w')

    lines0=["2 \t iformat (2)\n",
            str(nbin)+" \t species\n",
            hline]
    OPACOUT.writelines(lines0)
    # put first element to 10 if dustkapscatmat_species.inp input file, or 1 if dustkappa_species.inp input file
    if (scat_mode >= 3):
        inputstyle = 10
    else:
        inputstyle = 1
    for i in range(nbin):
        lines=[str(inputstyle)+" \t in which form the dust opacity of dust species is to be read\n",
               "0 \t 0 = thermal grains\n",
               species+str(i)+" \t dustkap***.inp file\n",
               hline
           ]
        OPACOUT.writelines(lines)
    OPACOUT.close()


# -------------------------------------------------------------------  
# read opacities
# -------------------------------------------------------------------  
def read_opacities(filein):
    params = open(filein,'r')
    lines_params = params.readlines()
    params.close()                 
    lbda = []                  
    kappa_abs = []               
    kappa_sca = []
    g = []
    for line in lines_params:      
        try:
            line.split()[0][0]     # check if blank line (GWF)
        except:
            continue
        if (line.split()[0][0]=='#'): # check if line starts with a # (comment)
            continue
        else:
            if (len(line.split()) == 4):
                l, a, s, gg = line.split()[0:4]
            else:
                continue
        lbda.append(float(l))
        kappa_abs.append(float(a))
        kappa_sca.append(float(s))
        g.append(float(gg))

    lbda = np.asarray(lbda)
    kappa_abs = np.asarray(kappa_abs)
    kappa_sca = np.asarray(kappa_sca)
    g = np.asarray(g)
    return [lbda,kappa_abs,kappa_sca,g]


# -------------------------------------------------------------------  
# plotting opacities
# -------------------------------------------------------------------  
def plot_opacities(species='mix_2species_porous',amin=0.1,amax=1000,nbin=10,lbda1=1e-3):
    ax = plt.gca()
    ax.tick_params(axis='both',length = 10, width=1)

    plt.xlabel(r'Dust size [meters]')
    plt.ylabel(r'Opacities $[{\rm cm}^2\;{\rm g}^{-1}]$')

    absorption1 = np.zeros(nbin)
    scattering1 = np.zeros(nbin)
    sizes = np.logspace(np.log10(amin), np.log10(amax), nbin)

    for k in range(nbin):
        if polarized_scat == 'No':
            filein = 'dustkappa_'+species+str(k)+'.inp'
        else:
            filein = 'dustkapscatmat_'+species+str(k)+'.inp'
        (lbda, kappa_abs, kappa_sca, g) = read_opacities(filein)
        #(lbda, kappa_abs, kappa_sca, g) = np.loadtxt(filein, unpack=True, skiprows=2)
        
        i1 = np.argmin(np.abs(lbda-lbda1))

        # interpolation in log
        l1 = lbda[i1-1]
        l2 = lbda[i1+1]
        k1 = kappa_abs[i1-1]
        k2 = kappa_abs[i1+1]
        ks1 = kappa_sca[i1-1]
        ks2 = kappa_sca[i1+1]
        absorption1[k] =  (k1*np.log(l2/lbda1) +  k2*np.log(lbda1/l1))/np.log(l2/l1)
        scattering1[k] = (ks1*np.log(l2/lbda1) + ks2*np.log(lbda1/l1))/np.log(l2/l1)
        
    # nice colors 
    c20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
           (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
           (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
           (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
           (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(c20)):
        r, g, b = c20[i]    
        c20[i] = (r / 255., g / 255., b / 255.) 

    lbda1 *= 1e-3  # in mm

    plt.loglog(sizes, absorption1, lw=2., linestyle = 'solid', color = c20[1], label='$\kappa_{abs}$ at '+str(lbda1)+' mm')
    plt.loglog(sizes, absorption1+scattering1, lw=2., linestyle = 'dashed', color = c20[1], label='$\kappa_{abs}$+$\kappa_{sca}$ at '+str(lbda1)+' mm')
    plt.legend()

    plt.ylim(absorption1.min(),(absorption1+scattering1).max())
    filesaveopac = 'opacities_'+species+'.pdf'
    plt.savefig('./'+filesaveopac, bbox_inches='tight', dpi=160)
    plt.clf()


# ---------------------------------------
# write spatial grid in file amr_grid.inp
# ---------------------------------------
def write_AMRgrid(F, R_Scaling=1, Plot=False):

    print("writing spatial grid")
    path_grid='amr_grid.inp'

    grid=open(path_grid,'w')

    grid.write('1 \n')              # iformat/ format number = 1
    grid.write('0 \n')              # Grid style (regular = 0)
    grid.write('101 \n')            # coordsystem: 100 < spherical < 200 
    grid.write('0 \n')              # gridinfo
    grid.write('1 \t 1 \t 1 \n')    # incl x, incl y, incl z

    # spherical radius, colatitude, azimuth
    grid.write(str(F.nx)+ '\t'+ str(F.ny)+'\t'+ str(F.nz)+'\n') 

    # nx+1 dimension as we need to enter the coordinates of the cells edges
    # hence we put F.xm and F.xmed
    for i in range(F.nx + 1):  
        grid.write(str(F.xm[i]*F.culength*1e2)+'\t') # with unit conversion in cm
    grid.write('\n')

    # colatitude
    for i in range(F.ny + 1):
        grid.write(str(F.ym[i])+'\t')
    grid.write('\n')

    # azimuth 
    for i in range(F.nz + 1):
        grid.write(str(F.zm[i])+'\t') 
    grid.write('\n')

    grid.close()

    if Plot:
        import matplotlib.pyplot as plt
        ax = plt.gca()
        P,R = np.meshgrid(F.zm, F.xm)    
        X = R*np.cos(P)
        Y = R*np.sin(P) 
        plt.pcolor(X,Y,np.random.rand(len(F.xm),len(F.zm)),cmap='plasma',edgecolors='black')
            
        plt.axis('equal')
        plt.show()


# -----------------------
# writing out wavelength 
# -----------------------
def write_wavelength():
    wmin = 0.1
    wmax = 10000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))
    
    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in range(1, Nw):
        waves[i]=wmin*Pw**i

    print('writing wavelength_micron.inp')

    path = 'wavelength_micron.inp'
    wave = open(path,'w')
    wave.write(str(Nw)+'\n')
    for i in range(Nw):
        wave.write(str(waves[i])+'\n')
    wave.close()


# -----------------------
# writing out star parameters 
# -----------------------
def write_stars(Rstar = 1, Tstar = 6000):
    wmin = 0.1
    wmax = 10000.0
    Nw = 150
    Pw = (wmax/wmin)**(1.0/(Nw-1))
    
    waves = np.zeros(Nw)
    waves[0] = wmin
    for i in range(1, Nw):
        waves[i]=wmin*Pw**i

    print('writing stars.inp')

    path = 'stars.inp'
    wave = open(path,'w')

    wave.write('\t 2\n') 
    wave.write('1 \t'+str(Nw)+'\n')
    wave.write(str(Rstar*R_Sun)+'\t'+str(M_Sun)+'\t 0 \t 0 \t 0 \n')
    for i in range(Nw):
        wave.write('\t'+str(waves[i])+'\n')
    wave.write('\t -'+str(Tstar)+'\n')
    wave.close()


# --------------------
# writing radmc3d.inp
# --------------------
def write_radmc3dinp(incl_dust = 1,
                     incl_lines = 0,
                     nphot = 1000000,
                     nphot_scat = 1000000,
                     nphot_spec = 1000000,
                     nphot_mono = 1000000,
                     istar_sphere = 0,
                     scattering_mode_max = 0,
                     tgas_eq_tdust = 1,
                     modified_random_walk = 0,
                     itempdecoup=1,
                     setthreads=2,
                     rto_style=3 ):

    print('writing radmc3d.inp')

    RADMCINP = open('radmc3d.inp','w')
    inplines = ["incl_dust = "+str(int(incl_dust))+"\n",
                "incl_lines = "+str(int(incl_lines))+"\n",
                "nphot = "+str(int(nphot))+"\n",
                "nphot_scat = "+str(int(nphot_scat))+"\n",
                "nphot_spec = "+str(int(nphot_spec))+"\n",
                "nphot_mono = "+str(int(nphot_mono))+"\n",
                "istar_sphere = "+str(int(istar_sphere))+"\n",
                "scattering_mode_max = "+str(int(scattering_mode_max))+"\n",
                "tgas_eq_tdust = "+str(int(tgas_eq_tdust))+"\n",
                "modified_random_walk = "+str(int(modified_random_walk))+"\n",
                "itempdecoup = "+str(int(itempdecoup))+"\n",
                "setthreads="+str(int(setthreads))+"\n",
                "rto_style="+str(int(rto_style))+"\n"]

    RADMCINP.writelines(inplines)
    RADMCINP.close()


# -----------------------------------------------------
# Main routine to deproject cartesian flux maps onto polar maps (SC, SP)
# -----------------------------------------------------
def exec_polar_expansions(filename_source,workdir,PA,cosi,RA=False,DEC=False,alpha_min=False,Delta_min=False,XCheckInv=False,DoRadialProfile=True,ProfileExtractRadius=-1,DoAzimuthalProfile=False,PlotRadialProfile=True,a_min=-1,a_max=-1,zoomfactor=1.):

    fieldscale=2. # shrink radial field of view of polar maps by this factor

    if workdir[-1] != '/':
        workdir += '/'
    os.system("rm -rf mkdir "+workdir)
    os.system("mkdir "+workdir)
    inbasename=os.path.basename(filename_source)
    filename_fullim=re.sub('.fits', '_fullim.fits', inbasename)
    filename_fullim=workdir+filename_fullim
    
    hdu0 = fits.open(filename_source)
    hdr0 = hdu0[0].header
    # copy content of original fits file
    os.system("rsync -va "+filename_source+" "+filename_fullim)
    hdu = fits.open(filename_fullim)
    im1=hdu[0].data
    hdr1=hdu[0].header

    if (not (isinstance(Delta_min,bool))):
        if (isinstance(RA,bool)):
            if (not RA):
                RA=hdr1['CRVAL1']
                DEC=hdr1['CRVAL2']
        
        RA=RA+(np.sin(alpha_min*np.pi/180.)*Delta_min/3600.)/np.cos(DEC*np.pi/180.)
        DEC=DEC+np.cos(alpha_min*np.pi/180.)*Delta_min/3600.

    elif (not isinstance(RA,float)):
        sys.exit("must provide a center")

    nx=int(hdr1['NAXIS1']/zoomfactor)   # zoomfactor = 1 in practice
    ny=nx
    if ( (nx % 2) == 0):
        nx=nx+1
        ny=ny+1

    hdr2 = deepcopy(hdr1)    
    hdr2['NAXIS1']=nx
    hdr2['NAXIS2']=ny
    hdr2['CRPIX1']=(nx+1)/2
    hdr2['CRPIX2']=(ny+1)/2
    hdr2['CRVAL1']=RA
    hdr2['CRVAL2']=DEC

    resamp=gridding(filename_fullim,hdr2, fullWCS=False)
    fileout_centered=re.sub('fullim.fits', 'centered.fits', filename_fullim)
    fits.writeto(fileout_centered,resamp, hdr2, overwrite=True)

    # First rotate original, centred image by position angle
    rotangle = PA
    im1rot = ndimage.rotate(resamp, rotangle, reshape=False)
    fileout_rotated=re.sub('fullim.fits', 'rotated.fits', filename_fullim)
    fits.writeto(fileout_rotated,im1rot, hdr2, overwrite=True)
    hdr3 = deepcopy(hdr2)
    hdr3['CDELT1']=hdr3['CDELT1']*cosi

    # Then deproject with inclination via hdr3
    im3=gridding(fileout_rotated,hdr3)
    fileout_stretched=re.sub('fullim.fits', 'stretched.fits', filename_fullim)
    fits.writeto(fileout_stretched,im3, hdr2, overwrite=True)

    # Finally work out polar transformation
    im_polar = sp.ndimage.geometric_transform(im3,cartesian2polar, 
                                              order=1,
                                              output_shape = (im3.shape[0], im3.shape[1]),
                                              extra_keywords = {'inputshape':im3.shape,'fieldscale':fieldscale,
                                                                'origin':(((nx+1)/2)-1,((ny+1)/2)-1)}) 
    
    nphis,nrs=im_polar.shape
    
    hdupolar = fits.PrimaryHDU()
    hdupolar.data = im_polar
    hdrpolar=hdupolar.header
    hdrpolar['CRPIX1']=1
    hdrpolar['CRVAL1']=0.
    hdrpolar['CDELT1']=2. * np.pi / nphis
    hdrpolar['CRPIX2']=1
    hdrpolar['CRVAL2']=0.
    hdrpolar['CDELT2']=(hdr3['CDELT2'] / fieldscale)
    hdupolar.header = hdrpolar
    
    fileout_polar=re.sub('fullim.fits', 'polar.fits', filename_fullim)
    hdupolar.writeto(fileout_polar, overwrite=True)


# --------------------
#Function required in exec_polar_expansions
# -------------------- 
def datafits(namefile):
    """
    Open a FITS image and return datacube and header.
    """
    datacube = fits.open(namefile)[0].data
    hdr = fits.open(namefile)[0].header
    return datacube, hdr

# --------------------
#Function required in exec_polar_expansions
# -------------------- 
def gridding(imagefile_1, imagefile_2,fileout=False,fullWCS=True):
    """
    Interpolates Using ndimage and astropy.wcs for coordinate system.
    """
    if (isinstance(imagefile_1,str)):
        im1, hdr1 = datafits(imagefile_1)
    elif (isinstance(imagefile_1,fits.hdu.image.PrimaryHDU)):
        im1 = imagefile_1.data
        hdr1 = imagefile_1.header
    elif (isinstance(imagefile_1,fits.hdu.hdulist.HDUList)):
        im1 = imagefile_1[0].data
        hdr1 = imagefile_1[0].header
    else:
        sys.exit("not an recognized input format")
        
    if (isinstance(imagefile_2,str)):
        im2, hdr2 = datafits(imagefile_2)
    else:
        hdr2=imagefile_2
        
    w1 = WCS(hdr1)
    w2 = WCS(hdr2)
    
    n2x = hdr2['NAXIS1']
    n2y = hdr2['NAXIS2']
    k2s=sp.arange(0,n2x)
    l2s=sp.arange(0,n2y)
    kk2s, ll2s = sp.meshgrid(k2s, l2s)

    if (fullWCS):
        xxs2wcs, yys2wcs = w2.all_pix2world(kk2s, ll2s, 0)
        kk1s, ll1s = w1.all_world2pix(xxs2wcs,yys2wcs,0,tolerance=1e-12)
    else:
        xxs2wcs, yys2wcs = w2.wcs_pix2world(kk2s, ll2s, 0)
        kk1s, ll1s = w1.wcs_world2pix(xxs2wcs,yys2wcs,0)

    resamp = map_coordinates(im1, [ll1s, kk1s])

    if (fileout):
        fits.writeto(fileout,resamp, hdr2, overwrite=True)

    return resamp

# --------------------
#Function required in exec_polar_expansions
# -------------------- 
def cartesian2polar(outcoords, inputshape, origin, fieldscale=1.):
    # Routine from original PolarMaps.py
    """Coordinate transform for converting a polar array to Cartesian coordinates. 
    inputshape is a tuple containing the shape of the polar array. origin is a
    tuple containing the x and y indices of where the origin should be in the
    output array."""

    rindex, thetaindex = outcoords
    x0, y0 = origin

    theta = thetaindex * 2 * np.pi / (inputshape[0]-1)   # inputshape[0] = nbpixels+1
    y = rindex*np.cos(theta)/fieldscale
    x = rindex*np.sin(theta)/fieldscale
    ix = -x + x0
    iy = y +  y0
    
    return (iy,ix)

# -----------------------------------------------------

# =========================
# 1. read RT parameter file
# =========================
params = open("params.dat",'r')# opening parfile
lines_params = params.readlines()     # reading parfile
params.close()                 # closing parfile
par = []                       # allocating a dictionary
var = []                       # allocating a dictionary
for line in lines_params:      # iterating over parfile 
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
        except ValueError:     # if it is not integer, we know it is str
            value = '"' + value + '"'
    par.append(value)
    var.append(name)

for i in range(len(par)):
    #exec(var[i]+"="+par[i])
    exec(var[i]+"="+str(par[i]))

# save copy of params.dat
os.system('cp params.dat params_last.dat')


print('--------- RT parameters ----------')
print('directory = ', dir)
print('output number  = ', on)
print('wavelength [mm] = ', wavelength)
print('do we plot optical depth? : ', plot_tau)
print('do we compute polarized intensity image? : ', polarized_scat)
if polarized_scat == 'Yes':
    z_expansion = 'G'
    print('truncation_radius for dust pressure scale height [arcseconds] : ', truncation_radius)
    if scat_mode != 5:
        sys.exit("To get a polarized intensity image you need scat_mode = 5 in params.dat. Abort!")
    print('mask_radius in polarized intensity image [arcseconds] : ', mask_radius)
print('dust minimum size [m] = ', amin)
print('dust maximum size [m] = ', amax)
print('minus slope of dust size distribution = ', pindex)
print('dust-to-gas mass ratio = ', ratio)
print('number of size bins = ', nbin)
print('disc distance [pc] = ', distance)
print('disc inclination [deg] = ', inclination)
print('disc phi angle [deg] = ', phiangle)
print('disc position angle [deg] = ', posangle)
print('beam major axis [arcsec] = ', bmaj)
print('beam minor axis [ascsec] = ', bmin)
print('beam position angle [deg] = ', bpaangle)
print('star radius [Rsun] = ', rstar)
print('star effective temperature [K] = ', teff)
print('number of grid cells in colatitude for RADMC3D = ', ncol)
if ncol%2 == 1:
    sys.exit('the number of columns needs to be an even number, please try again!')
print('type of vertical expansion done for dust mass volume density = ', z_expansion)
if z_expansion == 'G':
    print('Hd / Hgas = 1 if R < truncation_radius, R^-2 decrease beyond')
if z_expansion == 'T':
    print('Hd / Hgas = sqrt(alpha/(alpha+St))')
if z_expansion == 'T2':
    print('Hd / Hgas = sqrt(10*alpha/(10*alpha+St))')
if z_expansion == 'F':
    print('Hd / Hgas = 0.7 x ((St + 1/St)/1000)^0.2 (Fromang & Nelson 09)')
print('do we recompute all dust densities? : ', recalc_density)
print('do we include a bin with small dust tightly coupled to the gas? : ', bin_small_dust)
print('what is the maximum altitude of the 3D grid in pressure scale heights? : ', zmax_over_H)
print('do we recompute all dust opacities? : ', recalc_opac)
print('do we plot dust opacities? : ', plot_opac)
print('do we use pre-calculated opacities, which are located in opacity_dir directory? : ', precalc_opac)
print('do we run RADMC3D calculation of temperature and ray tracing? : ', recalc_radmc)
if recalc_radmc == 'Yes':
    print('how many cores do we use for radmc calculation? : ', nbcores)
    recalc_fluxmap = 'Yes'
print('do we recompute fits file of raw flux map from image.out? This is useful if radmc has been run on a different platform: ', recalc_rawfits)
if recalc_rawfits == 'Yes':
    recalc_fluxmap = 'Yes'
print('do we recompute convolved flux map from the output of RADMC3D calculation? : ', recalc_fluxmap)
print('do we compute 2D analytical solution to RT equation w/o scattering? : ', calc_abs_map)
if calc_abs_map == 'Yes':
    print('if so, do we assume the dust surface density equal to the gas surface density? : ', dustdens_eq_gasdens)
print('do we take the gas (hydro) temperature for the dust temperature? : ', Tdust_eq_Tgas)
print('do we add white noise to the raw flux maps? : ', add_noise)
if add_noise == 'Yes':
    print('if so, level of noise in Jy / beam', noise_dev_std)
print('number of pixels in each direction for flux map computed by RADMC3D = ', nbpixels)
print('scattering mode max for RADMC3D = ', scat_mode)
nb_photons = int(nb_photons)
print('number of photon packaged used by RADMC3D', nb_photons)
nb_photons_scat = int(nb_photons_scat)
print('number of photon packaged used by RADMC3D for scattering', nb_photons_scat)
print('type of dust particles for calculation of dust opacity files = ', species)
print('name of directory with .lnk dust opacity files = ', opacity_dir)
print('x- and y-max in final image [arcsec] = ', minmaxaxis)
print('do we use second-order integration for ray tracing in RADMC3D? : ', secondorder)
print('do we flip x-axis in disc plane? (mirror symmetry x -> -x in the simulation): ', xaxisflip)
print('do we check beam shape by adding a source point at the origin?: ', check_beam)
print('do we deproject the predicted image into polar coords?:', deproj_polar) # SP
print('do we plot radial temperature profiles? : ', plot_temperature)

# x-axis flip means that we apply a mirror-symetry x -> -x to the 2D simulation plane,
# which in practice is done by adding 180 degrees to the disc's inclination wrt the line of sight
inclination_input = inclination
if xaxisflip == 'Yes':
    inclination = inclination + 180.0

# this is how the beam position angle should be modified to be understood as
# measuring East from North. You can check this by setting check_beam to Yes
# in the params.dat parameter file
bpaangle = -90.0-bpaangle

# label for the name of the image file created by RADMC3D
label = dir+'_o'+str(on)+'_p'+str(pindex)+'_r'+str(ratio)+'_a'+str(amin)+'_'+str(amax)+'_nb'+str(nbin)+'_mode'+str(scat_mode)+'_np'+str('{:.0e}'.format(nb_photons))+'_nc'+str(ncol)+'_z'+str(z_expansion)+'_xf'+str(xaxisflip)+'_Td'+str(Tdust_eq_Tgas)+'_bmaj'+str(bmaj)

# set spherical grid, array allocation.
# get the aspect ratio and flaring index used in the numerical simulation
command = 'awk " /^AspectRatio/ " '+dir+'/*.par'
# check which version of python we're using
if sys.version_info[0] < 3:   # python 2.X
    buf = subprocess.check_output(command, shell=True)
else:                         # python 3.X
    buf = subprocess.getoutput(command)
aspectratio = float(buf.split()[1])
# get the flaring index used in the numerical simulation
command = 'awk " /^FlaringIndex/ " '+dir+'/*.par'
if sys.version_info[0] < 3:
    buf = subprocess.check_output(command, shell=True)
else:
    buf = subprocess.getoutput(command)
flaringindex = float(buf.split()[1])
# get the alpha viscosity used in the numerical simulation
command = 'awk " /^AlphaViscosity/ " '+dir+'/*.par'
if sys.version_info[0] < 3:
    buf = subprocess.check_output(command, shell=True)
else:
    buf = subprocess.getoutput(command)
alphaviscosity = float(buf.split()[1])
# if alphaviscosity is null, then try to see if a constant
# kinematic viscosity has been used in the simulation
if alphaviscosity == 0:
    command = 'awk " /^Viscosity/ " '+dir+'/*.par'
    if sys.version_info[0] < 3:
        buf = subprocess.check_output(command, shell=True)
    else:
        buf = subprocess.getoutput(command)
    viscosity = float(buf.split()[1])
    # simply set constant alpha value as nu / h^2 (ok at code's unit of length)
    alphaviscosity = viscosity * (aspectratio**(-2.0))
# get the grid radial spacing used in the numerical simulation
command = 'awk " /^RadialSpacing/ " '+dir+'/*.par'
if sys.version_info[0] < 3:
    buf = subprocess.check_output(command, shell=True)
else:
    buf = subprocess.getoutput(command)
radialspacing = str(buf.split()[1])
print('gas aspect ratio = ', aspectratio)
print('gas flaring index = ', flaringindex)
print('gas alpha turbulent viscosity = ', alphaviscosity)
print('gas radial spacing = ', radialspacing)

# gas surface density field: 
gas  = Field(field='gasdens'+str(on)+'.dat', directory=dir)

# computational grid: R = grid cylindrical radius in code units, T = grid azimuth in radians
R = gas.xm
T = gas.zm
# number of grid cells in the radial and azimuthal directions
nrad = gas.nrad
nsec = gas.nsec
# extra useful quantities (code units)
Rinf = gas.xm[0:len(gas.xm)-1]
Rsup = gas.xm[1:len(gas.xm)]
surface  = np.zeros(gas.data.shape) # 2D array containing surface of each grid cell
surf = pi * (Rsup*Rsup - Rinf*Rinf) / nsec # surface of each grid cell (code units)
for th in range(nsec):
    surface[:,th] = surf

# Allocate arrays
bins = np.logspace(np.log10(amin), np.log10(amax), nbin+1)
dust = np.zeros((nsec*nrad*nbin))
nparticles = np.zeros(nbin)     # number of particles per bin size
avgstokes  = np.zeros(nbin)     # average Stokes number of particles per bin size

# check out memory available on your architecture
mem_gib = float(psutil.virtual_memory().total/1024**3)
mem_array_gib = nrad*nsec*ncol*nbin*8.0/(1024.**3)
if (mem_array_gib/mem_gib > 0.5):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Beware that the memory requested for allocating the dust mass volume density or the temperature arrays')
    print('is very close to the amount of memory available on your architecture...')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')


# =========================
# 2. compute dust surface density for each size bin
# =========================
print('--------- computing dust surface density ----------')

# -------------------------
# a) Case with no polarized scattering: we infer the dust's surface
# density from the results of the gas+dust hydrodynamical simulation
# -------------------------
if (recalc_density == 'Yes' and polarized_scat == 'No'):

    # read information on the dust particles
    (rad, azi, vr, vt, Stokes, a) = np.loadtxt(dir+'/dustsystat'+str(on)+'.dat', unpack=True)

    # Populate dust bins
    for m in range (len(a)):   # CB: sum over dust particles
        r = rad[m]
        t = azi[m]
        # radial index of the cell where the particle is
        if radialspacing == 'L':
            i = int(np.log(r/gas.xm.min())/np.log(gas.xm.max()/gas.xm.min()) * nrad)
        else:
            i = np.argmin(np.abs(gas.xm-r))
        if (i < 0 or i >= nrad):
            sys.exit('pb with i = ', i, ' in recalc_density step: I must exit!')
        # azimuthal index of the cell where the particle is
        # (general expression since grid spacing in azimuth is always arithmetic)
        j = int((t-gas.zm.min())/(gas.zm.max()-gas.zm.min()) * nsec)
        if (j < 0 or j >= nsec):
            sys.exit('pb with j = ', j, ' in recalc_density step: I must exit!')
        # particle size
        pcsize = a[m]   
        # find out which bin particle belongs to
        ibin = int(np.log(pcsize/bins.min())/np.log(bins.max()/bins.min()) * nbin)
        if (ibin >= 0 and ibin < nbin):
            k = ibin*nsec*nrad + j*nrad + i
            dust[k] +=1
            nparticles[ibin] += 1
            avgstokes[ibin] += Stokes[m]

    for ibin in range(nbin):
        if nparticles[ibin] == 0:
            nparticles[ibin] = 1
        avgstokes[ibin] /= nparticles[ibin]
        print(str(nparticles[ibin])+' grains between '+str(bins[ibin])+' and '+str(bins[ibin+1])+' meters')
    
    # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
    dustcube = dust.reshape(nbin, nsec, nrad)  
    dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec

    # Mass of gas in units of the star's mass
    Mgas = np.sum(gas.data*surface)
    print('Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*gas.cumass))

    frac = np.zeros(nbin)
    buf = 0.0
    # finally compute dust surface density for each size bin
    for ibin in range(nbin):
        # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
        frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin]**(4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
        # total mass of dust particles in current size bin 'ibin'
        M_i_dust = ratio * Mgas * frac[ibin]
        buf += M_i_dust
        print('Dust mass [in units of Mstar] in species ', ibin, ' = ', M_i_dust)
        # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
        dustcube[ibin,:,:] *= M_i_dust / surface / nparticles[ibin]
        # conversion in g/cm^2
        dustcube[ibin,:,:] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec

    # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
    if bin_small_dust == 'Yes':
        frac[0] *= 5e3
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
        print("Mass fraction of bin 0 changed to: ",str(frac[0]))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        imin = np.argmin(np.abs(gas.xmed-1.4))  # radial index corresponding to 0.3"
        imax = np.argmin(np.abs(gas.xmed-2.8))  # radial index corresponding to 0.6"
        dustcube[0,imin:imax,:] = gas.data[imin:imax,:] * ratio * frac[0] * (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
        
    print('Total dust mass [g] = ', np.sum(dustcube[:,:,:]*surface*(gas.culength*1e2)**2.))
    print('Total dust mass [Mgas] = ', np.sum(dustcube[:,:,:]*surface*(gas.culength*1e2)**2.)/(Mgas*gas.cumass*1e3))
    print('Total dust mass [Mstar] = ', np.sum(dustcube[:,:,:]*surface*(gas.culength*1e2)**2.)/(gas.cumass*1e3))
    
    # Total dust surface density
    dust_surface_density = np.sum(dustcube,axis=0)
    print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())

# -------------------------
# b) Case with polarized scattering: we say that the dust is perfectly
# coupled to the gas
# -------------------------
if (recalc_density == 'Yes' and polarized_scat == 'Yes'):

    # Mass of gas in units of the star's mass
    Mgas = np.sum(gas.data*surface)
    print 'Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*gas.cumass) 

    dustcube = dust.reshape(nbin, nsec, nrad)  
    dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
    frac = np.zeros(nbin)
    buf = 0.0
    
    # compute dust surface density for each size bin
    for ibin in range(nbin):
        # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
        frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin]**(4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
        # total mass of dust particles in current size bin 'ibin'
        M_i_dust = ratio * Mgas * frac[ibin]
        buf += M_i_dust
        print('Dust mass [in units of Mstar] in species ', ibin, ' = ', M_i_dust)
        # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
        dustcube[ibin,:,:] = ratio * gas.data * frac[ibin]
        # conversion in g/cm^2
        dustcube[ibin,:,:] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
        # decrease dust surface density beyond truncation radius by R^-2
        # NB: truncation_radius is in arcseconds
        rcut_in_code_units  = truncation_radius*distance*au/gas.culength/1e2
        rmask_in_code_units = mask_radius*distance*au/gas.culength/1e2
        for i in range(nrad):
            if (gas.xmed[i] > rcut_in_code_units):
                dustcube[ibin,i,:] *= ( (gas.xmed[i]/rcut_in_code_units)**(-10.0) )
            if (gas.xmed[i] < rmask_in_code_units):
                dustcube[ibin,i,:] = 0.0 # *= ( (gas.xmed[i]/rmask_in_code_units)**(10.0) ) CUIDADIN!

    print('Total dust mass [g] = ', buf*gas.cumass*1e3)
    
    # Total dust surface density
    dust_surface_density = np.sum(dustcube,axis=0)
    print('Maximum dust surface density [in g/cm^2] is ', dust_surface_density.max())


# =========================
# 3. Compute dust mass volume density for each size bin
#    (vertical expansion assuming hydrostatic equilibrium)
# =========================
if recalc_density == 'Yes':
    print('--------- computing dust mass volume density ----------')

    DUSTOUT = open('dust_density.inp','w')
    DUSTOUT.write('1 \n')                           # iformat  
    DUSTOUT.write(str(nrad*nsec*ncol)+' \n')        # n cells
    DUSTOUT.write(str(int(nbin))+' \n')             # nbin size bins 

    # array (ncol, nbin, nrad, nsec)
    rhodustcube = np.zeros((ncol,nbin,nrad,nsec))

    # dust aspect ratio as function of ibin and r (or actually, R, cylindrical radius)
    hd = np.zeros((nbin,nrad))

    for ibin in range(nbin):
        if polarized_scat == 'No':
            St = avgstokes[ibin]                              # avg stokes number for that bin 
        hgas = aspectratio * (gas.xmed)**(flaringindex)       # gas aspect ratio (gas.xmed[i] = R in code units)
        # vertical extension depends on grain Stokes number
        # T = theoretical: hd/hgas = sqrt(alpha/(St+alpha))
        # T2 = theoretical: hd/hgas = sqrt(Dz/(St+Dz)) with Dz = 10xalpha here is the coefficient for
        # vertical diffusion at midplane, which can differ from alpha
        # F = extrapolation from the simulations by Fromang & Nelson 09
        # G = Gaussian = same as gas (case of well-coupled dust for polarized intensity images)
        if z_expansion == 'F':
            hd[ibin,:] = 0.7 * hgas * ((St+1./St)/1000.)**(0.2)     
        if z_expansion == 'T':
            hd[ibin,:] = hgas * np.sqrt(alphaviscosity/(alphaviscosity+St))
        if z_expansion == 'T2':
            hd[ibin,:] = hgas * np.sqrt(10.0*alphaviscosity/(10.0*alphaviscosity+St))
        if z_expansion == 'G':
            hd[ibin,:] = hgas
                                       
    # dust aspect ratio as function of ibin, r and phi (2D array for each size bin)
    hd2D = np.zeros((nbin,nrad,nsec))
    for th in range(nsec):
        hd2D[:,:,th] = hd    # nbin, nrad, nsec

    # grid radius function of ibin, r and phi (2D array for each size bin)
    r2D = np.zeros((nbin,nrad,nsec))
    for ibin in range(nbin):
        for th in range(nsec):
            r2D[ibin,:,th] = gas.xmed

    # work out exponential and normalization factors exp(-z^2 / 2H_d^2)
    # with z = r cos(theta) and H_d = h_d x R = h_d x r sin(theta)
    # r = spherical radius, R = cylindrical radius
    for j in range(ncol):
        rhodustcube[j,:,:,:] = dustcube * np.exp( -0.5*(np.cos(gas.ymed[j]) / hd2D)**2.0 )   # ncol, nbin, nrad, nsec
        rhodustcube[j,:,:,:] /= ( np.sqrt(2.*pi) * r2D * hd2D  * gas.culength*1e2 )          # quantity is now in g / cm^3

    # for plotting purposes
    axirhodustcube = np.sum(rhodustcube,axis=3)/nsec  # ncol, nbin, nrad
    
    # Renormalize dust's mass volume density such that the sum over the 3D grid's volume of
    # the dust's mass volume density x the volume of each grid cell does give us the right
    # total dust mass, which equals ratio x Mgas.
    rhofield = np.sum(rhodustcube, axis=1)  # sum over dust bins
    Redge,Cedge,Aedge = np.meshgrid(gas.xm, gas.ym, gas.zm)   # ncol+1, nrad+1, Nsec+1
    r2 = Redge*Redge
    jacob  = r2[:-1,:-1,:-1] * np.sin(Cedge[:-1,:-1,:-1])
    dphi   = Aedge[:-1,:-1,1:] - Aedge[:-1,:-1,:-1]     # same as 2pi/nsec
    dr     = Redge[:-1,1:,:-1] - Redge[:-1,:-1,:-1]     # same as Rsup-Rinf
    dtheta = Cedge[1:,:-1,:-1] - Cedge[:-1,:-1,:-1]
    # volume of a cell in cm^3
    vol = jacob * dr * dphi * dtheta * ((gas.culength*1e2)**3)       # ncol, nrad, Nsec
    total_mass = np.sum(rhofield*vol)
    normalization_factor =  ratio * Mgas * (gas.cumass*1e3) / total_mass
    rhodustcube *= normalization_factor
    print('total dust mass after vertical expansion [g] = ', np.sum(np.sum(rhodustcube, axis=1)*vol), ' as normalization factor = ', normalization_factor)
    
    # write mass volume densities for all size bins
    for ibin in range(nbin):
        print('dust species in bin', ibin, 'out of ',nbin-1)
        for k in range(nsec):
            for j in range(ncol):
                for i in range(nrad):
                    DUSTOUT.write(str(rhodustcube[j,ibin,i,k])+' \n')

    # print max of dust's mass volume density at each colatitude
    for j in range(ncol):
        print('max(rho_dustcube) for colatitude index j = ', j, ' = ', rhodustcube[j,:,:,:].max())

    DUSTOUT.close()

    # free RAM memory
    del rhodustcube, dustcube, dust, hd2D, r2D

    # Write 3D spherical grid for RT computational calculation
    write_AMRgrid(gas, Plot=False)
else:
    print('--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------')


# =========================
# 4. Compute dust opacities
# =========================
if recalc_opac == 'Yes':
    print('--------- computing dust opacities ----------')

    # Calculation of opacities uses the python scripts makedustopac.py and bhmie.py
    # which were written by C. Dullemond, based on the original code by Bohren & Huffman.
    
    logawidth = 0.05          # Smear out the grain size by 5% in both directions
    na        = 20            # Use 10 grain size samples per bin size
    chop      = 5.            # Remove forward scattering within an angle of 5 degrees
    extrapol  = True          # Extrapolate optical constants beyond its wavelength grid, if necessary
    verbose   = False         # If True, then write out status information
    ntheta    = 999           # Number of scattering angle sampling points
    optconstfile = os.path.expanduser(opacity_dir)+'/'+species+'.lnk'    # link to optical constants file

    # The material density in gram / cm^3
    graindens = 2.0 # default density in g / cc
    if (species == 'mix_2species_porous' or species == 'mix_2species_porous_ice' or species == 'mix_2species_porous_ice70'):
        graindens = 0.1 # g / cc
    if species == 'mix_2species':
        graindens = 1.7 # g / cc
    if species == 'mix_2species_ice70':
        graindens = 1.26 # g / cc
    if species == 'mix_2species_60silicates_40carbons':
        graindens = 2.7 # g / cc
    
    # Set up a wavelength grid (in cm) upon which we want to compute the opacities
    # 1 micron -> 1 cm
    lamcm     = 10.0**np.linspace(0,4,200)*1e-4   

    # Set up an angular grid for which we want to compute the scattering matrix Z
    theta     = np.linspace(0.,180.,ntheta)

    for ibin in range(int(nbin)):
        # median grain size in cm in current bin size:
        agraincm   = 10.0**(0.5*(np.log10(1e2*bins[ibin]) + np.log10(1e2*bins[ibin+1])))
        print('====================')
        print('bin ', ibin+1,'/',nbin)
        print('grain size [cm]: ', agraincm, ' with grain density [g/cc] = ', graindens)
        print('====================')
        pathout    = species+str(ibin)
        opac       = compute_opac_mie(optconstfile,graindens,agraincm,lamcm,theta=theta,
                                      extrapolate=extrapol,logawidth=logawidth,na=na,
                                      chopforward=chop,verbose=verbose)
        if (scat_mode >= 3):
            print("Writing dust opacities in dustkapscatmat* files")
            write_radmc3d_scatmat_file(opac,pathout)
        else:
            print("Writing dust opacities in dustkappa* files")
            write_radmc3d_kappa_file(opac,pathout)
else:
    print('------- taking dustkap* opacity files in current directory (recalc_opac = No in params.dat file) ------ ')

# Write dustopac.inp file even if we don't (re)calculate dust opacities
write_dustopac(species,nbin)
if plot_opac == 'Yes':
    print('--------- plotting dust opacities ----------')
    plot_opacities(species=species,amin=amin,amax=amax,nbin=nbin,lbda1=wavelength*1e3)

# write radmc3d script in case radmc3d mctherm / ray_tracing are run on a different platform
write_radmc3d_script()


# =========================
# 5. Call to RADMC3D thermal solution and ray tracing 
# =========================
if (recalc_radmc == 'Yes' or recalc_rawfits == 'Yes'):
    # Write other parameter files required by RADMC3D
    print('--------- printing auxiliary files ----------')

    # need to check why we need to output wavelength...
    if recalc_rawfits == 'No':
        write_wavelength()
        write_stars(Rstar = rstar, Tstar = teff)

    # rto_style = 3 means that RADMC3D will write binary output files
    # setthreads corresponds to the number of threads (cores) over which radmc3d runs
    if recalc_rawfits == 'No':
        write_radmc3dinp(nphot_scat=nb_photons_scat, nphot=nb_photons, rto_style=3, 
                         modified_random_walk=1, scattering_mode_max=scat_mode, setthreads=nbcores)
    
    # Add 90 degrees to position angle so that RADMC3D's definition of
    # position angle be consistent with observed position
    # angle, which is what we enter in the params.dat file
    M = RTmodel(distance=distance, Lambda=wavelength*1e3, label=label,
                npix=nbpixels, phi=phiangle, incl=inclination, posang=posangle+90.0) 

    # Get dust temperature
    if recalc_rawfits == 'No':
        print('--------- executing Monte-Carlo thermal calculation with RADMC3D ----------')
        run_mctherm()
    
    # Run ray tracing
    if recalc_rawfits == 'No':
        print('--------- executing ray tracing with RADMC3D ----------')    
        run_raytracing(M)

    print('--------- exporting results in fits format ----------')
    outfile = exportfits(M)

    if plot_temperature == 'Yes':
        # Plot midplane and surface temperature profiles
        Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
        Temp = Temp[4:]
        Temp = Temp.reshape(nbin,nsec,ncol,nrad)
        # Keep temperature of the largest dust species
        Temp = Temp[-1,:,:,:]
        # Temperature in the midplane (ncol/2 given that the grid extends on both sides about the midplane)
        # not really in the midplane because theta=pi/2 is an edge colatitude...
        Tm = Temp[:,ncol//2,:]
        # Temperature at one surface
        Ts = Temp[:,0,:]
        # Azimuthally-averaged radial profiles
        axiTm = np.sum(Tm,axis=0)/nsec
        axiTs = np.sum(Ts,axis=0)/nsec
        fig = plt.figure(figsize=(4.,3.))
        ax = fig.gca()
        S = gas.xmed*gas.culength/1.5e11  # radius in a.u.
        # gas temperature in hydro simulation in Kelvin (assuming T in R^-1/2, no matter
        # the value of the gas flaring index in the simulation)
        Tm_model = aspectratio*aspectratio*gas.cutemp*gas.xmed**(-1.0+2.0*flaringindex)
        ax.plot(S, axiTm, 'bo', markersize=1., label='midplane')
        ax.plot(S, Tm_model, 'b--', markersize=1., label='midplane hydro')
        ax.plot(S, axiTs, 'rs', markersize=1., label='surface')
        ax.set_xlabel(r'$R ({\rm au})$', fontsize=12)
        ax.set_ylabel(r'$T ({\rm K})$', fontsize=12)
        ax.set_xlim(20.0, 100.0) # cuidadin!
        #ax.set_xlim(S.min(), S.max())
        ax.set_ylim(10.0, 150.0)  # cuidadin!
        #ax.set_ylim(Tm.min(), Ts.max())
        ax.tick_params(axis='both', direction='in', top='on', right='on')
        ax.tick_params(axis='both', which='minor', top='on', right='on', direction='in')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(frameon=False)
        fig.add_subplot(ax)
        filenameT = 'T_R_'+label+'.pdf'
        fig.savefig(filenameT, dpi=180, bbox_inches='tight')
        fig.clf()
        # Save radial profiles in an ascii file
        filenameT2 = 'T_R_'+label+'.dat'
        TEMPOUT=open(filenameT2,'w')
        TEMPOUT.write('# radius [au] \t T_midplane_radmc3d \t T_surface_radmc3d \t T_midplane_hydro\n')
        for i in range(nrad):
            TEMPOUT.write('%f \t %f \t %f \t %f\n' %(S[i],axiTm[i],axiTs[i],Tm_model[i]))
        TEMPOUT.close()
            # free RAM memory
        del Temp
else:
    print('------- I did not run RADMC3D, using existing .fits file for convolution ')
    print('------- (recalc_radmc = No in params.dat file) and final image ------ ')
    outfile = 'image_'+str(label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)
    if secondorder == 'Yes':
        outfile = outfile+'_so'
    if dustdens_eq_gasdens == 'Yes':
        outfile = outfile+'_ddeqgd'
    if bin_small_dust == 'Yes':
        outfile = outfile+'_bin0'
    outfile = outfile+'.fits'


# =========================
# 6. Convolve raw flux with beam and produce final image
# =========================
if recalc_fluxmap == 'Yes':
    print('--------- Convolving and writing final image ----------')

    f = fits.open('./'+outfile)
    hdr = f[0].header
    # pixel size converted from degrees to arcseconds
    cdelt = np.abs(hdr['CDELT1']*3600.)

    # a) case with no polarized scattering: fits file directly contains raw intensity field
    if polarized_scat == 'No':
        nx = hdr['NAXIS1']
        ny = hdr['NAXIS2']
        raw_intensity = f[0].data
        if (recalc_radmc == 'No' and plot_tau == 'No'):
            print("Total flux [Jy] = "+str(np.sum(raw_intensity)))   # sum over pixels
        # check beam is correctly handled by inserting a source point at the
        # origin of the raw intensity image
        if check_beam == 'Yes':
            raw_intensity[:,:] = 0.0
            raw_intensity[nx//2-1,ny//2-1] = 1.0
        # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
        if (add_noise == 'Yes' and plot_tau == 'No'):
            # beam area in pixel^2
            beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
            # noise standard deviation in Jy per pixel (I've checked the expression below works well)
            noise_dev_std_Jy_per_pixel = noise_dev_std / np.sqrt(0.5*beam)  # 1D
            # noise array
            noise_array = np.random.normal(0.0,noise_dev_std_Jy_per_pixel,size=nbpixels*nbpixels)
            noise_array = noise_array.reshape(nbpixels,nbpixels)
            raw_intensity += noise_array

    # b) case with polarized scattering: fits file contains raw Stokes vectors
    if polarized_scat == 'Yes':
        cube = f[0].data        
        Q = cube[1,:,:]
        U = cube[2,:,:]
        #I = cube[0,:,:]
        #P = cube[4,:,:]
        (nx, ny) = Q.shape
        # define theta angle for calculation of Q_phi below (Avenhaus+ 14)
        x = np.arange(1,nx+1)
        y = np.arange(1,ny+1)
        XXs,YYs = np.meshgrid(x,y)
        X0 = nx/2-1
        Y0 = ny/2-1
        rrs = np.sqrt((XXs-X0)**2+(YYs-Y0)**2)
        theta = np.arctan2(-(XXs-X0),(YYs-Y0)) # notice atan(x/y)
        if add_noise == 'Yes':
            # add noise to Q and U Stokes arrays
            # noise array
            noise_array_Q = np.random.normal(0.0,0.004*Q.max(),size=nbpixels*nbpixels)
            noise_array_Q = noise_array_Q.reshape(nbpixels,nbpixels)
            Q += noise_array_Q
            noise_array_U = np.random.normal(0.0,0.004*U.max(),size=nbpixels*nbpixels)
            noise_array_U = noise_array_U.reshape(nbpixels,nbpixels)
            U += noise_array_U 
        # add mask in polarized intensity Qphi image if mask_radius != 0
        if mask_radius != 0.0:
            pillbox = np.ones((nx,ny))
            imaskrad = mask_radius/cdelt  # since cdelt is pixel size in arcseconds
            pillbox[np.where(rrs<imaskrad)] = 0.
    
    # ------------
    # smooth image
    # ------------
    # beam area in pixel^2
    beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
    # stdev lengths in pixel
    stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
    stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / cdelt

    # a) case with no polarized scattering
    if (polarized_scat == 'No' and plot_tau == 'No'):
        # Call to Gauss_filter function
        smooth = Gauss_filter(raw_intensity, stdev_x, stdev_y, bpaangle, Plot=False)
        # convert image from Jy/pixel to mJy/beam or microJy/beam
        # could be refined...
        convolved_intensity = smooth * 1e3 * beam   # mJy/beam
        strflux = 'mJy/beam'
        if convolved_intensity.max() < 1.0:
            convolved_intensity = smooth * 1e6 * beam   # microJy/beam
            strflux = '$\mu$Jy/beam'
    if plot_tau == 'Yes':
        convolved_intensity = raw_intensity
        strflux = '$\tau'

    # b) case with polarized scattering
    if polarized_scat == 'Yes':       
        Q_smooth = Gauss_filter(Q,stdev_x,stdev_y,bpaangle,Plot=False)
        U_smooth = Gauss_filter(U,stdev_x,stdev_y,bpaangle,Plot=False)
        if mask_radius != 0.0:
            pillbox_smooth = Gauss_filter(pillbox, stdev_x, stdev_y, bpaangle, Plot=False)
            Q_smooth *= pillbox_smooth
            U_smooth *= pillbox_smooth
        Q_phi = Q_smooth * np.cos(2*theta) + U_smooth * np.sin(2*theta)
        convolved_intensity = Q_phi
        strflux = 'arb. units'

    # -------------------------------------
    # SP: save convolved flux map solution to fits 
    # -------------------------------------
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32    
    hdu.header['NAXIS'] = 2  # 2
    hdu.header['NAXIS1'] = nbpixels
    hdu.header['NAXIS2'] = nbpixels
    #hdu.header['NAXIS3'] = 1
    #hdu.header['NAXIS4'] = 1
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    #hdu.header['CTYPE3'] = 'FREQ'
    #hdu.header['CTYPE4'] = 'STOKES'
    hdu.header['CRVAL1'] = 8.261472379700E+01 # float(0.0)
    hdu.header['CRVAL2'] = 2.533239051468E+01 # float(0.0)
    #hdu.header['CRVAL3'] = 33.0E+09
    #hdu.header['CRVAL4'] = 1.0E+00
    hdu.header['CDELT1'] = hdr['CDELT1']
    hdu.header['CDELT2'] = hdr['CDELT2']
    #hdu.header['CDELT3'] = 8.05E+09
    #hdu.header['CDELT4'] = 1.0E+00
    hdu.header['CUNIT1'] = 'deg     '
    hdu.header['CUNIT2'] = 'deg     '
    #hdu.header['CUNIT3'] = 'Hz     '
    #hdu.header['CUNIT4'] = ' '
    hdu.header['CRPIX1'] = float((nbpixels+1.)/2.)
    hdu.header['CRPIX2'] = float((nbpixels+1.)/2.)
    if strflux == 'mJy/beam':
        hdu.header['BUNIT'] = 'milliJY/BEAM'
    if strflux == '$\mu$Jy/beam':
        hdu.header['BUNIT'] = 'microJY/BEAM'
    if strflux == '':
        hdu.header['BUNIT'] = ''
    hdu.header['BTYPE'] = 'FLUX DENSITY'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0
    del hdu.header['EXTEND']
    # keep track of all parameters in params.dat file
    #for i in range(len(lines_params)):
    #    hdu.header[var[i]] = par[i]
    hdu.data = convolved_intensity
    inbasename = os.path.basename('./'+outfile)
    if add_noise == 'Yes':
        jybeamfileout=re.sub('.fits', '_wn_JyBeam.fits', inbasename)
    else:
        jybeamfileout=re.sub('.fits', '_JyBeam.fits', inbasename)
    if polarized_scat == 'Yes':
        jybeamfileout = 'Qphi.fits'
    hdu.writeto(jybeamfileout, overwrite=True)

    # ----------------------------
    # if polarised imaging, first de-project Qphi image to multiply by R^2
    # then re-project back
    # ----------------------------
    if polarized_scat == 'Yes':
        hdu0 = fits.open(jybeamfileout)
        hdr0 = hdu0[0].header
        nx = int(hdr0['NAXIS1'])
        ny = nx
        if ( (nx % 2) == 0):
            nx = nx+1
            ny = ny+1
        hdr1 = deepcopy(hdr0)    
        hdr1['NAXIS1']=nx
        hdr1['NAXIS2']=ny
        hdr1['CRPIX1']=(nx+1)/2
        hdr1['CRPIX2']=(ny+1)/2
        
        # slightly modify original image such that centre is at middle of image -> odd number of cells
        image_centered = gridding(jybeamfileout,hdr1,fullWCS=False)
        fileout_centered = re.sub('.fits', 'centered.fits', jybeamfileout)
        fits.writeto(fileout_centered, image_centered, hdr1, overwrite=True)
        
        # rotate original, centred image by position angle (posangle)
        image_rotated = ndimage.rotate(image_centered, posangle, reshape=False)
        fileout_rotated = re.sub('.fits', 'rotated.fits', jybeamfileout)
        fits.writeto(fileout_rotated, image_rotated, hdr1, overwrite=True)
        hdr2 = deepcopy(hdr1)
        cosi = np.cos(inclination_input*np.pi/180.)
        hdr2['CDELT1']=hdr2['CDELT1']*cosi

        # Then deproject with inclination via gridding interpolation function and hdr2
        image_stretched = gridding(fileout_rotated,hdr2)

         # rescale stretched image by r^2
        nx = hdr2['NAXIS1']
        ny = hdr2['NAXIS2']
        cdelt = abs(hdr2['CDELT1']*3600)  # in arcseconds
        (x0,y0) = (nx/2, ny/2)
        mymax = 0.0
        for j in xrange(nx):
            for k in xrange(ny):
                dx = (j-x0)*cdelt
                dy = (k-y0)*cdelt
                rad = np.sqrt(dx*dx + dy*dy)
                image_stretched[j,k] *= (rad*rad)
                if (rad <= truncation_radius and image_stretched[j,k] > mymax):
                    mymax = image_stretched[j,k]
                #else:
                #    image_stretched[j,k] = 0.0
                    
                    
        # Normalize PI intensity
        image_stretched /= mymax
        fileout_stretched = re.sub('.fits', 'stretched.fits', jybeamfileout)
        fits.writeto(fileout_stretched, image_stretched, hdr2, overwrite=True)

        # Then deproject via gridding interpolatin function and hdr1
        image_destretched = gridding(fileout_stretched,hdr1)

        # and finally de-rotate by -position angle
        final_image = ndimage.rotate(image_destretched, -posangle, reshape=False)

        # save final fits
        inbasename = os.path.basename('./'+outfile)
        if add_noise == 'Yes':
            jybeamfileout=re.sub('.fits', '_wn_JyBeam.fits', inbasename)
        else:
            jybeamfileout=re.sub('.fits', '_JyBeam.fits', inbasename)
        fits.writeto(jybeamfileout,final_image,hdr1,clobber=True)
        convolved_intensity = final_image
        #os.system('rm -f Qphi*.fits')

    
    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial') 
    fontcolor='white'

    # name of pdf file for final image
    fileout = re.sub('.fits', '.pdf', jybeamfileout)
    fig = plt.figure(figsize=(8.,8.))
    ax = plt.gca()
    plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.09)

    # Set x-axis orientation, x- and y-ranges
    # Convention is that RA offset increases leftwards (ie,
    # east is to the left), while Dec offset increases from
    # bottom to top (ie, north is the top)
    if ( (nx % 2) == 0):
        dpix = 0.5
    else:
        dpix = 0.0
    a0 = cdelt*(nx//2.-dpix)   # >0
    a1 = -cdelt*(nx//2.+dpix)  # <0
    d0 = -cdelt*(nx//2.-dpix)  # <0
    d1 = cdelt*(nx//2.+dpix)   # >0
    # da positive definite
    if (minmaxaxis < abs(a0)):
        da = minmaxaxis
    else:
        da = abs(a0)
    ax.set_xlim(da,-da)      # x (=R.A.) increases leftward
    mina = da
    maxa = -da
    xlambda = mina - 0.166*da
    ax.set_ylim(-da,da)
    dmin = -da
    dmax = da

    # x- and y-ticks and labels
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian 
    ax.set_xlabel('RA offset [arcsec]')
    ax.set_ylabel('Dec offset [arcsec]')

    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    min = convolved_intensity.min()
    max = convolved_intensity.max()
    if polarized_scat == 'Yes':
        min = 0.0
        max = 1.0  # cuidadin 1.0
    CM = ax.imshow(convolved_intensity, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[a0,a1,d0,d1], vmin=min, vmax=max)

    # Add wavelength in top-left corner
    strlambda = str(wavelength)+' mm (model)'
    if wavelength < 0.01:
        strlambda = str(wavelength*1e3)+'$\mu$m (model)'
    ax.text(xlambda,dmax-0.166*da,strlambda, fontsize=20, color = 'white',weight='bold')

    # Add + sign at the origin
    ax.plot(0.0,0.0,'+',color='white',markersize=10)
    '''
    if check_beam == 'Yes':
        ax.contour(convolved_intensity,levels=[0.5*convolved_intensity.max()],color='black', linestyles='-',origin='lower',extent=[a0,a1,d0,d1])
    '''
    
    # plot beam
    if plot_tau == 'No':
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=[xlambda,dmin+0.166*da], width=bmin, height=bmaj, angle=bpaangle+90.0)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('white')
        e.set_alpha(0.8)
        ax.add_artist(e)
    # plot beam
    '''
    if check_beam == 'Yes':
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=[0.0,0.0], width=bmin, height=bmaj, angle=bpaangle+90.0)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('white')
        e.set_alpha(1.0)
        ax.add_artist(e)
    '''
        
    # plot color-bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4.5%", pad=0.3)
    cax.yaxis.set_ticks_position('left')
    cax.xaxis.set_ticks_position('top')
    cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.yaxis.set_tick_params(labelsize=20, direction='out')
    cax.text(.990, 0.22, strflux, fontsize=20, horizontalalignment='right', color='white', weight='bold')

    plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
    plt.clf()

    # =====================
    # Compute deprojection and polar expansion (SP)
    # =====================
    if deproj_polar == 'Yes':
        currentdir = os.getcwd()
        alpha_min = 0.;          # deg, PA of offset from the star
        Delta_min = 0.;          # arcsec, amplitude of offset from the star
        RA = 0.0  # if input image is a prediction, star should be at the center
        DEC = 0.0 # note that this deprojection routine works in WCS coordinates
        cosi = np.cos(inclination_input*np.pi/180.)

        print('deprojection around PA [deg] = ',posangle)
        print('and inclination [deg] = ',inclination_input)
        
        # makes a new directory "deproj_polar_dir" and calculates a number
        # of products: copy of the input image [_fullim], centered at
        # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
        # image [_polar], etc. Also, a _radial_profile which is the
        # average radial intensity.
        exec_polar_expansions(jybeamfileout,'deproj_polar_dir',posangle,cosi,RA=RA,DEC=DEC,
                              alpha_min=alpha_min, Delta_min=Delta_min,
                              XCheckInv=False,DoRadialProfile=False,
                              DoAzimuthalProfile=False,PlotRadialProfile=False,
                              a_min=0.02,a_max=1.0,zoomfactor=1.)
        
        # Save polar fits in current directory
        fileout = re.sub('.pdf', '_polar.fits', fileout)
        command = 'cp deproj_polar_dir/'+fileout+' .'
        os.system(command)

        filein = re.sub('.pdf', '_polar.fits', fileout)
        # Read fits file with deprojected field in polar coordinates
        f = fits.open(filein)
        convolved_intensity = f[0].data    # uJy/beam

        # azimuthal shift such that PA=0 corresponds to y-axis pointing upwards, and
        # increases counter-clockwise from that axis
        if xaxisflip == 'Yes':
            jshift = int(nbpixels/4)
        else:
            jshift = int(nbpixels/4)
        convolved_intensity = np.roll(convolved_intensity, shift=-jshift, axis=1)
        
        # -------------------------------
        # plot image in polar coordinates
        # -------------------------------
        fileout = re.sub('.fits', '.pdf', filein)
        fig = plt.figure(figsize=(8.,8.))
        ax = plt.gca()
        plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.09)

        # Set x- and y-ranges
        ax.set_xlim(-180,180)          # PA relative to Clump 1's
        ax.set_ylim(0,minmaxaxis)      # Deprojected radius in arcsec
        if ( (nx % 2) == 0):
            dpix = 0.5
        else:
            dpix = 0.0
        a0 = cdelt*(nx//2.-dpix)   # >0

        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.set_xticks((-180,-120,-60,0,60,120,180))
        #ax.set_yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))
        ax.set_xlabel('Position Angle [deg]')
        ax.set_ylabel('Radius [arcsec]')

        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = convolved_intensity.min()  # not exactly same as 0
        max = convolved_intensity.max()
        CM = ax.imshow(convolved_intensity, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[-180,180,0,a0], vmin=min, vmax=max, aspect='auto')   # (left, right, bottom, top)

        # Add wavelength in bottom-left corner
        strlambda = str(wavelength)+' mm (model)'
        if wavelength < 0.01:
            strlambda = str(wavelength*1e3)+'$\mu$m (model)'
        ax.text(60,0.02,strlambda,fontsize=16,color='white',weight='bold')

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="4.5%", pad=0.3)
        cax.yaxis.set_ticks_position('left')
        cax.xaxis.set_ticks_position('top')
        cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        cax.yaxis.set_tick_params(labelsize=20, direction='out')
        cax.text(.990, 0.22, strflux, fontsize=20, horizontalalignment='right', color='white', weight='bold')

        plt.savefig('./'+fileout, dpi=160)
        plt.clf()

        os.system('rm -rf deproj_polar_dir')
        os.chdir(currentdir)


# =========================
# 7. Compute 2D analytical solution to RT equation w/o scattering
# =========================
if calc_abs_map == 'Yes':
    print('--------- Computing 2D analytical solution to RT equation w/o scattering ----------')

    # ---------------------------
    # a) assume dust surface density != gas surface density (default case)
    # ---------------------------
    if dustdens_eq_gasdens == 'No':
        # read information on the dust particles
        (rad, azi, vr, vt, Stokes, a) = np.loadtxt(dir+'/dustsystat'+str(on)+'.dat', unpack=True)

        # We need to recompute dustcube again as we sweeped it off the memory before. Since it
        # is now quick to compute it, we simply do it again!
        # Populate dust bins
        dust = np.zeros((nsec*nrad*nbin))
        for m in range (len(a)):   # CB: sum over dust particles
            r = rad[m]
            t = azi[m]
            # radial index of the cell where the particle is
            if radialspacing == 'L':
                i = int(np.log(r/gas.xm.min())/np.log(gas.xm.max()/gas.xm.min()) * nrad)
            else:
                i = np.argmin(np.abs(gas.xm-r))
            if (i < 0 or i >= nrad):
                sys.exit('pb with i = ', i, ' in calc_abs_map step: I must exit!')
            # azimuthal index of the cell where the particle is
            # (general expression since grid spacing in azimuth is always arithmetic)
            j = int((t-gas.zm.min())/(gas.zm.max()-gas.zm.min()) * nsec)
            if (j < 0 or j >= nsec):
                sys.exit('pb with j = ', j, ' in calc_abs_map step: I must exit!')
            # particle size
            pcsize = a[m]   
            # find out which bin particle belongs to. Here we do nearest-grid point. We
            # could also do a bilinear interpolation!
            ibin = int(np.log(pcsize/bins.min())/np.log(bins.max()/bins.min()) * nbin)
            if (ibin >= 0 and ibin < nbin):
                k = ibin*nsec*nrad + j*nrad + i
                dust[k] +=1
                nparticles[ibin] += 1
                avgstokes[ibin] += Stokes[m]

        for ibin in range(nbin):
            if nparticles[ibin] == 0:
                nparticles[ibin] = 1
            avgstokes[ibin] /= nparticles[ibin]
            print(str(nparticles[ibin])+' grains between '+str(bins[ibin])+' and '+str(bins[ibin+1])+' meters')
            
        # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
        dustcube = dust.reshape(nbin, nsec, nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec

        # Mass of gas in units of the star's mass
        Mgas = np.sum(gas.data*surface)
        print('Mgas / Mstar= '+str(Mgas)+' and Mgas [kg] = '+str(Mgas*gas.cumass))

        frac = np.zeros(nbin)
        # finally compute dust surface density for each size bin
        for ibin in range(nbin):
            # fraction of dust mass in current size bin 'ibin', easy to check numerically that sum_frac = 1
            frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin]**(4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
            # total mass of dust particles in current size bin 'ibin'
            M_i_dust = ratio * Mgas * frac[ibin]
            # dustcube, which contained N_i(r,phi), now contains sigma_i_dust (r,phi)
            dustcube[ibin,:,:] *= M_i_dust / surface / nparticles[ibin]
            # conversion in g/cm^2
            dustcube[ibin,:,:] *= (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec

        # Overwrite first bin (ibin = 0) to model extra bin with small dust tightly coupled to the gas
        if bin_small_dust == 'Yes':
            frac[0] *= 5e3
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Bin with index 0 changed to include arbitrarilly small dust tightly coupled to the gas")
            print("Mass fraction of bin 0 changed to: ",str(frac[0]))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            imin = np.argmin(np.abs(gas.xmed-1.2))  # radial index corresponding to 0.25"
            imax = np.argmin(np.abs(gas.xmed-3.0))  # radial index corresponding to 0.6"
            dustcube[0,imin:imax,:] = gas.data[imin:imax,:] * ratio * frac[0] * (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec

    # ---------------------------
    # b) assume dust surface density = gas surface density
    # ---------------------------
    if dustdens_eq_gasdens == 'Yes':
        dust = np.zeros((nsec*nrad*nbin))
        # dustcube currently contains N_i (r,phi), the number of particles per bin size in every grid cell
        dustcube = dust.reshape(nbin, nsec, nrad)  
        dustcube = np.swapaxes(dustcube,1,2)  # means nbin, nrad, nsec
        frac = np.zeros(nbin)
        for ibin in range(nbin):
            frac[ibin] = (bins[ibin+1]**(4.0-pindex) - bins[ibin]**(4.0-pindex)) / (amax**(4.0-pindex) - amin**(4.0-pindex))
            dustcube[ibin,:,:] = gas.data * ratio * frac[ibin] * (gas.cumass*1e3)/((gas.culength*1e2)**2.)  # dimensions: nbin, nrad, nsec
        
    # We then need to recompute absorption mass opacity from dustkappa files
    abs_opacity = np.zeros(nbin)
    lbda1 = wavelength * 1e3  # wavelength in microns
    if precalc_opac == 'Yes':
        opacdir = os.path.expanduser(opacity_dir)
        # Case where we use pre-calculated dustkappa* files located in opacity_dir files
        # whatever the composition, precomputed opacities are for dust sizes between 1 microns and 10 cm, with 50 bins
        sizemin_file = 1e-6          # in meters, do not edit!
        sizemax_file = 1e-1          # in meters, do not edit!
        nbfiles = 50                 # do not edit
        else:
            sys.exit('I do not have pre-calculated opacity files for your type of species in the opacity_dir: I must exit!')
        size_file = sizemin_file * (sizemax_file/sizemin_file)**(np.arange(nbfiles)/(nbfiles-1.0))
    # Loop over size bins
    for k in range(nbin):
        if precalc_opac == 'No':
            # Case where we use dustkappa* files in current directory
            file = 'dustkappa_'+species+str(k)+'.inp'
            (lbda, kappa_abs, kappa_sca, g) = read_opacities(file)
            #(lbda, kappa_abs, kappa_sca, g) = np.loadtxt(file, unpack=True, skiprows=2)
            i1 = np.argmin(np.abs(lbda-lbda1))
            # linear interpolation (in log)
            l1 = lbda[i1-1]
            l2 = lbda[i1+1]
            k1 = kappa_abs[i1-1]
            k2 = kappa_abs[i1+1]
            abs_opacity[k] = (k1*np.log(l2/lbda1) + k2*np.log(lbda1/l1))/np.log(l2/l1)
            print('absorption opacity [cm^2/g] of bin ',k,' with average size ', bins[k], ' = ',abs_opacity[k])
        else:
            # Case where we use pre-calculated dustkappa* files in opacity_dir directory
            index_inf = int(np.log(bins[k]/sizemin_file)/np.log(sizemax_file/sizemin_file) * nbfiles)
            if (index_inf < nbfiles-1):
                index_sup = index_inf+1
                file_index_inf = opacdir+'/dustkappa_'+species+str(index_inf)+'.inp'
                file_index_sup = opacdir+'/dustkappa_'+species+str(index_sup)+'.inp'
                (lbda_inf, kappa_abs_inf, kappa_sca_inf, g_inf) = read_opacities(file_index_inf)
                #np.loadtxt(file_index_inf, unpack=True, skiprows=2)
                (lbda_sup, kappa_abs_sup, kappa_sca_sup, g_sup) = read_opacities(file_index_sup)
                #np.loadtxt(file_index_sup, unpack=True, skiprows=2)
                i1_inf = np.argmin(np.abs(lbda_inf-lbda1))
                l1 = lbda_inf[i1_inf-1]
                l2 = lbda_inf[i1_inf+1]
                k1_inf = kappa_abs_inf[i1_inf-1]
                k2_inf = kappa_abs_inf[i1_inf+1]
                i1_sup = np.argmin(np.abs(lbda_sup-lbda1))
                k1_sup = kappa_abs_sup[i1_sup-1]
                k2_sup = kappa_abs_sup[i1_sup+1]
                abs_opacity[k] = k1_inf*np.log(l2/lbda1)*np.log(size_file[index_sup]/bins[k]) \
                    + k2_inf*np.log(lbda1/l1)*np.log(size_file[index_sup]/bins[k]) \
                    + k1_sup*np.log(l2/lbda1)*np.log(bins[k]/size_file[index_inf]) \
                    + k2_sup*np.log(lbda1/l1)*np.log(bins[k]/size_file[index_inf])
                abs_opacity[k] /= ( np.log(l2/l1) * np.log(size_file[index_sup]/size_file[index_inf]) )
            if (index_inf == nbfiles-1):
                file_index_inf = opacdir+'/dustkappa_'+species+str(index_inf)+'.inp'
                (lbda_inf, kappa_abs_inf, kappa_sca_inf, g_inf) = read_opacities(file_index_inf)
                #np.loadtxt(file_index_inf, unpack=True, skiprows=2)
                i1_inf = np.argmin(np.abs(lbda_inf-lbda1))
                l1 = lbda_inf[i1_inf-1]
                l2 = lbda_inf[i1_inf+1]
                k1_inf = kappa_abs_inf[i1_inf-1]
                k2_inf = kappa_abs_inf[i1_inf+1]
                abs_opacity[k] = (k1_inf*np.log(l2/lbda1) + k2_inf*np.log(lbda1/l1))/np.log(l2/l1)
            # distinguish cases where index_inf = nbfiles-1 and index_inf >= nbfiles (extrapolation)
            print('absorption opacity (pre-calc) [cm^2/g] of bin ',k,' with average size ', bins[k], ' = ',abs_opacity[k])

    # kappa_abs as function of ibin, r and phi (2D array for each size bin)
    abs_opacity_2D = np.zeros((nbin,nrad,nsec))
    for i in range(nrad):
        for j in range(nsec):
            abs_opacity_2D[:,i,j] = abs_opacity    # nbin, nrad, nsec

    # Infer optical depth array
    optical_depth = np.zeros(gas.data.shape)  # 2D array containing tau at each grid cell
    optical_depth = np.sum(dustcube*abs_opacity_2D,axis=0)    # nrad, nsec
    # divide by cos(inclination) since integral over ds = cos(i) x integral over dz
    optical_depth /= np.abs(np.cos(inclination*np.pi/180.0))  
    optical_depth = optical_depth.reshape(nrad,nsec) 
    optical_depth = np.swapaxes(optical_depth,0,1)  # means nsec, nrad

    print('max(optical depth) = ', optical_depth.max())

    # Get dust temperature
    if Tdust_eq_Tgas == 'No':
        Temp = np.fromfile('dust_temperature.bdat', dtype='float64')
        Temp = Temp[4:]
        Temp = Temp.reshape(nbin,nsec,ncol,nrad)
        # Keep temperature of the largest dust species
        Temp = Temp[-1,:,:,:]
        # Temperature in the midplane (ncol/2 given that the grid extends on both sides about the midplane)
        # is it the midplane temperature that should be adopted? or a vertically averaged temperature?
        Tdust = Temp[:,ncol//2,:]+0.1  # avoid division by zero afterwards  (nsec, nrad)
        # Free RAM memory
        del Temp
    else:
        Tdust = np.zeros(gas.data.shape)
        Tdust = np.swapaxes(Tdust,0,1)   # means nsec, nrad
        T_model = aspectratio*aspectratio*gas.cutemp*gas.xmed**(-1.0+2.0*flaringindex)
        for j in range(nsec):
            Tdust[j,:] = T_model

    # Now get Bnu(Tdust)
    # Frequency in Hz; wavelength is currently in mm
    nu = c / (wavelength*1e-1)
    # 2D array containing Bnu(Tdust) at each grid cell
    Bnu = np.zeros(gas.data.shape)
    Bnu = 2.0*h*(nu**3.0)/c/c/(np.exp(h*nu/kB/Tdust)-1.0)  # in cgs: erg cm-2 sterad-1 = g s-2 ster-1
    # Specific intensity on the simulation's polar grid in Jy/steradian:
    Inu_polar = Bnu * (1.0 - np.exp(-optical_depth)) * 1e23   # nsec, nrad

    # Now define Cartesian grid corresponding to the image plane
    # we overwrite the number of pixels
    nbpixels = 2*nrad   
    #
    # recall computational grid: R = grid radius in code units, T = grid azimuth in radians
    R = gas.xmed
    T = gas.zm
    # x- and y- coordinates of image plane
    minxinterf = -R.max()
    maxxinterf = R.max()
    xinterf = minxinterf + (maxxinterf-minxinterf)*np.arange(nbpixels)/(nbpixels-1)
    yinterf = xinterf
    dxy = abs(xinterf[1]-xinterf[0])
    xgrid = xinterf+0.5*dxy
    ygrid = yinterf+0.5*dxy
    
    print('projecting polar specific intensity onto image plane...')
    # First do a rotation in disc plane by phiangle (clockwise), cartesian projection
    # (bilinear interpolation)
    phiangle_in_rad = phiangle*np.pi/180.0
    Inu_cart = np.zeros(nbpixels*nbpixels)
    Inu_cart = Inu_cart.reshape(nbpixels,nbpixels)
    for i in range(nbpixels):
        for j in range(nbpixels):
            xc =  xgrid[i]*np.cos(phiangle_in_rad) + ygrid[j]*np.sin(phiangle_in_rad)
            yc = -xgrid[i]*np.sin(phiangle_in_rad) + ygrid[j]*np.cos(phiangle_in_rad)
            rc = np.sqrt( xc*xc + yc*yc )
            if ( (rc >= R.min()) and (rc < R.max()) ):
                phic = math.atan2(yc,xc) + np.pi  # result between 0 and 2pi
                # expression for ir might not be general if simulation grid has an arithmetic spacing...
                ir = int(np.log(rc/R.min())/np.log(R.max()/R.min()) * (nrad-1.0))
                # CB: since T = pmed, we need to use nsec below instead of nsec-1
                jr = int((phic-T.min())/(T.max()-T.min()) * nsec)
                if (jr == nsec):
                    phic = 0.0
                    Tjr = T.min()
                    jr = 0
                else:
                    Tjr = T[jr]
                if (jr == nsec-1):
                    Tjrp1 = T.max()
                    jrp1  = 0
                else:
                    Tjrp1 = T[jr+1]
                    jrp1  = jr+1
                Inu_cart[j,i] = Inu_polar[jr,ir] * (R[ir+1]-rc) * (Tjrp1-phic) \
                    + Inu_polar[jrp1,ir]   * (R[ir+1]-rc) * (phic - Tjr) \
                    + Inu_polar[jr,ir+1]   * (rc - R[ir]) * (Tjrp1-phic) \
                    + Inu_polar[jrp1,ir+1] * (rc - R[ir]) * (phic - Tjr)
                Inu_cart[j,i] /= ( (R[ir+1]-R[ir])*(Tjrp1-Tjr) )
                '''
                if (Inu_cart[j,i] < 0):
                    sys.exit("Inu_cart < 0 in calc_abs_map: I must exit!")
                '''
            else:
                Inu_cart[j,i] = 0.0

    # Then project with inclination along line of sight:
    # CB: x needs to be changed, not y (checked, at least with xaxisflip set to Yes)
    inclination_in_rad = inclination*np.pi/180.0
    Inu_cart2 = np.zeros(nbpixels*nbpixels)
    Inu_cart2 = Inu_cart2.reshape(nbpixels,nbpixels)
    for i in range(nbpixels):
        xc = xgrid[i]*np.cos(inclination_in_rad)   # || < xgrid[i]
        ir = int((xc-xgrid.min())/(xgrid.max()-xgrid.min()) * (nbpixels-1.0))
        if (ir < nbpixels-1):
            Inu_cart2[:,i] = (Inu_cart[:,ir]*(xgrid[ir+1]-xc) + Inu_cart[:,ir+1]*(xc-xgrid[ir]))/(xgrid[ir+1]-xgrid[ir])
        else:
            Inu_cart2[:,i] = Inu_cart[:,nbpixels-1]
    
    # Finally do a rotation in the image plane by posangle
    posangle_in_rad = (posangle+90.0)*np.pi/180.0     # add 90 degrees to be consistent with RADMC3D's convention for position angle
    Inu_cart3 = np.zeros(nbpixels*nbpixels)
    Inu_cart3 = Inu_cart3.reshape(nbpixels,nbpixels)
    for i in range(nbpixels):
        for j in range(nbpixels):
            xc =  xgrid[i]*np.cos(posangle_in_rad) + ygrid[j]*np.sin(posangle_in_rad)
            yc = -xgrid[i]*np.sin(posangle_in_rad) + ygrid[j]*np.cos(posangle_in_rad)
            ir = int((xc-xgrid.min())/(xgrid.max()-xgrid.min()) * (nbpixels-1.0))
            jr = int((yc-ygrid.min())/(ygrid.max()-ygrid.min()) * (nbpixels-1.0))
            if ( (ir >= 0) and (jr >= 0) and (ir < nbpixels-1) and (jr < nbpixels-1) ):
                Inu_cart3[j,i] = Inu_cart2[jr,ir] * (xgrid[ir+1]-xc) * (ygrid[jr+1]-yc) \
                    + Inu_cart2[jr+1,ir]   * (xgrid[ir+1]-xc) * (yc-ygrid[jr]) \
                    + Inu_cart2[jr,ir+1]   * (xc-xgrid[ir]) * (ygrid[jr+1]-yc) \
                    + Inu_cart2[jr+1,ir+1] * (xc-xgrid[ir]) * (yc-ygrid[jr])
                Inu_cart3[j,i] /= ((xgrid[ir+1]-xgrid[ir]) * (ygrid[jr+1]-ygrid[jr]))
            else:
                if ( (ir >= nbpixels-1) and (jr < nbpixels-1) ):
                    Inu_cart3[j,i] = 0.0 # Inu_cart2[jr-1,nbpixels-1]
                if ( (jr >= nbpixels-1) and (ir < nbpixels-1) ):
                    Inu_cart3[j,i] = 0.0 # Inu_cart2[nbpixels-1,ir-1]
                if ( (ir >= nbpixels-1) and (jr >= nbpixels-1) ):
                    Inu_cart3[j,i] = 0.0 # Inu_cart2[nbpixels-1,nbpixels-1]


    # Inu contains the specific intensity in Jy/steradian projected onto the image plane
    Inu = Inu_cart3
    # Disc distance in metres
    D = distance * 206265.0 * 1.5e11
    # Convert specific intensity from Jy/steradian to Jy/pixel^2
    pixsurf_ster = (dxy*gas.culength/D)**2
    Inu *= pixsurf_ster    # Jy/pixel
    print("Total flux of 2D method [Jy] = "+str(np.sum(Inu)))

    # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
    if add_noise == 'Yes':
        # beam area in pixel^2
        beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(dxy**2.)
        # noise standard deviation in Jy per pixel (I've checked the expression below works well)
        noise_dev_std_Jy_per_pixel = noise_dev_std / np.sqrt(0.5*beam)  # 1D
        # noise array
        noise_array = np.random.normal(0.0,noise_dev_std_Jy_per_pixel,size=nbpixels*nbpixels)
        noise_array = noise_array.reshape(nbpixels,nbpixels)
        Inu += noise_array
    
    # pixel (cell) size in arcseconds 
    dxy *= (gas.culength/1.5e11/distance)  
    # beam area in pixel^2
    beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(dxy**2.)
    # stdev lengths in pixel
    stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / dxy
    stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / dxy

    # check beam is correctly handled by inserting a source point at the
    # origin of the raw intensity image
    if check_beam == 'Yes':
        Inu[nbpixels//2-1,nbpixels//2-1] = 500.0*Inu.max()

    # Call to Gauss_filter function
    print('convolution...')
    smooth2D = Gauss_filter(Inu, stdev_x, stdev_y, bpaangle, Plot=False)

    # convert image in mJy/beam or in microJy/beam
    # could be refined...
    convolved_Inu = smooth2D * 1e3 * beam   # mJy/beam
    strflux = 'mJy/beam'
    if convolved_Inu.max() < 1.0:
        convolved_Inu = smooth2D * 1e6 * beam   # microJy/beam
        strflux = '$\mu$Jy/beam'

    # ---------------------------------
    # save 2D flux map solution to fits 
    # ---------------------------------
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32    
    hdu.header['NAXIS'] = 2
    hdu.header['NAXIS1'] = nbpixels
    hdu.header['NAXIS2'] = nbpixels
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = float(-1.*dxy)
    hdu.header['CDELT2'] = float(dxy)
    hdu.header['CUNIT1'] = 'arcsec    '
    hdu.header['CUNIT2'] = 'arcsec    '
    hdu.header['CRPIX1'] = float((nbpixels+1.)/2.)
    hdu.header['CRPIX2'] = float((nbpixels+1.)/2.)
    if strflux == 'mJy/beam':
        hdu.header['BUNIT'] = 'milliJY/BEAM'
    if strflux == '$\mu$Jy/beam':
        hdu.header['BUNIT'] = 'microJY/BEAM'
    hdu.header['BTYPE'] = 'FLUX DENSITY'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0
    # keep track of all parameters in params.dat file
    # for i in range(len(lines_params)):
    #    hdu.header[var[i]] = par[i]
    hdu.data = convolved_Inu
    inbasename = os.path.basename('./'+outfile)
    if add_noise == 'Yes':
        jybeamfileout=re.sub('.fits', '_wn_JyBeam2D.fits', inbasename)
    else:
        jybeamfileout=re.sub('.fits', '_JyBeam2D.fits', inbasename)
    hdu.writeto(jybeamfileout, overwrite=True)

    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial') 
    fontcolor='white'

    # name of pdf file for final image
    fileout = re.sub('.fits', '.pdf', jybeamfileout)
    fig = plt.figure(figsize=(8.,8.))
    ax = plt.gca()
    plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.09)

    # Set x-axis orientation, x- and y-ranges
    # Convention is that RA offset increases leftwards (ie,
    # east is to the left), while Dec offset increases from
    # bottom to top (ie, north is the top).
    # CB: do not remove the multiplication by |cos(inclination)| !
    a0 = -xgrid[0]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))   # >0
    a1 = -xgrid[nbpixels-1]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))    # <0
    d0 = xgrid[0]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))    # <0
    d1 = xgrid[nbpixels-1]*gas.culength/1.5e11/distance*np.abs(np.cos(inclination_in_rad))    # >0
    # da positive definite
    if (minmaxaxis < abs(a0)):
        da = minmaxaxis
    else:
        da = abs(a0)
    ax.set_xlim(da,-da)      # x (=R.A.) increases leftward
    #ax.set_xlim(minmaxaxis,-minmaxaxis)      # x (=R.A.) increases leftward
    mina = da
    maxa = -da
    xlambda = mina - 0.166*da
    ax.set_ylim(-da,da)
    #ax.set_ylim(-minmaxaxis,minmaxaxis)      # x (=R.A.) increases leftward
    dmin = -da
    dmax = da

    # x- and y-ticks and labels
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian 
    ax.set_xlabel('RA offset [arcsec]')
    ax.set_ylabel('Dec offset [arcsec]')

    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    min = 0.0  # convolved_Inu.min()
    max = convolved_Inu.max()
    CM = ax.imshow(convolved_Inu, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[a0,a1,d0,d1], vmin=min, vmax=max)

    # Add wavelength in top-left corner
    strlambda = str(wavelength)+' mm (model)'
    if wavelength < 0.01:
        strlambda = str(wavelength*1e3)+'$\mu$m (model)'
    ax.text(xlambda,dmax-0.166*da,strlambda, fontsize=20, color = 'white', weight='bold')

    # Add + sign at the origin
    ax.plot(0.0,0.0,'+',color='white',markersize=10)

    # plot beam
    from matplotlib.patches import Ellipse
    e = Ellipse(xy=[xlambda,dmin+0.166*da], width=bmin, height=bmaj, angle=bpaangle+90.0)
    e.set_clip_box(ax.bbox)
    e.set_facecolor('white')
    e.set_alpha(0.8)
    ax.add_artist(e)

    # plot color-bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4.5%", pad=0.3)
    cax.yaxis.set_ticks_position('left')
    cax.xaxis.set_ticks_position('top')
    cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.yaxis.set_tick_params(labelsize=20, direction='out')
    cax.text(.990, 0.22, strflux, fontsize=20, horizontalalignment='right', color='white', weight='bold')

    plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
    plt.clf()

    # =====================
    # Compute deprojection and polar expansion (SP)
    # =====================
    if deproj_polar == 'Yes':
        currentdir = os.getcwd()
        alpha_min = 0.;          # deg, PA of offset from the star
        Delta_min = 0.;          # arcsec, amplitude of offset from the star
        RA = 0.0  # if input image is a prediction, star should be at the center
        DEC = 0.0 # note that this deprojection routine works in WCS coordinates
        cosi = np.cos(inclination_input*np.pi/180.)

        print('deprojection around PA [deg] = ',posangle)
        print('and inclination [deg] = ',inclination_input)
        
        # makes a new directory "deproj_polar_dir" and calculates a number
        # of products: copy of the input image [_fullim], centered at
        # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
        # image [_polar], etc. Also, a _radial_profile which is the
        # average radial intensity.
        exec_polar_expansions(jybeamfileout,'deproj_polar_dir',posangle,cosi,RA=RA,DEC=DEC,
                              alpha_min=alpha_min, Delta_min=Delta_min,
                              XCheckInv=False,DoRadialProfile=False,
                              DoAzimuthalProfile=False,PlotRadialProfile=False,
                              a_min=0.02,a_max=1.0,zoomfactor=1.)
        
        # Save polar fits in current directory
        fileout = re.sub('.pdf', '_polar.fits', fileout)
        command = 'cp deproj_polar_dir/'+fileout+' .'
        os.system(command)

        filein = re.sub('.pdf', '_polar.fits', fileout)
        # Read fits file with deprojected field in polar coordinates
        f = fits.open(filein)
        convolved_intensity = f[0].data    # uJy/beam

        # azimuthal shift such that PA=0 corresponds to y-axis pointing upwards, and
        # increases counter-clockwise from that axis
        if xaxisflip == 'Yes':
            jshift = int(nbpixels/4)
        else:
            jshift = int(nbpixels/4)
        convolved_intensity = np.roll(convolved_intensity, shift=-jshift, axis=1)

        # -------------------------------
        # plot image in polar coordinates
        # -------------------------------
        fileout = re.sub('.fits', '.pdf', filein)
        fig = plt.figure(figsize=(8.,8.))
        ax = plt.gca()
        plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.09)

        # Set x- and y-ranges
        ax.set_xlim(-180,180)          # PA relative to Clump 1's
        ax.set_ylim(0,minmaxaxis)      # Deprojected radius in arcsec
        
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        #ax.set_yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))
        ax.set_xticks((-180,-120,-60,0,60,120,180))
        ax.set_xlabel('Position angle [deg]')
        ax.set_ylabel('Radius [arcsec]')

        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = convolved_intensity.min()  # not exactly same as 0
        max = convolved_intensity.max()
        CM = ax.imshow(convolved_intensity, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[-180,180,0,a0], vmin=min, vmax=max, aspect='auto')   # (left, right, bottom, top)

        # Plot axes to show intensity peaks
        ax.plot([-180,180],[0.53,0.53],'--',linewidth=2,color='darkgray')
        ax.plot([0,0],[0.0,0.7],'--',linewidth=2,color='darkgray')
        ax.plot([-180,180],[0.31,0.31],'--',linewidth=2,color='darkgray')
        ax.plot([-130,-130],[0.0,0.7],'--',linewidth=2,color='darkgray')

        # Add wavelength in bottom-left corner
        strlambda = str(wavelength)+' mm (model)'
        if wavelength < 0.01:
            strlambda = str(wavelength*1e3)+'$\mu$m (model)'
        ax.text(60,0.02,strlambda,fontsize=16,color='white',weight='bold')

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="4.5%", pad=0.3)
        cax.yaxis.set_ticks_position('left')
        cax.xaxis.set_ticks_position('top')
        cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        cax.yaxis.set_tick_params(labelsize=20, direction='out')
        cax.text(.990, 0.22, strflux, fontsize=20, horizontalalignment='right', color='white', weight='bold')

        plt.savefig('./'+fileout, dpi=160)
        plt.clf()

        os.system('rm -rf deproj_polar_dir')
        os.chdir(currentdir)

        
print('--------- done! ----------')
