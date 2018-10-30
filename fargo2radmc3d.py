# =================================================================== 
#                         FARGO2D to RADMC3D
#                 originally written by Sebastian Perez (SP)
#              with modifications made by Clement Baruteau (CB)
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
# - work out dustopac.inp file when we don't recalculate opacities (make
# sure the number of bins is correct)
# - should I really use temperature from MC thermal simulation and not the
# gas temperature out of the hydro simulation? -> try to have an optional flag
# to use the gas temperature and put it in a dust_temperature.dat file.
# - issue with the grid? r, theta, phi and not R(cylindrical radius), theta, phi -> issue with z in z_expansion? 
# - check that all goes well without x-axisflip!
# =========================================


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
from astropy.convolution import convolve
from matplotlib.colors import LinearSegmentedColormap
import psutil

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
        # thmin is set as ~pi/2+atan(2.0*h) with h the gas aspect ratio (-> z_max ~ 2H_gas)
        thmin = np.pi/2. - math.atan(2.0*aspectratio)
        thmax = np.pi/2.
        # first define array of colatitudes above midplane
        ymp = np.linspace(np.log10(thmin),np.log10(thmax),self.ny//2+1)
        # refine towards the midplane
        ym_lower = -1.0*10**(ymp)+thmin+thmax 
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
                 incl=30.0, posang=0.0, phi=0.0, Loadlambda = 0,
                 line = '12co', imolspec=1, iline=3, linenlam=80, widthkms=4,):
        
        # disk parameters
        self.distance = distance * pc
        self.label = label
        # RT pars
        self.Lambda = Lambda
        self.Loadlambda = Loadlambda
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
    if M.Loadlambda==1:
        command='radmc3d image loadlambda npix '+str(M.npix)+' incl '+str(M.incl)+' posang '+str(M.posang)+' phi '+str(M.phi)
    if secondorder == 'Yes':
        command=command+' secondorder'
        
    print(command)
    os.system(command)


# -------------------------
# Convert result of RADMC3D calculation into fits file
# -------------------------
def exportfits(M, Plot=False):

    outfile = 'image_'+str(M.label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(M.phi)+'_PA'+str(posangle)+'.fits'
    if secondorder == 'Yes':
        outfile = 'image_'+str(M.label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(M.phi)+'_PA'+str(posangle)+'_secondorder.fits'
    
    infile = 'image.out'
    
    LOG = open('fluxlog.txt','a')
    LOG.write(outfile+"\n")

    # read header info:
    f = open(infile,'r')
    iformat = int(f.readline())
    # nb of pixels
    im_nx, im_ny = tuple(np.array(f.readline().split(),dtype=int))  
    # nb of wavelengths
    nlam = int(f.readline())
    # pixel size in each direction in cm
    pixsize_x, pixsize_y = tuple(np.array(f.readline().split(),dtype=float))  
    lbda = np.empty(nlam)
    for i in range(nlam):
        lbda[i] = float(f.readline())
    f.readline()                # empty line

    # load image data
    images = np.loadtxt(infile, skiprows=(5+nlam))

    # calculate physical scales
    distance = M.distance          # distance is in cm here
    pixsize_x_deg = 180.0*pixsize_x / distance / pi
    pixsize_y_deg = 180.0*pixsize_y / distance / pi

    # surface of a pixel in radian squared
    pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (pi/180.)**2
    # 1 Jansky converted in cgs x pixel surface in cm^2
    fluxfactor = 1.e23 * pixsurf_ster
    
    if nlam>1:
        print("multiple ("+str(nlam)+") lambdas:")
        print(lbda)
        im = images.reshape(nlam,im_ny,im_nx)
        naxis = 3
    else:
        im = images.reshape(im_ny,im_nx)
        naxis = 2
    
    if Plot:
        import matplotlib.pyplot as plt
        plt.imshow(im, cmap = 'plasma', origin='lower',aspect='auto')
        plt.axis('equal')
        plt.show()

    hdu = fits.PrimaryHDU()

    # hdu.header['SIMPLE'] = 'T       '; # makes simobserve crash
    hdu.header['BITPIX'] = -32
    
    hdu.header['NAXIS'] = naxis
    hdu.header['NAXIS1'] = im_nx
    hdu.header['NAXIS2'] = im_ny
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    # hdu.header['SPECSYS'] = 'LSRK    '
    hdu.header['CTYPE1'] = 'RA---TAN'
    hdu.header['CTYPE2'] = 'DEC--TAN'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
    hdu.header['CDELT2'] = float(pixsize_y_deg)
    hdu.header['CUNIT1'] = 'DEG     '
    hdu.header['CUNIT2'] = 'DEG     '
    hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
    hdu.header['CRPIX2'] = float((im_ny+1.)/2.)

    hdu.header['BUNIT'] = 'JY/PIXEL'
    hdu.header['BTYPE'] = 'Intensity'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0

    # keep track of all parameters in params.dat file
    for i in range(len(lines_params)):
        hdu.header[var[i]] = par[i]

    LOG.write('pixsize '+str(pixsize_x_deg*3600.)+"\n")

    if nlam > 1:
        restfreq = c * 1.e4 / lbda[int((nlam+1.)/2.)] # micron to Hz
        nus = c * 1.e4 / lbda                         # Hx 
        dnu = nus[1] - nus[0]
        hdu.header['NAXIS3'] = int(nlam)
        hdu.header['CTYPE3'] = 'FREQ    '
        hdu.header['CUNIT3'] = 'Hz      '
        hdu.header['CRPIX3'] = float((nlam+1.)/2.)
        hdu.header['CRVAL3'] = float(restfreq)
        hdu.header['CDELT3'] = float(dnu)
        hdu.header['RESTFREQ'] = float(restfreq)

    # conversion of the intensity from erg/s/cm^2/steradian to Jy/pix
    im = im*fluxfactor
    hdu.data = im.astype('float32')

    if nlam>1:
        print("reporting fluxes")
        for i in range(nlam):
            iflux = np.sum(hdu.data[i,:,:])
            print("F("+str(lbda[i])+") = "+str(iflux))
            LOG.write("F("+str(lbda[i])+") = "+str(iflux)+"\n")
    else:
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
                
    result = convolve(data, Z, boundary='extend')

    if Plot==True:
        plt.imshow(result, cmap=cm.magma)
        plt.show()

    return result


# -------------------------------------------------------------------  
# produce the dust kappa files
# requires the .lnk files and the bhmie.f code inside a folder
# called opac
# -------------------------------------------------------------------  
def make_dustkappa(species='Draine_Si', amin=0.1, amax=1000,
                   graindens=2.0, nbins=20, abin = 0, alpha=3.5):

    print('making dustkappa files')

    currentdir = os.getcwd()
    # we need to have a global extension for the opacity directory
    opacdir = os.path.expanduser(opacity_dir)    
    command = "cd "+opacdir+"; make clean; make"
    os.system(command)
    os.chdir(opacdir)
    
    pathout='dustkappa_'+species+str(abin)+'.inp'
    lnk_file=species
    Type=species
    Pa=(amax/amin)**(1.0/(nbins-1.0))

    A=np.zeros(nbins)
    A[0]=amin

    for i in range(nbins):
        command = 'rm -f param.inp'
        os.system(command)
        A[i]=amin*(Pa**(i))
        acm=A[i]*10.0**(-4.0)    # in cm
        print("a = %1.2e um"  %A[i])
        file_inp=open('param.inp','w')
        file_inp.write(lnk_file+'\n')
        e=round(np.log10(acm))
        b=acm/(10.0**e)
        file_inp.write('%1.2fd%i \n' %(b,e))
        file_inp.write('%1.2f \n' %graindens) 
        file_inp.write('1')
        
        file_inp.close()
        os.system('./makeopac')
        os.system('mv dustkappa_'+lnk_file+'.inp dustkappa_'+Type+'_'+str(i+1)+'.inp ') 


    #--------- READ OPACITIES AND COMPUTE MEAN OPACITY

    # read number of wavelengths
    opct=np.loadtxt(lnk_file+'.lnk')
    Nw=len(opct[:,0])
        
    Op=np.zeros((Nw,4))         # wl, kappa_abs, kappa_scat, g 
    Op[:,0]=opct[:,0]

    Ws_mass=np.zeros(nbins)     # weigths by mass and abundances
    for i in range(nbins):
        Ws_mass[i] = (A[i]**(-alpha))*(A[i]**(3.0))*A[i]  # w(a) propto n(a)*m(a)*da and da propto a

    W_mass = Ws_mass/np.sum(Ws_mass)

    for i in range(nbins):
        file_inp=open('dustkappa_'+Type+'_'+str(i+1)+'.inp','r')
        file_inp.readline()
        file_inp.readline()

        for j in range(Nw):
            line=file_inp.readline()
            dat=line.split()
            kabs=float(dat[1])
            kscat=float(dat[2])
            g=float(dat[3])

            Op[j,1]+=kabs*W_mass[i]
            Op[j,2]+=kscat*W_mass[i]
            Op[j,3]+=g*W_mass[i] 
                
        file_inp.close()
        os.system('rm dustkappa_'+Type+'_'+str(i+1)+'.inp')

    #---------- WRITE MEAN OPACITY
    final=open(pathout,'w')

    final.write('3 \n')
    final.write(str(Nw)+'\n')
    for i in range(Nw):
        final.write('%f \t %f \t %f \t %f\n' %(Op[i,0],Op[i,1],Op[i,2],Op[i,3]))
    final.close()

    os.system('mv '+pathout+' '+currentdir)
    os.chdir(currentdir)


# -------------------------------------------------------------------  
# writing dustopac
# -------------------------------------------------------------------  
def write_dustopac(species=['ac_opct', 'Draine_Si']):
    nspec = len(species)
    print('writing dust opacity out')
    hline="-----------------------------------------------------------------------------\n"
    OPACOUT=open('dustopac.inp','w')
    lines0=["2               iformat (2)\n",
            str(nspec)+"               species\n",
            hline]
    OPACOUT.writelines(lines0)
    for i in range(nspec):
        lines=["1               in which form the dust opacity of dust species is to be read\n",
               "0               0 = thermal grains\n",
               species[i]+"         dustkappa_***.inp file\n",
               hline
           ]
        OPACOUT.writelines(lines)
    OPACOUT.close()


# -------------------------------------------------------------------  
# plotting opacities
# -------------------------------------------------------------------  
def plot_opacities(species='mix_2species_porous',amin=0.1,amax=1000,nbin=10):
    ax = plt.gca()
    ax.tick_params(axis='both',length = 10, width=1)

    plt.xlabel(r'Dust size [meters]')
    plt.ylabel(r'Opacities $[{\rm cm}^2\;{\rm g}^{-1}]$')

    absorption1 = np.zeros(nbin)
    absorption2 = np.zeros(nbin)
    scattering1 = np.zeros(nbin)
    scattering2 = np.zeros(nbin)
    sizes = np.logspace(np.log10(amin), np.log10(amax), nbin)

    lbda1 = 870.  # 0.87 mm in microns
    lbda2 = 9000. # 9 mm in microns

    for k in range(nbin):
        file = 'dustkappa_'+species+str(k)+'.inp'
        (lbda, kappa_abs, kappa_sca, g) = np.loadtxt(file, unpack=True, skiprows=2)

        i1 = np.argmin(np.abs(lbda-lbda1))
        i2 = np.argmin(np.abs(lbda-lbda2))

        # interpolation in log
        l1 = lbda[i1-1]
        l2 = lbda[i1+1]
        k1 = kappa_abs[i1-1]
        k2 = kappa_abs[i1+1]
        ks1 = kappa_sca[i1-1]
        ks2 = kappa_sca[i1+1]
        absorption1[k] =  (k1*np.log(l2/lbda1) +  k2*np.log(lbda1/l1))/np.log(l2/l1)
        scattering1[k] = (ks1*np.log(l2/lbda1) + ks2*np.log(lbda1/l1))/np.log(l2/l1)
        #
        l1 = lbda[i2-1]
        l2 = lbda[i2+1]
        k1 = kappa_abs[i2-1]
        k2 = kappa_abs[i2+1]
        ks1 = kappa_sca[i2-1]
        ks2 = kappa_sca[i2+1]
        absorption2[k] =  (k1*np.log(l2/lbda2) +  k2*np.log(lbda2/l1))/np.log(l2/l1)
        scattering2[k] = (ks1*np.log(l2/lbda2) + ks2*np.log(lbda2/l1))/np.log(l2/l1)

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
    lbda2 *= 1e-3  # in mm

    plt.loglog(sizes, absorption1, lw=2., linestyle = 'solid', color = c20[1], label='$\kappa_{abs}$ at '+str(lbda1)+' mm')
    plt.loglog(sizes, absorption2, lw=2., linestyle = 'solid', color = c20[3], label='$\kappa_{abs}$ at '+str(lbda2)+' mm')
    plt.loglog(sizes, absorption1+scattering1, lw=2., linestyle = 'dashed', color = c20[1], label='$\kappa_{abs}$+$\kappa_{sca}$ at '+str(lbda1)+' mm')
    plt.loglog(sizes, absorption2+scattering2, lw=2., linestyle = 'dashed', color = c20[3], label='$\kappa_{abs}$+$\kappa_{sca}$ at '+str(lbda2)+' mm')
    plt.legend()

    plt.ylim(1e-2,1e2)
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

    print('writing radmc3d.inp out')

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

# =========================
# 1. read RT parameter file
# =========================
params = open("params.dat",'r')# opening parfile
lines_params = params.readlines()     # reading parfile
params.close()                 # closing parfile
par = []                       # allocating a dictionary
var = []                       # allocating a dictionary
for line in lines_params:             # iterating over parfile
    name, value, comment = line.split() # spliting name and value (first blank)
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
    exec(var[i]+"="+par[i])


print('--------- RT parameters ----------')
print('directory = ', dir)
print('output number  = ', on)
print('wavelength [mm] = ', wavelength)
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
print('type of vertical expansion done for dust mass volume density = ', z_expansion)
if z_expansion == 'T':
    print('Hd / Hgas = sqrt(alpha/(alpha+St))')
if z_expansion == 'T2':
    print('Hd / Hgas = sqrt(10*alpha/(10*alpha+St))')
if z_expansion == 'F':
    print('Hd / Hgas = 0.7 x ((St + 1/St)/1000)^0.2 (Fromang & Nelson 09)')
print('do we recompute all dust densities? : ', recalc_density)
print('do we recompute all dust opacities? : ', recalc_opac)
print('do we use pre-calculated opacities, which are located in opacity_dir directory? : ', precalc_opac)
print('do we run RADMC3D calculation of temperature and ray tracing? : ', recalc_radmc)
if recalc_radmc == 'Yes':
    recalc_fluxmap = 'Yes'
print('do we recompute convolved flux map from the output of RADMC3D calculation? : ', recalc_fluxmap)
print('do we compute 2D analytical solution to RT equation w/o scattering? : ', calc_abs_map)
print('do we take the gas (hydro) temperature for the dust temperature? : ', Tdust_eq_Tgas)
print('do we also plot intensity map in polar coordinates? : ', polar_extra_map)
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

# x-axis flip means that we apply a mirror-symetry x -> -x to the 2D simulation plane,
# which in practice is done by adding 180 degrees to the disc's inclination wrt the line of sight
if xaxisflip == 'Yes':
    inclination = inclination + 180.0

# this is how the beam position angle should be modified to be understood as
# measuring East from North. You can check this by setting check_beam to Yes
# in the params.dat parameter file
bpaangle = -90.0-bpaangle

# label for the name of the image file created by RADMC3D
label = dir+'_o'+str(on)+'_p'+str(pindex)+'_r'+str(ratio)+'_amin'+str(amin)+'_amax'+str(amax)+'_nbin'+str(nbin)+'_scatmode'+str(scat_mode)+'_nphot'+str('{:.0e}'.format(nb_photons))+'_nphotscat'+str('{:.0e}'.format(nb_photons_scat))+'_ncol'+str(ncol)+'_zexp'+str(z_expansion)+'_xaxisflip'+str(xaxisflip)

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
if recalc_density == 'Yes':
    print('--------- computing dust surface density ----------')

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
            print('pb with i = ', i, ' in recalc_density step: I must exit!')
            exit()
        # azimuthal index of the cell where the particle is
        # (general expression since grid spacing in azimuth is always arithmetic)
        j = int((t-gas.zm.min())/(gas.zm.max()-gas.zm.min()) * nsec)
        if (j < 0 or j >= nsec):
            print('pb with j = ', j, ' in recalc_density step: I must exit!')
            exit()
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
        St = avgstokes[ibin]                                  # avg stokes number for that bin 
        hgas = aspectratio * (gas.xmed)**(flaringindex)       # gas aspect ratio (gas.xmed[i] = R in code units)
        # vertical extension depends on grain Stokes number
        # T = theoretical: hd/hgas = sqrt(alpha/(St+alpha))
        # T2 = theoretical: hd/hgas = sqrt(Dz/(St+Dz)) with Dz = 10xalpha here is the coefficient for
        # vertical diffusion at midplane, which can differ from alpha
        # F = extrapolation from the simulations by Fromang & Nelson 09
        if z_expansion == 'F':
            hd[ibin,:] = 0.7 * hgas * ((St+1./St)/1000.)**(0.2)     
        if z_expansion == 'T':
            hd[ibin,:] = hgas * np.sqrt(alphaviscosity/(alphaviscosity+St))
        if z_expansion == 'T2':
            hd[ibin,:] = hgas * np.sqrt(10.0*alphaviscosity/(10.0*alphaviscosity+St))
    
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

else:
    print('--------- I did not compute dust densities (recalc_density = No in params.dat file) ----------')


# =========================
# 4. Compute dust opacities
# =========================
if recalc_opac == 'Yes':
    print('--------- computing dust opacities ----------')
    ldustopac = []
    graindens = 2.0 # g / cc
    if species == 'mix_2species_porous':
        graindens = 0.1 # g / cc
    if species == 'mix_2species':
        graindens = 1.7 # g / cc
    for ibin in range(int(nbin)):
        print(ibin)
        # function make_dustkappa is defined in the top of the program
        make_dustkappa(species=species, abin=ibin, nbins=nbin, alpha=float(pindex),graindens=graindens,
                       amin=bins[ibin]*1.e6, amax=bins[ibin+1]*1.e6)
        ldustopac.append(species+str(ibin))
    print(ldustopac)
    write_dustopac(species=ldustopac)
    plot_opacities(species=species,amin=amin,amax=amax,nbin=nbin)
else:
    print('------- taking dustkappa* opacity files in current directory (recalc_opac = No in params.dat file) ------ ')


# =========================
# 5. Call to RADMC3D thermal solution and ray tracing 
# =========================
if recalc_radmc == 'Yes':
    # Write other parameter files required by RADMC3D
    print('--------- printing auxiliary files ----------')

    # Write 3D spherical grid for RT computational calculation
    write_AMRgrid(gas, Plot=False)

    # need to check why we need to output wavelength...
    write_wavelength()
    write_stars(Rstar = rstar, Tstar = teff)

    # rto_style = 3 means that RADMC3D will write binary output files
    # setthreads corresponds to the number of threads / cpus over which radmc3d runs
    write_radmc3dinp(nphot_scat=nb_photons_scat, nphot=nb_photons, rto_style=3, 
                     modified_random_walk=1, scattering_mode_max=scat_mode, setthreads=4)
    
    # Add 90 degrees to position angle so that RADMC3D's definition of
    # position angle be consistent with observed position
    # angle, which is what we enter in the params.dat file
    M = RTmodel(distance=distance, Lambda=wavelength*1e3, Loadlambda=0, label=label,
                npix=nbpixels, phi=phiangle, incl=inclination, posang=posangle+90.0) 

    # Get dust temperature
    print('--------- executing Monte-Carlo thermal calculation with RADMC3D ----------')
    run_mctherm()
    
    # Run ray tracing
    print('--------- executing ray tracing with RADMC3D ----------')
    run_raytracing(M)

    print('--------- exporting results in fits format ----------')
    # Put Plot=True to display image of the raw intensity
    outfile = exportfits(M, Plot=False)

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
    ax.plot(S, axiTm, 'bo', markersize=1., label='midplane');
    ax.plot(S, Tm_model, 'b--', markersize=1., label='midplane hydro');
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
    outfile = 'image_'+str(label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)+'.fits'
    if secondorder == 'Yes':
        outfile = 'image_'+str(label)+'_lbda'+str(wavelength)+'_i'+str(inclination)+'_phi'+str(phiangle)+'_PA'+str(posangle)+'_secondorder.fits'


# =========================
# 6. Convolve raw flux with beam and produce final image
# =========================
if recalc_fluxmap == 'Yes':
    print('--------- Convolving and writing final image ----------')

    f = fits.open('./'+outfile)
    raw_intensity = f[0].data
    if recalc_radmc == 'No':
        print("Total flux [Jy] = "+str(np.sum(raw_intensity)))   # sum over pixels
    hdr = f[0].header
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']
    # pixel size converted from degrees to arcseconds
    cdelt = np.abs(hdr['CDELT1']*3600.)

    # check beam is correctly handled by inserting a source point at the
    # origin of the raw intensity image
    if check_beam == 'Yes':
        raw_intensity[nx//2-1,ny//2-1] = 100.0*raw_intensity.max()

    # ------------
    # smooth image
    # ------------
    # beam area in pixel^2
    beam =  (np.pi/(4.*np.log(2.)))*bmaj*bmin/(cdelt**2.)
    # stdev lengths in pixel
    stdev_x = (bmaj/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
    stdev_y = (bmin/(2.*np.sqrt(2.*np.log(2.)))) / cdelt

    # Call to Gauss_filter function
    smooth = Gauss_filter(raw_intensity, stdev_x, stdev_y, bpaangle, Plot=False)

    # convert image in mJy/beam or in microJy/beam
    # could be refined...
    convolved_intensity = smooth * 1e3 * beam   # mJy/beam
    strflux = 'mJy/beam'
    if convolved_intensity.max() < 1.0:
        convolved_intensity = smooth * 1e6 * beam   # microJy/beam
        strflux = '$\mu$Jy/beam'

    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rc('font', family='Arial') 
    fontcolor='white'

    inbasename = os.path.basename('./'+outfile)
    # name of pdf file for final image
    fileout = re.sub('.fits', '.pdf', inbasename)
    fig = plt.figure(figsize=(8.,8.))
    ax = plt.gca()

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

    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.set_xlabel(r'$\Delta \alpha $ / arcsec')
    ax.set_ylabel(r'$\Delta \delta $ / arcsec ')

    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    min = convolved_intensity.min()
    max = convolved_intensity.max()
    CM = ax.imshow(convolved_intensity, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[a0,a1,d0,d1], vmin=min, vmax=max)

    # Add wavelength in top-left corner
    strlambda = '$\lambda$ = '+str(wavelength)+' mm'
    ax.text(xlambda,dmax-0.166*da,strlambda, fontsize=18, color = 'white')

    # Add + sign at the origin
    ax.plot(0.0,0,0,'+',color='white')

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
    cax.xaxis.set_tick_params(labelsize=15, direction='out')
    cax.yaxis.set_tick_params(labelsize=15, direction='out')
    cax.text(.990, 0.30, strflux, fontsize=15, horizontalalignment='right', color='white')

    plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
    plt.clf()

    # --------------------
    # plotting image in polar coordinates
    # --------------------
    if polar_extra_map == 'Yes':
        print('projecting convolved intensity map onto polar plane...')
        Inu_polar3 = np.zeros(nbpixels*nbpixels)
        Inu_polar3 = Inu_polar3.reshape(nbpixels,nbpixels)
        # x-grid contains right ascension offset
        xgrid = a0 + (a1-a0)*np.arange(nbpixels)/(nbpixels-1)
        xgrid = -xgrid
        # x-grid contains declination offset
        ygrid = d0 + (d1-d0)*np.arange(nbpixels)/(nbpixels-1)
        # rc-grid is radius in the polar map
        rmin = 0.0    # specify other value in you want to do a radial zoom
        rmax = np.minimum(minmaxaxis,abs(a0))
        rc   = rmin + (rmax-rmin)* np.arange(nbpixels)/(nbpixels-1)
        # phic-grid is position angle in the polar map
        phic = 2.0*np.pi * np.arange(nbpixels)/nbpixels
        for i in range(nbpixels):
            for j in range(nbpixels):
                # remove 3pi/2 to phic to comply with definition of position angle east of north
                xc = rc[i]*np.cos(phic[j]-1.5*np.pi)    
                yc = rc[i]*np.sin(phic[j]-1.5*np.pi)
                ir = int((xc-xgrid.min())/(xgrid.max()-xgrid.min()) * (nbpixels-1.0))
                jr = int((yc-ygrid.min())/(ygrid.max()-ygrid.min()) * (nbpixels-1.0))
                if ( (ir < nbpixels-1) and (jr < nbpixels-1) ):
                    Inu_polar3[j,i] = convolved_intensity[jr,ir] * (xgrid[ir+1]-xc) * (ygrid[jr+1]-yc) \
                        + convolved_intensity[jr+1,ir]   * (xgrid[ir+1]-xc) * (yc-ygrid[jr]) \
                        + convolved_intensity[jr,ir+1]   * (xc-xgrid[ir]) * (ygrid[jr+1]-yc) \
                        + convolved_intensity[jr+1,ir+1] * (xc-xgrid[ir]) * (yc-ygrid[jr])
                    Inu_polar3[j,i] /= ((xgrid[ir+1]-xgrid[ir]) * (ygrid[jr+1]-ygrid[jr]))
                else:
                    Inu_polar3[j,i] = 0.0

        # name of pdf file for final image
        fileout = re.sub('.fits', 'polar.pdf', inbasename)
        fig = plt.figure(figsize=(8.,8.))
        ax = plt.gca()

        # Labels and ticks
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.set_xlabel('Position angle / deg')
        ax.set_ylabel('Radius / arcsec')

        # x- and y-ranges
        ax.set_xlim(0,360)      # PA between 0 and 360 degrees
        ax.set_ylim(rmin,rmax)  # radius between rmin and rmax (see above)
        
        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = Inu_polar3.min()
        max = Inu_polar3.max()
        CM = ax.imshow(np.transpose(Inu_polar3), origin='lower', cmap='nipy_spectral', interpolation='bilinear', vmin=min, vmax=max, extent=[0.0,360.0,rmin,rmax],aspect='auto')

        # Add wavelength in top-left corner
        strlambda = '$\lambda$ = '+str(wavelength)+' mm'
        ax.text(30.0,rmin+0.9*(rmax-rmin),strlambda, fontsize=18, color = 'white')

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="4.5%", pad=0.3)
        cax.yaxis.set_ticks_position('left')
        cax.xaxis.set_ticks_position('top')
        cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=15, direction='out')
        cax.yaxis.set_tick_params(labelsize=15, direction='out')
        cax.text(.990, 0.30, strflux, fontsize=15, horizontalalignment='right', color='white')

        # save in pdf file
        plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
        plt.clf()


# =========================
# 7. Compute 2D analytical solution to RT equation w/o scattering
# =========================
if calc_abs_map == 'Yes':
    print('--------- Computing 2D analytical solution to RT equation w/o scattering ----------')
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
            print('pb with i = ', i, ' in calc_abs_map step: I must exit!')
            exit()
        # azimuthal index of the cell where the particle is
        # (general expression since grid spacing in azimuth is always arithmetic)
        j = int((t-gas.zm.min())/(gas.zm.max()-gas.zm.min()) * nsec)
        if (j < 0 or j >= nsec):
            print('pb with j = ', j, ' in calc_abs_map step: I must exit!')
            exit()
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

    # We then need to recompute absorption mass opacity from dustkappa files
    abs_opacity = np.zeros(nbin)
    lbda1 = wavelength * 1e3  # wavelength in microns
    if precalc_opac == 'Yes':
        opacdir = os.path.expanduser(opacity_dir)
        # Case where we use pre-calculated dustkappa* files located in opacity_dir files
        if species == 'mix_2species':
            sizemin_file = 1e-6          # in meters, do not edit!
            sizemax_file = 1e-1          # in meters, do not edit!
            nbfiles = 100
        elif species == 'mix_2species_porous':
            sizemin_file = 1e-5          # in meters, do not edit!
            sizemax_file = 3e-1          # in meters, do not edit!
            nbfiles = 90
        else:
            print('I do not have pre-calculated opacity files for your type of species in the opacity_dir: I must exit!')
            exit()            
        size_file = sizemin_file * (sizemax_file/sizemin_file)**(np.arange(nbfiles)/(nbfiles-1.0))
    # Loop over size bins
    for k in range(nbin):
        if precalc_opac == 'No':
            # Case where we use dustkappa* files in current directory
            file = 'dustkappa_'+species+str(k)+'.inp'
            (lbda, kappa_abs, kappa_sca, g) = np.loadtxt(file, unpack=True, skiprows=2)
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
                (lbda_inf, kappa_abs_inf, kappa_sca_inf, g_inf) = np.loadtxt(file_index_inf, unpack=True, skiprows=2)
                (lbda_sup, kappa_abs_sup, kappa_sca_sup, g_sup) = np.loadtxt(file_index_sup, unpack=True, skiprows=2)
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
                (lbda_inf, kappa_abs_inf, kappa_sca_inf, g_inf) = np.loadtxt(file_index_inf, unpack=True, skiprows=2)
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

    # free RAM memory
    del dustcube, abs_opacity_2D

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
                    print("Inu_cart < 0 in calc_abs_map: I must exit!")
                    exit()
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
            if ( (ir < nbpixels-1) and (jr < nbpixels-1) ):
                Inu_cart3[j,i] = Inu_cart2[jr,ir] * (xgrid[ir+1]-xc) * (ygrid[jr+1]-yc) \
                    + Inu_cart2[jr+1,ir]   * (xgrid[ir+1]-xc) * (yc-ygrid[jr]) \
                    + Inu_cart2[jr,ir+1]   * (xc-xgrid[ir]) * (ygrid[jr+1]-yc) \
                    + Inu_cart2[jr+1,ir+1] * (xc-xgrid[ir]) * (yc-ygrid[jr])
                Inu_cart3[j,i] /= ((xgrid[ir+1]-xgrid[ir]) * (ygrid[jr+1]-ygrid[jr]))
            else:
                if ( (ir >= nbpixels-1) and (jr < nbpixels-1) ):
                    Inu_cart3[j,i] = Inu_cart2[jr-1,nbpixels-1]
                if ( (jr >= nbpixels-1) and (ir < nbpixels-1) ):
                    Inu_cart3[j,i] = Inu_cart2[nbpixels-1,ir-1]
                if ( (ir >= nbpixels-1) and (jr >= nbpixels-1) ):
                    Inu_cart3[j,i] = Inu_cart2[nbpixels-1,nbpixels-1]

    
    # Inu contains the specific intensity in Jy/steradian projected onto the image plane
    Inu = Inu_cart3
    # Disc distance in metres
    D = distance * 206265.0 * 1.5e11
    # Convert specific intensity from Jy/steradian to Jy/pixel^2
    pixsurf_ster = (dxy*gas.culength/D)**2
    Inu *= pixsurf_ster    # Jy/pixel
    print("Total flux of 2D method [Jy] = "+str(np.sum(Inu)))
    
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
        Inu[nbpixels//2-1,nbpixels//2-1] = 1000.0*Inu.max()

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

    # -------------------------------------
    # SP: save 2D flux map solution to fits 
    # -------------------------------------
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
    hdu.header['CUNIT1'] = 'DEG     '
    hdu.header['CUNIT2'] = 'DEG     '
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
    for i in range(len(lines_params)):
        hdu.header[var[i]] = par[i]
    hdu.data = convolved_Inu
    inbasename = os.path.basename('./'+outfile)
    jybeamfileout=re.sub('.fits', '_JyBeam.fits', inbasename)
    hdu.writeto(jybeamfileout, overwrite=True)

    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rc('font', family='Arial') 
    fontcolor='white'

    inbasename = os.path.basename('./'+outfile)
    # name of pdf file for final image
    fileout = re.sub('.fits', '2D.pdf', inbasename)
    fig = plt.figure(figsize=(8.,8.))
    ax = plt.gca()

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
    mina = da
    maxa = -da
    xlambda = mina - 0.166*da
    ax.set_ylim(-da,da)
    dmin = -da
    dmax = da

    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.set_xlabel(r'$\Delta \alpha $ / arcsec')
    ax.set_ylabel(r'$\Delta \delta $ / arcsec')

    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    min = convolved_Inu.min()
    max = convolved_Inu.max()
    CM = ax.imshow(convolved_Inu, origin='lower', cmap='nipy_spectral', interpolation='bilinear', extent=[a0,a1,d0,d1], vmin=min, vmax=max)

    # Add wavelength in top-left corner
    strlambda = '$\lambda$ = '+str(wavelength)+' mm'
    ax.text(xlambda,dmax-0.166*da,strlambda, fontsize=18, color = 'white')

    # Add + sign at the origin
    ax.plot(0.0,0,0,'+',color='white')

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
    cax.xaxis.set_tick_params(labelsize=15, direction='out')
    cax.yaxis.set_tick_params(labelsize=15, direction='out')
    cax.text(.990, 0.30, strflux, fontsize=15, horizontalalignment='right', color='white')

    plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
    plt.clf()


    # --------------------
    # plotting image in polar coordinates
    # --------------------
    if polar_extra_map == 'Yes':
        print('projecting convolved intensity map onto polar plane...')
        Inu_polar2 = np.zeros(nbpixels*nbpixels)
        Inu_polar2 = Inu_polar2.reshape(nbpixels,nbpixels)
        # x-grid contains right ascension offset
        xgrid = a0 + (a1-a0)*np.arange(nbpixels)/(nbpixels-1)
        xgrid = -xgrid
        # x-grid contains declination offset
        ygrid = d0 + (d1-d0)*np.arange(nbpixels)/(nbpixels-1)
        # rc-grid is radius in the polar map
        rmin = 0.0    # specify other value in you want to do a radial zoom
        rmax = np.minimum(minmaxaxis,abs(a0))
        rc   = rmin + (rmax-rmin)* np.arange(nbpixels)/(nbpixels-1)
        # phic-grid is position angle in the polar map
        phic = 2.0*np.pi * np.arange(nbpixels)/nbpixels
        for i in range(nbpixels):
            for j in range(nbpixels):
                # remove 3pi/2 to phic to comply with definition of position angle east of north
                xc = rc[i]*np.cos(phic[j]-1.5*np.pi)    
                yc = rc[i]*np.sin(phic[j]-1.5*np.pi)
                ir = int((xc-xgrid.min())/(xgrid.max()-xgrid.min()) * (nbpixels-1.0))
                jr = int((yc-ygrid.min())/(ygrid.max()-ygrid.min()) * (nbpixels-1.0))
                if ( (ir < nbpixels-1) and (jr < nbpixels-1) ):
                    Inu_polar2[j,i] = convolved_Inu[jr,ir] * (xgrid[ir+1]-xc) * (ygrid[jr+1]-yc) \
                        + convolved_Inu[jr+1,ir]   * (xgrid[ir+1]-xc) * (yc-ygrid[jr]) \
                        + convolved_Inu[jr,ir+1]   * (xc-xgrid[ir]) * (ygrid[jr+1]-yc) \
                        + convolved_Inu[jr+1,ir+1] * (xc-xgrid[ir]) * (yc-ygrid[jr])
                    Inu_polar2[j,i] /= ((xgrid[ir+1]-xgrid[ir]) * (ygrid[jr+1]-ygrid[jr]))
                else:
                    Inu_polar2[j,i] = 0.0

        # name of pdf file for final image
        fileout = re.sub('.fits', '2Dpolar.pdf', inbasename)
        fig = plt.figure(figsize=(8.,8.))
        ax = plt.gca()

        # Labels and ticks
        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.set_xlabel('Position angle / deg')
        ax.set_ylabel('Radius / arcsec')

        # x- and y-ranges
        ax.set_xlim(0,360)      # PA between 0 and 360 degrees
        ax.set_ylim(rmin,rmax)  # radius between rmin and rmax (see above)
        
        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'
        min = Inu_polar2.min()
        max = Inu_polar2.max()
        CM = ax.imshow(np.transpose(Inu_polar2), origin='lower', cmap='nipy_spectral', interpolation='bilinear', vmin=min, vmax=max, extent=[0.0,360.0,rmin,rmax],aspect='auto')

        # Add wavelength in top-left corner
        strlambda = '$\lambda$ = '+str(wavelength)+' mm'
        ax.text(30.0,rmin+0.9*(rmax-rmin),strlambda, fontsize=18, color = 'white')

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="4.5%", pad=0.3)
        cax.yaxis.set_ticks_position('left')
        cax.xaxis.set_ticks_position('top')
        cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=15, direction='out')
        cax.yaxis.set_tick_params(labelsize=15, direction='out')
        cax.text(.990, 0.30, strflux, fontsize=15, horizontalalignment='right', color='white')

        # save in pdf file
        plt.savefig('./'+fileout, bbox_inches='tight', dpi=160)
        plt.clf()

        
print('--------- done! ----------')
