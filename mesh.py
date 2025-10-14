import numpy as np
import math

# import global variables
import par
import sys
import subprocess

# ---------------------
# building mesh arrays 
# ---------------------

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
        # cylindrical radius
        # -----        
        # For 2D hydro simulations, the spherical radii used by
        # RADMC3D are built from the set of cylindrical radii in the
        # FARGO 2D simulation
        try:
            domain_rad = np.loadtxt(directory+"used_rad.dat")  # radial interfaces of grid cells
        except IOError:
            print('IOError')
        self.redge = domain_rad                                # r-edge
        self.rmed = 2.0*(domain_rad[1:]*domain_rad[1:]*domain_rad[1:] - domain_rad[:-1]*domain_rad[:-1]*domain_rad[:-1]) / (domain_rad[1:]*domain_rad[1:] - domain_rad[:-1]*domain_rad[:-1]) / 3.0  # r-center 


        # -----
        # azimuth
        # -----
        '''
        if par.fargo3d == 'No':
            domain_azi = np.loadtxt(directory+"used_azi.dat")  # azimuthal interfaces of grid cells
            self.pedge = np.append(domain_azi[:,1],domain_azi[-1:,2][0])
        else:
            self.pedge = np.linspace(0.,2.*np.pi,self.nsec+1)  # phi-edge
        '''
        self.pedge = np.linspace(0.,2.*np.pi,self.nsec+1)  # phi-edge
        self.pmed = 0.5*(self.pedge[:-1] + self.pedge[1:])     # phi-center

        
        # -----
        # colatitude
        # -----
        if par.fargo3d == 'Yes' and par.hydro2D == 'No' and par.periodicz == 0:  # 3D simulation with fargo3d
            try:
                domain_col = np.loadtxt(directory+"domain_z.dat")  # latitudinal interfaces of grid cells
            except IOError:
                print('IOError')
            
            # check if 3D simulation covers half a disk only in the latitudinal direction 
            command = par.awk_command+' " /^ZMAX/ " '+directory+'/variables.par'
            if sys.version_info[0] < 3:   # python 2.X
                buf = subprocess.check_output(command, shell=True)
            else:                         # python 3.X
                buf = subprocess.getoutput(command)
            zmax = float(buf.split()[1])

            # We can't do mirror symmetry in RADMC3D when scattering_mode_max = 2
            # (i.e., when anisotropic scattering is assumed). We need to define the
            # grid's colatitude on both sides about the disc midplane:
            if np.abs(zmax - 1.57) < 0.01:             # half disk
                ym_lower = domain_col[3:-3]            # lat-edge
                # Below: need to check...
                ym_lower = ym_lower[::-1]
                # then define array of colatitudes below midplane
                ym_upper = np.pi-ym_lower
                # and finally concatenate
                ym = np.concatenate([ym_lower[::-1],ym_upper])
            else:
                ym = domain_col[3:-3]           # lat-edge
            
        else:    # 2D simulation with dusty fargo adsg or fargo3d
            # Again, we need to define the grid's colatitude on both sides about 
            # the disc midplane (defined where theta = pi/2)
            #
            # thmin is set as pi/2 - atan(zmax_over_H*h) with h the gas
            # aspect ratio zmax_over_H = z_max_grid / pressure scale
            # height, which is set in params.dat
            thmin = np.pi/2. - math.atan(par.zmax_over_H*par.aspectratio)
            thmax = np.pi/2.
            if par.polarized_scat == 'No':
                # first define array of colatitudes above midplane
                ymp = np.linspace(np.log10(thmin),np.log10(thmax),self.ncol//2+1)
                # refine towards the midplane
                ym_lower = -1.0*10**(ymp)+thmin+thmax
            else:
                # first define array of colatitudes above midplane
                ymp = np.linspace(thmin,thmax,self.ncol//2+1)
                # no refinement towards the midplane
                ym_lower = -ymp+thmin+thmax
            # then define array of colatitudes below midplane
            ym_upper = np.pi-ym_lower[1:self.ncol//2+1]
            # finally concatenate
            ym = np.concatenate([ym_lower[::-1],ym_upper])

        self.tedge = ym                    # colatitude of cell faces
        self.tmed = 0.5*(ym[:-1] + ym[1:]) # colatitude of cell centers

            
        # -----
        # cylindrical radius
        # -----
        # define number of cells in vertical direction for arrays in
        # cylindrical coordinates
        self.nver = 200  # seems large enough
        # define an array for vertical altitude across the midplane
        zbuf = -self.rmed.max()*np.cos(self.tmed)
        self.zmed = np.linspace(zbuf.min(),zbuf.max(),self.nver)

   
