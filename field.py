import numpy as np
import os
import sys
import subprocess

from mesh import *

# import global variables
import par


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
        if par.hydro2D == 'Yes':
            self.ncol = par.ncol              # colatitude (ncol is a global variable)
        else:
            command = par.awk_command+' " /^NZ/ " ' + directory + 'variables.par'
            # check which version of python we're using
            if sys.version_info[0] < 3:   # python 2.X
                buf = subprocess.check_output(command, shell=True)
            else:                         # python 3.X
                buf = subprocess.getoutput(command)
            self.ncol = int(buf.split()[1])

        Mesh.__init__(self, directory)    # all Mesh attributes inside Field

        # Get units from Fargo simulations
        if par.override_units == 'No':
            if par.fargo3d == 'No':
                # units.dat contains units of mass [kg], length [m], time [s], and temperature [k] 
                cumass, culength, cutime, cutemp = np.loadtxt(directory+"units.dat",unpack=True)
                self.cumass = cumass
                self.culength = culength
                self.cutime = cutime
                self.cutemp = cutemp
            else:
                # get units via variable.par file
                command = par.awk_command+' " /^UNITOFLENGTHAU/ " '+directory+'variables.par'
                # check which version of python we're using
                if sys.version_info[0] < 3:   # python 2.X
                    buf = subprocess.check_output(command, shell=True)
                else:                         # python 3.X
                    buf = subprocess.getoutput(command)
                self.culength = float(buf.split()[1])*1.5e11  #from au to meters
                command = par.awk_command+' " /^UNITOFMASSMSUN/ " '+directory+'variables.par'
                # check which version of python we're using
                if sys.version_info[0] < 3:   # python 2.X
                    buf = subprocess.check_output(command, shell=True)
                else:                         # python 3.X
                    buf = subprocess.getoutput(command)
                self.cumass = float(buf.split()[1])*2e30  #from Msol to kg        
                # unit of temperature = mean molecular weight * 8.0841643e-15 * M / L;
                self.cutemp = 2.35 * 8.0841643e-15 * self.cumass / self.culength
                self.cutime = np.sqrt( self.culength**3 / 6.673e-11 / self.cumass)

        # override units used in Fargo simulations:
        else:
            self.cumass = par.new_unit_mass
            self.culength = par.new_unit_length
            # Deduce new units of time and temperature:
            # T = sqrt( pow(L,3.) / 6.673e-11 / M )
            # U = mmw * 8.0841643e-15 * M / L;
            self.cutime = np.sqrt( self.culength**3 / 6.673e-11 / self.cumass)
            self.cutemp = 2.35 * 8.0841643e-15 * self.cumass / self.culength


        # now, staggering:
        if staggered.count('r')>0:
            self.r = self.redge[:-1] # do not dump last element
        else:
            self.r = self.rmed
            
        self.data = self.__open_field(directory+field,dtype) # scalar data is here.

    def __open_field(self, f, dtype):
        """
        Reading the data
        """
        field = np.fromfile(f, dtype=dtype)
        if par.hydro2D == 'No': #3D simulation with fargo3d
            array = field.reshape(self.ncol,self.nrad,self.nsec) # 3D
            # need to roll field by nsec/2 along azimuthal direction for FARGO3D runs!
            array = np.roll(array, shift=int(self.nsec//2), axis=2)
        else:   # 2D simulation with dusty fargo adsg or fargo3d
            array = field.reshape(self.nrad,self.nsec) # 2D
            if par.fargo3d == 'Yes':
                # need to roll field by nsec/2 along azimuthal direction for FARGO3D runs!
                array = np.roll(array, shift=int(self.nsec//2), axis=1)
        
        return array
