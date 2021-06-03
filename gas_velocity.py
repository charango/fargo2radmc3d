# import global variables
import par

import numpy as np

from mesh import *
from field import *

# =========================
# Microturbulent line broadening
# =========================
def write_gas_microturb():
    
    MTURB = open('microturbulence.inp','w')
    MTURB.write('1 \n')                           # iformat
    MTURB.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells

    microturb = np.zeros((par.gas.ncol,par.gas.nrad,par.gas.nsec))

    if par.turbvel == 'cavity':
        for k in range(par.gas.nsec):
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    if par.gas.rmed[i] < 1.0:
                        myalpha = 3e-2   # inside cavity
                    else:
                        myalpha = 3e-4   # outside cavity
                    # v_turb ~ sqrt(alpha) x isothermal sound speed
                    microturb[j,i,k] = np.sqrt(myalpha * par.kB * gas_temp[j,i,k] / 2.3 / par.mp)  
                    MTURB.write(str(microturb[j,i,k])+' \n')
        if par.verbose:
            print('min and max of microturbulent velocity in cm/s = ',microturb.min(),microturb.max())

    else:
        for k in range(par.gas.nsec):
            for j in range(par.gas.ncol):
                for i in range(par.gas.nrad):
                    microturb[j,i,k] = par.turbvel*1.0e2          # ncol, nrad, nsec in cm/s
                    MTURB.write(str(microturb[j,i,k])+' \n')

    MTURB.close()


# =========================
# Compute gas velocity field on RADMC 3D grid
# =========================
def compute_gas_velocity():
    
    if par.fargo3d == 'Yes':

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
        
        vrad2D   = Field(field='gasvrad'+str(par.on)+'.dat', directory=par.dir).data  # code units
        vrad2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

        vphi2D   = Field(field='gasvtheta'+str(par.on)+'.dat', directory=par.dir).data  # code units
        f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        omegaframe = omega[par.on]

        for phi in range(par.gas.nsec):
            vphi2D[:,phi] += par.gas.rmed*omegaframe
        vphi2D   *= (par.gas.culength*1e2)/(par.gas.cutime) #cm/s

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
    GASVEL = open('gas_velocity.inp','w')
    GASVEL.write('1 \n')                           # iformat
    GASVEL.write(str(par.gas.nrad*par.gas.nsec*par.gas.ncol)+' \n')        # n cells
    for k in range(par.gas.nsec):
        for j in range(par.gas.ncol):
            for i in range(par.gas.nrad):
                GASVEL.write(str(vrad3D[j,i,k])+' '+str(vtheta3D[j,i,k])+' '+str(vphi3D[j,i,k])+' \n')
    GASVEL.close()
    
    del vrad3D, vphi3D, vtheta3D, vrad3D_cyl, vphi3D_cyl
