# import global variables
import par

import numpy as np
import os


# -------------------------
# script calling RADMC3D
# -------------------------
def write_radmc3d_script():
    
    # RT in the dust continuum
    if par.RTdust_or_gas == 'dust':
        command ='radmc3d image lambda '+str(par.wavelength*1e3)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        if par.plot_tau == 'Yes':
            command ='radmc3d image tracetau lambda '+str(par.wavelength*1e3)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        if par.polarized_scat == 'Yes':
            command=command+' stokes'

    # RT in gas lines
    if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both':
        if par.widthkms == 0.0:
            command='radmc3d image iline '+str(par.iline)+' vkms '+str(par.vkms)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
        else:
            command='radmc3d image iline '+str(par.iline)+' widthkms '+str(par.widthkms)+' linenlam '+str(par.linenlam)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
            if par.plot_tau == 'Yes':
                command='radmc3d image tracetau iline '+str(par.iline)+' widthkms '+str(par.widthkms)+' linenlam '+str(par.linenlam)+' npix '+str(par.nbpixels)+' incl '+str(par.inclination)+' posang '+str(par.posangle+90.0)+' phi '+str(par.phiangle)
                #command='radmc3d tausurf 1.0 iline '+str(iline)+' widthkms '+str(widthkms)+' linenlam '+str(linenlam)+' npix '+str(nbpixels)+' incl '+str(inclination)+' posang '+str(posangle+90.0)+' phi '+str(phiangle)

    # optional: second-order ray tracing
    if par.secondorder == 'Yes':
        command=command+' secondorder'

    # write execution script
    if par.verbose == 'Yes':
        print(command)
    SCRIPT = open('script_radmc','w')
    '''
    if par.Tdust_eq_Thydro == 'No':
        SCRIPT.write('radmc3d mctherm; '+command)
    else:
        SCRIPT.write(command)        
    '''
    SCRIPT.write(command)
    SCRIPT.close()
    os.system('chmod a+x script_radmc')


# ---------------------------------------
# write spatial grid in file amr_grid.inp
# ---------------------------------------
def write_AMRgrid(F, R_Scaling=1, Plot=False):

    if par.verbose == 'Yes':
        print("writing spatial grid")
    path_grid='amr_grid.inp'

    grid=open(path_grid,'w')

    grid.write('1 \n')              # iformat/ format number = 1
    grid.write('0 \n')              # Grid style (regular = 0)
    grid.write('101 \n')            # coordsystem: 100 < spherical < 200 
    grid.write('0 \n')              # gridinfo
    grid.write('1 \t 1 \t 1 \n')    # incl x, incl y, incl z

    # spherical radius, colatitude, azimuth
    grid.write(str(F.nrad)+ '\t'+ str(F.ncol)+'\t'+ str(F.nsec)+'\n') 

    # nrad+1 dimension as we need to enter the coordinates of the cells edges
    for i in range(F.nrad + 1):  
        grid.write(str(F.redge[i]*F.culength*1e2)+'\t') # with unit conversion in cm
    grid.write('\n')

    # colatitude
    for i in range(F.ncol + 1):
        grid.write(str(F.tedge[i])+'\t')
    grid.write('\n')

    # azimuth
    for i in range(F.nsec + 1):
        grid.write(str(F.pedge[i])+'\t')
    grid.write('\n')
    
    grid.close()


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

    if par.verbose == 'Yes':
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

    if par.verbose == 'Yes':
        print('writing stars.inp')

    path = 'stars.inp'
    wave = open(path,'w')

    wave.write('\t 2\n') 
    wave.write('1 \t'+str(Nw)+'\n')
    wave.write(str(Rstar*par.R_Sun)+'\t'+str(par.M_Sun)+'\t 0 \t 0 \t 0 \n')
    for i in range(Nw):
        wave.write('\t'+str(waves[i])+'\n')
    wave.write('\t -'+str(Tstar)+'\n')
    wave.close()


# --------------------
# writing radmc3d.inp
# --------------------
def write_radmc3dinp(incl_dust = 1,
                     incl_lines = 0,
                     lines_mode = 1,
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

    if par.verbose == 'Yes':
        print('writing radmc3d.inp')

    RADMCINP = open('radmc3d.inp','w')
    inplines = ["incl_dust = "+str(int(incl_dust))+"\n",
                "incl_lines = "+str(int(incl_lines))+"\n",
                "lines_mode = "+str(int(lines_mode))+"\n",
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

    
# --------------------
# writing lines.inp
# --------------------
def write_lines(specie,lines_mode):

    if par.verbose == 'Yes':
        print("writing lines.inp")
    path_lines='lines.inp'

    lines=open(path_lines,'w')

    lines.write('2 \n')              # <=== Put this to 2
    lines.write('1 \n')              # Nr of molecular or atomic species to be modeled
    # LTE calculations
    if lines_mode == 1:
        lines.write('%s    leiden    0    0    0'%specie)    # incl x, incl y, incl z
    else:
    # non-LTE calculations
        lines.write('%s    leiden    0    0    1\n'%specie)    # incl x, incl y, incl z
        lines.write('h2')
    lines.close()

    # Get molecular data file
    molecular_file = 'molecule_'+str(par.gasspecies)+'.inp'
    if os.path.isfile(molecular_file) == False:
        if par.verbose == 'Yes':
            print('--------- Downloading molecular data file ----------')
        datafile = str(par.gasspecies)
        if par.gasspecies == 'hco+':
            datafile = 'hco+@xpol'
        if par.gasspecies == 'so':
            datafile = 'so@lique'
        if par.gasspecies == 'cs':
            datafile = 'cs'
        command = 'curl -O https://home.strw.leidenuniv.nl/~moldata/datafiles/'+datafile+'.dat'
        print(command)
        os.system(command)
        command = 'mv '+datafile+'.dat molecule_'+str(par.gasspecies)+'.inp'
        os.system(command)
