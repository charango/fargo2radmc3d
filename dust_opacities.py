# import global variables
import par

import os

from makedustopac import *

def compute_dust_opacities():

    # Calculation of opacities uses the python scripts makedustopac.py and bhmie.py
    # which were written by C. Dullemond, based on the original code by Bohren & Huffman.
    
    logawidth = 0.05          # Smear out the grain size by 5% in both directions
    na        = 20            # Use 10 grain size samples per bin size
    chop      = 1.            # Remove forward scattering within an angle of 5 degrees
    extrapol  = True          # Extrapolate optical constants beyond its wavelength grid, if necessary
    verbose   = False         # If True, then write out status information
    ntheta    = 181           # Number of scattering angle sampling points
    optconstfile = os.path.expanduser(par.opacity_dir)+'/'+par.species+'.lnk'    # link to optical constants file

    # The material density in gram / cm^3
    graindens = 2.0 # default density in g / cc
    if (par.species == 'mix_2species_porous' or par.species == 'mix_2species_porous_ice' or par.species == 'mix_2species_porous_ice70'):
        graindens = 0.1 # g / cc
    if (par.species == 'mix_2species' or par.species == 'mix_2species_60silicates_40ice'):
        graindens = 1.7 # g / cc
    if par.species == 'mix_2species_ice70':
        graindens = 1.26 # g / cc
    if par.species == 'mix_2species_60silicates_40carbons':
        graindens = 2.7 # g / cc
    
    # Set up a wavelength grid (in cm) upon which we want to compute the opacities
    # 1 micron -> 1 cm
    lamcm     = 10.0**np.linspace(0,4,200)*1e-4   

    # Set up an angular grid for which we want to compute the scattering matrix Z
    theta     = np.linspace(0.,180.,ntheta)

    for ibin in range(int(par.nbin)):
        # median grain size in cm in current bin size:
        if par.fargo3d == 'No':
            agraincm = 10.0**(0.5*(np.log10(1e2*par.bins[ibin]) + np.log10(1e2*par.bins[ibin+1])))
        else:
            agraincm = 1e2*par.dust_size[ibin]
        print('====================')
        print('bin ', ibin+1,'/',par.nbin)
        print('grain size [cm]: ', agraincm, ' with grain density [g/cc] = ', graindens)
        print('====================')
        pathout    = par.species+str(ibin)
        opac       = compute_opac_mie(optconstfile,graindens,agraincm,lamcm,theta=theta,
                                      extrapolate=extrapol,logawidth=logawidth,na=na,
                                      chopforward=chop,verbose=verbose)
        if (par.scat_mode >= 3):
            print("Writing dust opacities in dustkapscatmat* files")
            write_radmc3d_scatmat_file(opac,pathout)
        else:
            print("Writing dust opacities in dustkappa* files")
            write_radmc3d_kappa_file(opac,pathout)


# -----------------
# writing dustopac
# -----------------
def write_dustopac(species=['ac_opct', 'Draine_Si'],nbin=20):
    print('writing dustopac.inp')
    hline = "-----------------------------------------------------------------------------\n"
    OPACOUT = open('dustopac.inp','w')

    lines0=["2 \t iformat (2)\n",
            str(nbin)+" \t species\n",
            hline]
    OPACOUT.writelines(lines0)
    # put first element to 10 if dustkapscatmat_species.inp input file, or 1 if dustkappa_species.inp input file
    if (par.scat_mode >= 3):
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


# ---------------
# read opacities
# ---------------
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


# -------------------
# plotting opacities
# -------------------
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
        
    lbda1 *= 1e-3  # in mm

    plt.loglog(sizes, absorption1, lw=2., linestyle = 'solid', color = par.c20[1], label='$\kappa_{abs}$ at '+str(lbda1)+' mm')
    plt.loglog(sizes, absorption1+scattering1, lw=2., linestyle = 'dashed', color = par.c20[1], label='$\kappa_{abs}$+$\kappa_{sca}$ at '+str(lbda1)+' mm')
    plt.legend()

    plt.ylim(absorption1.min(),(absorption1+scattering1).max())
    filesaveopac = 'opacities_'+species+'.pdf'
    plt.savefig('./'+filesaveopac, bbox_inches='tight', dpi=160)
    plt.clf()
