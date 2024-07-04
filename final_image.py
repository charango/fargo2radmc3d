# import global variables
import par

import os
import numpy as np
import re
from copy import deepcopy
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits

import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogLocator, LogFormatter)

from beam import *
from polar import *

# -------------------
# Produce final image
# -------------------
def produce_final_image(input=''):

    if input == '':
        f = fits.open(par.outputfitsfile)
        # remove .fits extension
        outfile = os.path.splitext(par.outputfitsfile)[0]
    elif input == 'dust':
        print('here par.RTdust_or_gas = ', par.RTdust_or_gas)
        f = fits.open(par.outputfitsfile_dust)
        # remove .fits extension
        outfile = os.path.splitext(par.outputfitsfile_dust)[0]

    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
        outfile += '_mask'+str(par.mask_radius)
        outfile += '_'+str(par.polarized_scat_field)
        outfile += '_r2'+str(par.r2_rescale)

    if par.log_colorscale == 'Yes':
        outfile += '_logYes'
        
    # add bmaj information
    outfile = outfile + '_bmaj'+str(par.bmaj) + '_bmin'+str(par.bmin)
    outfile = outfile+'.fits'
        
    hdr = f[0].header
    # pixel size converted from degrees to arcseconds
    cdelt = np.abs(hdr['CDELT1']*3600.)

    # get wavelength and convert it from microns to mm
    lbda0 = hdr['LBDAMIC']*1e-3

    # a) case with no polarized scattering: fits file directly contains raw intensity field
    if par.polarized_scat == 'No':
        
        nx = hdr['NAXIS1']
        ny = hdr['NAXIS2']
        raw_intensity = f[0].data
        
        if (par.recalc_radmc == 'No' and par.plot_tau == 'No' and par.verbose == 'Yes'):
            print("Total flux [Jy] = "+str(np.sum(raw_intensity)))   # sum over pixels

        # check beam is correctly handled by inserting a source point at the
        # origin of the raw intensity image
        if par.check_beam == 'Yes':
            raw_intensity[:,:] = 0.0
            raw_intensity[nx//2-1,ny//2-1] = 1.0
            
        # Add white (Gaussian) noise to raw flux image to simulate effects of 'thermal' noise
        if (par.add_noise == 'Yes' and par.RTdust_or_gas == 'dust' and par.plot_tau == 'No'):
            # beam area in pixel^2
            beam =  (np.pi/(4.*np.log(2.)))*par.bmaj*par.bmin/(cdelt**2.)
            # noise standard deviation in Jy per pixel (I've checked the expression below works well)
            noise_dev_std_Jy_per_pixel = par.noise_dev_std / np.sqrt(0.5*beam)  # 1D
            # noise array
            noise_array = np.random.normal(0.0,noise_dev_std_Jy_per_pixel,size=par.nbpixels*par.nbpixels)
            noise_array = noise_array.reshape(par.nbpixels,par.nbpixels)
            raw_intensity += noise_array
            
        if par.brightness_temp=='Yes':
            # beware that all units are in cgs! We need to convert
            # 'intensity' from Jy/pixel to cgs units!
            # pixel size in each direction in cm
            pixsize_x = cdelt*par.distance*par.au
            pixsize_y = pixsize_x
            # solid angle subtended by pixel size
            pixsurf_ster = pixsize_x*pixsize_y/par.distance/par.distance/par.pc/par.pc   
            # convert intensity from Jy/pixel to erg/s/cm2/Hz/sr
            intensity_buf = raw_intensity/1e23/pixsurf_ster
            # beware that lbda0 is in mm right now, we need to have it in cm in the expression below
            raw_intensity = (par.h*par.c/par.kB/(lbda0*1e-1))/np.log(1.+2.*par.h*par.c/intensity_buf/pow((lbda0*1e-1),3.))
            #raw_intensity = np.nan_to_num(raw_intensity)
            

    # b) case with polarized scattering: fits file contains raw Stokes vectors
    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
        
        cube = f[0].data
        I = cube[0,:,:]
        Q = cube[1,:,:]
        U = cube[2,:,:]
        (nx, ny) = Q.shape
         
        if par.add_noise == 'Yes':
            # add noise to Q and U Stokes arrays
            # noise array
            noise_array_Q = np.random.normal(0.0,0.01*Q.max(),size=par.nbpixels*par.nbpixels)
            noise_array_Q = noise_array_Q.reshape(par.nbpixels,par.nbpixels)
            Q += noise_array_Q
            noise_array_U = np.random.normal(0.0,0.01*U.max(),size=par.nbpixels*par.nbpixels)
            noise_array_U = noise_array_U.reshape(par.nbpixels,par.nbpixels)
            U += noise_array_U
            noise_array_I = np.random.normal(0.0,0.01*I.max(),size=par.nbpixels*par.nbpixels)
            noise_array_I = noise_array_I.reshape(par.nbpixels,par.nbpixels)
            I += noise_array_I

        # define theta angle for calculation of Q_phi below (Avenhaus+
        # 14). Expression (and sign) for theta checked by comparing
        # Qphi with polarised intensity P (both should be usually
        # identical, see Avenhaus+ 14b)
        x = np.arange(1,nx+1)
        y = np.arange(1,ny+1)
        XXs,YYs = np.meshgrid(x,y)
        X0 = nx/2-1
        Y0 = ny/2-1
        rrs = np.sqrt((XXs-X0)**2+(YYs-Y0)**2)
        theta = np.arctan2(-(XXs-X0),(YYs-Y0)) # between -pi and pi
    
        # add mask in polarized intensity Qphi image if mask_radius != 0
        if par.mask_radius != 0.0:
            pillbox = np.ones((nx,ny))
            imaskrad = par.mask_radius/cdelt  # since cdelt is pixel size in arcseconds
            pillbox[np.where(rrs<imaskrad)] = 0.

            
    # ------------
    # smooth image
    # ------------
    # beam area in pixel^2
    beam =  (np.pi/(4.*np.log(2.)))*par.bmaj*par.bmin/(cdelt**2.)
    # stdev lengths in pixel
    stdev_x = (par.bmaj/(2.*np.sqrt(2.*np.log(2.)))) / cdelt
    stdev_y = (par.bmin/(2.*np.sqrt(2.*np.log(2.)))) / cdelt

    # a) case with no polarized scattering
    if (par.polarized_scat == 'No' and par.plot_tau == 'No'):
        # Call to Gauss_filter function
        if par.moment_order != 1:
            smooth = Gauss_filter(raw_intensity, stdev_x, stdev_y, par.bpaangle, Plot=False)
        if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:
            smooth = raw_intensity

        # convert image from Jy/pixel to mJy/beam or microJy/beam
        # could be refined...
        if par.brightness_temp=='Yes':
            convolved_intensity = smooth 
        if par.brightness_temp=='No':
            convolved_intensity = smooth * 1e3 * beam   # mJy/beam

        strflux = 'Flux of continuum emission [mJy/beam]'
        if par.gasspecies == 'co':
            strgas = r'$^{12}$CO'
        elif par.gasspecies == '13co':
            strgas = r'$^{13}$CO'
        elif par.gasspecies == 'c17o':
            strgas = r'C$^{17}$O'
        elif par.gasspecies == 'c18o':
            strgas = r'C$^{18}$O'
        elif par.gasspecies == 'hco+':
            strgas = r'HCO+'
        elif par.gasspecies == 'so':
            strgas = r'SO'
        else:
            strgas = str(par.gasspecies).upper()  # capital letters
        if par.gasspecies != 'so':
            strgas+=r' ($%d \rightarrow %d$)' % (par.iline,par.iline-1)
        if par.gasspecies == 'so' and par.iline == 14:
            strgas+=r' ($5_6 \rightarrow 4_5$)'

    
        if par.brightness_temp=='Yes':
            # Gas RT and a single velocity channel
            if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.widthkms == 0.0:
                strflux = strgas+' brightness temperature [K]'
            # Gas RT and mooment order 0 map
            if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 0 and par.widthkms != 0.0:
                strflux = strgas+' integrated brightness temperature [K km/s]'
            if par.RTdust_or_gas == 'dust':
                strflux = r'Brightness temperature [K]'
                
        else:
            # Gas RT and a single velocity channel
            if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.widthkms == 0.0:
                strflux = strgas+' intensity [mJy/beam]'
            # Gas RT and mooment order 0 map
            if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 0 and par.widthkms != 0.0:
                strflux = strgas+' integrated intensity [mJy/beam km/s]'
            if convolved_intensity.max() < 1.0 and ('max_colorscale' in open('params.dat').read()):
                if not(par.max_colorscale == '#'):
                    if not(par.max_colorscale > 1.0):
                        convolved_intensity = smooth * 1e6 * beam   # microJy/beam
                        par.max_colorscale *= 1e3
                        strflux = r'Flux of continuum emission [$\mu$Jy/beam]'
                        # Gas RT and a single velocity channel
                        if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.widthkms == 0.0:
                            strflux = strgas+' intensity [$\mu$Jy/beam]'
                        if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 0 and par.widthkms != 0.0:
                            strflux = strgas+' integrated intensity [$\mu$Jy/beam km/s]'

        #
        if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:
            convolved_intensity = smooth
            # this is actually 'raw_intensity' since for moment 1 maps
            # the intensity in each channel map is already convolved,
            # so that we do not convolve a second time!...
            strflux = strgas+' velocity [km/s]'

    #
    if par.plot_tau == 'Yes':
        convolved_intensity = raw_intensity
        strflux = r'Absorption optical depth $\tau'

    # b) case with polarized scattering
    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
        I_smooth = Gauss_filter(I,stdev_x,stdev_y,par.bpaangle,Plot=False)
        Q_smooth = Gauss_filter(Q,stdev_x,stdev_y,par.bpaangle,Plot=False)
        U_smooth = Gauss_filter(U,stdev_x,stdev_y,par.bpaangle,Plot=False)
        if par.mask_radius != 0.0:
            pillbox_smooth = Gauss_filter(pillbox, stdev_x, stdev_y, par.bpaangle, Plot=False)
            I_smooth *= pillbox_smooth
            Q_smooth *= pillbox_smooth
            U_smooth *= pillbox_smooth
        Q_phi = Q_smooth * np.cos(2*theta) + U_smooth * np.sin(2*theta)  # (P_perp in Avenhaus+ 14)
        if par.polarized_scat_field == 'Qphi':
            convolved_intensity = Q_phi
            strflux = r'$Q_{\phi}$ [arb. units]'
        if par.polarized_scat_field == 'I':
            convolved_intensity = I_smooth
            strflux = 'Stokes I [arb. units]'
        if par.polarized_scat_field == 'PI':
            convolved_intensity = np.sqrt( Q_smooth*Q_smooth + U_smooth*U_smooth )
            strflux = 'Polarized intensity [arb. units]'

    # -------------------------------------
    # SP: save convolved flux map solution to fits 
    # -------------------------------------
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32    
    hdu.header['NAXIS'] = 2  # 2
    hdu.header['NAXIS1'] = par.nbpixels
    hdu.header['NAXIS2'] = par.nbpixels
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = hdr['CDELT1']
    hdu.header['CDELT2'] = hdr['CDELT2']
    hdu.header['LBDAMIC'] = hdr['LBDAMIC']
    hdu.header['CUNIT1'] = 'deg     '
    hdu.header['CUNIT2'] = 'deg     '
    hdu.header['CRPIX1'] = float((par.nbpixels+1.)/2.)
    hdu.header['CRPIX2'] = float((par.nbpixels+1.)/2.)
    if strflux == 'Flux of continuum emission [mJy/beam]':
        hdu.header['BUNIT'] = 'milliJY/BEAM'
    if strflux == r'Flux of continuum emission [$\mu$Jy/beam]':
        hdu.header['BUNIT'] = 'microJY/BEAM'
    if strflux == '':
        hdu.header['BUNIT'] = ''
    hdu.header['BTYPE'] = 'FLUX DENSITY'
    hdu.header['BSCALE'] = 1
    hdu.header['BZERO'] = 0
    del hdu.header['EXTEND']
    hdu.data = convolved_intensity
    inbasename = os.path.basename('./'+outfile)
    jybeamfileout=re.sub('.fits', '_JyBeam.fits', inbasename)
    hdu.writeto(jybeamfileout, overwrite=True)

    # ----------------------------
    # if polarised imaging, first de-project Qphi or PI image to multiply by R^2
    # then re-project back
    # ----------------------------
    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes' and par.r2_rescale == 'Yes':
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
        image_rotated = ndimage.rotate(image_centered, par.posangle, reshape=False)
        fileout_rotated = re.sub('.fits', 'rotated.fits', jybeamfileout)
        fits.writeto(fileout_rotated, image_rotated, hdr1, overwrite=True)
        hdr2 = deepcopy(hdr1)
        cosi = np.cos(par.inclination_input*np.pi/180.)
        hdr2['CDELT1']=hdr2['CDELT1']*cosi

        # Then deproject with inclination via gridding interpolation function and hdr2
        image_stretched = gridding(fileout_rotated,hdr2)

        # rescale stretched image by r^2
        nx = hdr2['NAXIS1']
        ny = hdr2['NAXIS2']
        cdelt_polar = abs(hdr2['CDELT1']*3600)  # in arcseconds
        (x0,y0) = (nx/2, ny/2)
        mymax = 0.0
        for j in range(nx):
            for k in range(ny):
                dx = (j-x0)*cdelt_polar
                dy = (k-y0)*cdelt_polar
                rad = np.sqrt(dx*dx + dy*dy)
                image_stretched[j,k] *= (rad*rad)
                    
        # Normalize PI intensity
        image_stretched /= image_stretched.max()
        fileout_stretched = re.sub('.fits', 'stretched.fits', jybeamfileout)
        fits.writeto(fileout_stretched, image_stretched, hdr2, overwrite=True)

        # Then deproject via gridding interpolatin function and hdr1
        image_destretched = gridding(fileout_stretched,hdr1)

        # and finally de-rotate by -position angle
        final_image = ndimage.rotate(image_destretched, -par.posangle, reshape=False)

        # save final fits
        inbasename = os.path.basename('./'+outfile)
        if par.add_noise == 'Yes':
            substr='_wn'+str(par.noise_dev_std)+'_JyBeam.fits' 
            jybeamfileout=re.sub('.fits', substr, inbasename)
        else:
            jybeamfileout=re.sub('.fits', '_JyBeam.fits', inbasename)
        fits.writeto(jybeamfileout,final_image,hdr1,overwrite=True)
        convolved_intensity = final_image

        # remove unnecessary fits files
        command = 'rm -f '+fileout_centered+' '+fileout_rotated+' '+fileout_stretched
        os.system(command)

    
    # --------------------
    # plotting image panel
    # --------------------
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rc('font', family='Arial') 
    '''
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Helvetica']
    '''
    fontcolor='white'

    # name of pdf file for final image
    fileout = re.sub('.fits', '.pdf', jybeamfileout)
    fig = plt.figure(figsize=(8.,8.))
    ax = plt.gca()
    plt.subplots_adjust(left=0.17, right=0.94, top=0.90, bottom=0.1)

    # Set x-axis orientation, x- and y-ranges
    # Convention is that RA offset increases leftwards (ie,
    # east is to the left), while Dec offset increases from
    # bottom to top (ie, north is the top)
    if ( (nx % 2) == 0):
        dpix = 0.5
    else:
        dpix = 0.0
    dpix = 0.0
    a0 = cdelt*(nx//2.-dpix)   # >0
    a1 = -cdelt*(nx//2.+dpix)  # <0
    d0 = -cdelt*(nx//2.-dpix)  # <0
    d1 = cdelt*(nx//2.+dpix)   # >0
    # da positive definite
    if (par.minmaxaxis < abs(a0)):
        da = par.minmaxaxis
    else:
        da = np.maximum(abs(a0),abs(a1))

    # CB (June 2023): if da too small, RA and DEC offsets are displayed in mas and not in arcsec
    if np.abs(da) < 0.01:
        a0 *= 1e3
        a1 *= 1e3
        d0 *= 1e3
        d1 *= 1e3
        da *= 1e3
        par.minmaxaxis *= 1e3
        par.bmaj *= 1e3
        par.bmin *= 1e3
        ax.set_xlabel('RA offset [mas]')
        ax.set_ylabel('Dec offset [mas]')
        strylabel_polar = 'Radius [mas]'
    else:
        ax.set_xlabel('RA offset [arcsec]')
        ax.set_ylabel('Dec offset [arcsec]')
        strylabel_polar = 'Radius [arcsec]'
        
    mina = da
    maxa = -da
    xlambda = mina - 0.166*da
    ax.set_ylim(-da,da)
    ax.set_xlim(da,-da)      # x (=R.A.) increases leftward
    dmin = -da
    dmax = da

    # x- and y-ticks and labels
    ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.yaxis.set_major_locator(plt.MaxNLocator(7))
    #ax.set_xticks(ax.get_yticks())    # set same ticks in x and y in cartesian
    #ax.set_yticks(ax.get_xticks())    # set same ticks in x and y in cartesian 
    

    # Normalization: linear or logarithmic scale
    if par.min_colorscale == '#':
        min = convolved_intensity.min()
        if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
            min = 0.0
    else:
        min = par.min_colorscale
    if par.max_colorscale == '#':
        max = convolved_intensity.max()
        if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
            max = 1.0
    else:
        max = par.max_colorscale
    if par.log_colorscale == 'Yes':
        if par.min_colorscale == '#':
            min = 1e-2*max
        else:
            min = par.min_colorscale
        # avoid negative values of array
        convolved_intensity[convolved_intensity <= min] = min

    if par.log_colorscale == 'Yes':
        mynorm = matplotlib.colors.LogNorm(vmin=min,vmax=max)
    else:
        mynorm = matplotlib.colors.Normalize(vmin=min,vmax=max)

    print(min,convolved_intensity.min())
    print(max,convolved_intensity.max())
    # imshow does a bilinear interpolation. You can switch it off by putting
    # interpolation='none'
    CM = ax.imshow(convolved_intensity, origin='lower', cmap=par.mycolormap, interpolation='bilinear', extent=[a0,a1,d0,d1], norm=mynorm, aspect='auto')
    print('=========')
    print(a0,a1,d0,d1)
    print('=========')

    # Add wavelength/user-defined string in top-left/right corners
    if ( ('display_label' in open('params.dat').read()) and (par.display_label != '#') ):
        strlambda = par.display_label
    else:
        strlambda = '$\lambda$='+str(round(lbda0, 2))+'mm' # round to 2 decimals
        if lbda0 < 0.01:
            strlambda = '$\lambda$='+str(round(lbda0*1e3,2))+'$\mu$m'
    ax.text(xlambda,dmax-0.166*da,strlambda, fontsize=20, color = 'white',weight='bold',horizontalalignment='left')

    if ( (('display_time' in open('params.dat').read()) and (par.display_time == 'Yes')) or (('spot_planet' in open('params.dat').read()) and (par.spot_planet == 'Yes')) ):
        import itertools
        with open(par.dir+"/orbit0.dat") as f_in:
            firstline_orbitfile = np.genfromtxt(itertools.islice(f_in, 0, 1, None), dtype=float)
        apla = firstline_orbitfile[2]
        fargo3d = 'No'
        if os.path.isfile(par.dir+'/summary0.dat') == True:
            fargo3d = 'Yes'   
        if fargo3d == 'Yes':
            f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        else:
            f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(par.dir+"/planet0.dat",unpack=True)
        
        # check if planet0.dat file has only one line or more!
        if isinstance(xpla, (list, tuple, np.ndarray)) == True:
            omegaframe = omega[par.on]
            time_in_code_units = round(date[par.on]/2./np.pi/apla/np.sqrt(apla),1)
        else:
            omegaframe = omega
            time_in_code_units = round(date/2./np.pi/apla/np.sqrt(apla),1)
        strtime = str(time_in_code_units)+' Porb'

        # Add time in top-right corner
        if (('display_time' in open('params.dat').read()) and (par.display_time == 'Yes')):
            ax.text(-xlambda,dmax-0.166*da,strtime, fontsize=20, color = 'white',weight='bold',horizontalalignment='right')

        # Spot planet position in sky-plane
        if (('spot_planet' in open('params.dat').read()) and (par.spot_planet == 'Yes')):
            xp = xpla[par.on] # in disc simulation plane
            yp = ypla[par.on] # in disc simulation plane

            # NEW (March 2024): case there is more than just one planet!
            if os.path.isfile(par.dir+"/planet1.dat") == True:
                print('Two planets to be displayed..')
                if fargo3d == 'Yes':
                    f1, xpla, ypla, f4, f5, f6, f7, f8, date, omega = np.loadtxt(par.dir+"/planet1.dat",unpack=True)
                else:
                    f1, xpla, ypla, f4, f5, f6, f7, date, omega, f10, f11 = np.loadtxt(par.dir+"/planet1.dat",unpack=True)
                xp = np.array([xp,xpla[par.on]])
                yp = np.array([yp,ypla[par.on]])
                print('xp = ', xp)
                print('yp = ', yp)
                
            if par.xaxisflip == 'Yes':
                xp = -xp
            else:
                xp = -xp  # cuidadin
                yp = -yp  # cuidadin
            print('planet position on simulation plane [code units]: xp = ', xp, ' and yp = ', yp)
            
            # convert from simulation units to arcsecond:
            if par.recalc_radmc == 'Yes':
                culength = par.gas.culength
            else:
                if par.override_units == 'No':
                    if par.fargo3d == 'No':
                        cumass, culength, cutime, cutemp = np.loadtxt(par.dir+"/units.dat",unpack=True)
                    else:
                        import sys
                        import subprocess
                        command = par.awk_command+' " /^UNITOFLENGTHAU/ " '+par.dir+'/variables.par'
                        # check which version of python we're using
                        if sys.version_info[0] < 3:   # python 2.X
                            buf = subprocess.check_output(command, shell=True)
                        else:                         # python 3.X
                            buf = subprocess.getoutput(command)
                        culength = float(buf.split()[1])*1.5e11  #from au to meters
                else:
                    culength = par.new_unit_length # in meters
            code_unit_of_length = 1e2*culength # in cm
            
            xp *= code_unit_of_length/par.au/par.distance
            yp *= code_unit_of_length/par.au/par.distance
            print('planet position on simulation plane [arcseconds]: xp = ', xp, ' and yp = ', yp)
            phiangle_in_rad = par.phiangle*np.pi/180.0
            # add 90 degrees to be consistent with RADMC3D's convention for position angle
            posangle_in_rad = (par.posangle+90.0)*np.pi/180.0
            inclination_in_rad = par.inclination*np.pi/180.0
            xp_sky =  (xp*np.cos(phiangle_in_rad)+yp*np.sin(phiangle_in_rad))*np.cos(posangle_in_rad) + (-xp*np.sin(phiangle_in_rad)+yp*np.cos(phiangle_in_rad))*np.cos(inclination_in_rad)*np.sin(posangle_in_rad)
            yp_sky = -(xp*np.cos(phiangle_in_rad)+yp*np.sin(phiangle_in_rad))*np.sin(posangle_in_rad) + (-xp*np.sin(phiangle_in_rad)+yp*np.cos(phiangle_in_rad))*np.cos(inclination_in_rad)*np.cos(posangle_in_rad)
            print('planet position on sky-plane [arcseconds]: xp_sky = ', xp_sky, ' and yp_sky = ', yp_sky)
            ax.plot(xp_sky,yp_sky,'x',color='white',markersize=10)
    
    # Add + sign at the origin
    ax.plot(0.0,0.0,'+',color='white',markersize=10)
    '''
    if check_beam == 'Yes':
        ax.contour(convolved_intensity,levels=[0.5*convolved_intensity.max()],color='black', linestyles='-',origin='lower',extent=[a0,a1,d0,d1])
    '''

    print('min and max of moment 1 velocity = ',convolved_intensity.min(),convolved_intensity.max())
    
    # Add a few contours in order 1 moment maps for gas emission
    if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:
        ax.contour(convolved_intensity,levels=10,color='black', linestyles='-',origin='lower',extent=[a0,a1,d0,d1])
    
    # plot beam
    if par.plot_tau == 'No':
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=[xlambda,dmin+0.166*da], width=par.bmin, height=par.bmaj, angle=par.bpaangle+90.0)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('white')
        e.set_alpha(0.8)
        ax.add_artist(e)
    # plot beam
    '''
    if par.check_beam == 'Yes':
        from matplotlib.patches import Ellipse
        e = Ellipse(xy=[0.0,0.0], width=par.bmin, height=par.bmaj, angle=par.bpaangle+90.0)
        e.set_clip_box(ax.bbox)
        e.set_facecolor('white')
        e.set_alpha(1.0)
        ax.add_artist(e)
    '''
        
    # plot color-bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="2.5%", pad=0.12)
    cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
    cax.xaxis.tick_top()
    cax.xaxis.set_tick_params(labelsize=20, direction='out')
    cax.xaxis.set_major_locator(plt.MaxNLocator(6))
    if par.log_colorscale == 'Yes':
        cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        # CB / Yinhao Wu: if the colorbar does not display properly, try to decrease numticks to 4 max.
    
    # title on top
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(strflux)
    cax.xaxis.labelpad = 8

    plt.savefig('./'+fileout, dpi=160)
    plt.clf()

    # =====================
    # Compute deprojection and polar expansion (SP)
    # =====================
    if par.deproj_polar == 'Yes':
        
        currentdir = os.getcwd()
        alpha_min = 0.;          # deg, PA of offset from the star
        Delta_min = 0.;          # arcsec, amplitude of offset from the star
        RA = 0.0  # if input image is a prediction, star should be at the center
        DEC = 0.0 # note that this deprojection routine works in WCS coordinates

        # CUIDADIN! testing purposes
        cosi = np.cos(par.inclination_input*np.pi/180.)

        if par.verbose == 'Yes':
            print('deprojection around PA [deg] = ',par.posangle)
            print('and inclination [deg] = ',par.inclination_input)
        
        # makes a new directory "deproj_polar_dir" and calculates a number
        # of products: copy of the input image [_fullim], centered at
        # (RA,DEC) [_centered], deprojection by cos(i) [_stretched], polar
        # image [_polar], etc. Also, a _radial_profile which is the
        # average radial intensity.
        exec_polar_expansions(jybeamfileout,'deproj_polar_dir',par.posangle,cosi,RA=RA,DEC=DEC,
                              alpha_min=alpha_min, Delta_min=Delta_min,
                              XCheckInv=False,DoRadialProfile=False,
                              DoAzimuthalProfile=False,PlotRadialProfile=False,
                              zoomfactor=1.)
        
        # Save polar fits in current directory
        fileout = re.sub('.pdf', '_polar.fits', fileout)
        command = 'cp deproj_polar_dir/'+fileout+' .'
        os.system(command)

        filein = re.sub('.pdf', '_polar.fits', fileout)
        # Read fits file with deprojected field in polar coordinates
        f = fits.open(filein)
        convolved_intensity = f[0].data    # uJy/beam

        if par.log_colorscale == 'Yes':
            convolved_intensity[convolved_intensity <= min] = min # min defined above

        # azimuthal shift such that PA=0 corresponds to y-axis pointing upwards, and
        # increases counter-clockwise from that axis
        if par.xaxisflip == 'Yes':
            jshift = int(par.nbpixels/4 + (90.0-par.posangle)*par.nbpixels/360.0)
        else:
            jshift = int(par.nbpixels/4 + (90.0-par.posangle)*par.nbpixels/360.0)  # ?? check!
        convolved_intensity = np.roll(convolved_intensity, shift=-jshift, axis=1)
    
        
        # -------------------------------
        # plot image in polar coordinates
        # -------------------------------
        fileout = re.sub('.fits', '.pdf', filein)
        fig = plt.figure(figsize=(8.,8.))
        plt.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.09)
        ax = plt.gca()

        # Set x- and y-ranges
        ax.set_xlim(-180,180)          # PA relative to Clump 1's
        if (par.minmaxaxis < np.maximum(abs(a0),abs(a1))):
            ymax = par.minmaxaxis
        else:
            ymax = np.maximum(abs(a0),abs(a1))
        ax.set_ylim(0,ymax)      # Deprojected radius in arcsec

        if ( (nx % 2) == 0):
            dpix = 0.5
        else:
            dpix = 0.0
        a0 = cdelt*(nx//2.-dpix)   # >0

        ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
        ax.set_xticks((-180,-120,-60,0,60,120,180))
        #ax.set_yticks((0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))
        ax.set_xlabel('Position Angle [deg]')
        ax.set_ylabel(strylabel_polar)

        
        # imshow does a bilinear interpolation. You can switch it off by putting
        # interpolation='none'. Note that mynorm has already been defined above
        CM = ax.imshow(convolved_intensity, origin='lower', cmap=par.mycolormap, interpolation='bilinear', extent=[-180,180,0,np.maximum(abs(a0),abs(a1))], norm=mynorm, aspect='auto')   # (left, right, bottom, top)

        # Add wavelength in top-left corner
        ax.text(-160,0.95*ymax,strlambda,fontsize=20,color='white',weight='bold',horizontalalignment='left',verticalalignment='top')

        # Option: add time in top-right corner
        if ('display_time' in open('params.dat').read()) and (par.display_time == 'Yes'):
            ax.text(160,0.95*ymax,strtime,fontsize=20,color='white',weight='bold',horizontalalignment='right',verticalalignment='top')

        # Option: spot planet position
        if (('spot_planet' in open('params.dat').read()) and (par.spot_planet == 'Yes')):
            import math
            xp_proj = xp_sky
            yp_proj = yp_sky / np.abs(np.cos(inclination_in_rad)) # cuidadin
            print('xp_proj = ', xp_proj)
            print('yp_proj = ', yp_proj)
            rp_proj = np.sqrt(xp_proj*xp_proj + yp_proj*yp_proj)

            if isinstance(xp_proj, float) == True:
                xp_proj = [xp_proj]
                yp_proj = [yp_proj]
            
            # in degrees, measured earth of north
            dim = len(xp_proj)
            tp_proj = np.zeros(dim)
            for i in range(dim):
                tp_proj[i] = (np.pi/2.0 + math.atan2(-yp_proj[i],xp_proj[i]))*180./np.pi
                if tp_proj[i] < -180.0:
                    tp_proj[i] += 360.0
                if tp_proj[i] > 180.0:
                    tp_proj[i] -= 360.0
            print('planet position on deprojected sky-plane: rp ["] = ', rp_proj, ' and tp_proj [deg] = ', tp_proj)
            ax.plot(tp_proj,rp_proj,'x',color='white',markersize=10)

        # plot color-bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="2.5%", pad=0.12)
        cb =  plt.colorbar(CM, cax=cax, orientation='horizontal')
        cax.xaxis.tick_top()
        cax.xaxis.set_tick_params(labelsize=20, direction='out')
        cax.xaxis.set_major_locator(plt.MaxNLocator(6))
        if par.log_colorscale == 'Yes':
            cax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
        # title on top
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(strflux)
        cax.xaxis.labelpad = 8

        plt.savefig('./'+fileout, dpi=160)
        plt.clf()

        # Plot azimuthal average of final intensity (GWF)
        if par.axi_intensity == 'Yes' and par.moment_order != 1:
            average_convolved_intensity=np.zeros(par.nbpixels)
            for j in range(par.nbpixels):
                for i in range(par.nbpixels):
                    average_convolved_intensity[j]+=convolved_intensity[j][i]/par.nbpixels

            rkarr = np.linspace(0,np.maximum(abs(a0),abs(a1)),par.nbpixels) # radius in arcseconds
            
            nb_noise = 0
            if par.add_noise == 'Yes':
                nb_noise = 1

            file = open('axiconv%d.dat' % (nb_noise),'w')
            for kk in range(par.nbpixels):
                file.write('%s\t%s\t%s\n' % (str(rkarr[kk]),str(np.mean(convolved_intensity[kk])),str(np.std(convolved_intensity[kk]))))
            file.close()

            fig = plt.figure(figsize=(8.,8.))
            ax = plt.gca()
            plt.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.09)

            ax.plot(rkarr,average_convolved_intensity, color=par.c20[0])
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.tick_params(top='on', right='on', length = 5, width=1.0, direction='out')
            
            ax.tick_params(axis='x', which='minor', top=True)
            ax.tick_params(axis='y', which='minor', right=True)
            ax.set_xlim(0,rkarr.max())      # Deprojected radius in arcsec
            ax.tick_params('both')
            ax.set_xlabel(strylabel_polar)
            ax.set_ylabel(strflux)

            ax.grid(axis='both', which='major', ls='-', alpha=0.8)
            
            plt.savefig('./'+'axi'+fileout, dpi=160)
            plt.clf()

        os.system('rm -rf deproj_polar_dir')
        os.chdir(currentdir)
