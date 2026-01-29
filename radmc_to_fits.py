# import global variables
import par

import numpy as np
import math
import os
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from pylab import *

from beam import *

# -------------------------
# Convert result of RADMC3D calculation into fits file
# -------------------------
def exportfits():

    # ----------
    # read input file produced by radmc3D if it hasn't been deleted!
    # ----------
    if os.path.isfile('image.out') == True:

        infile = 'image.out'
        print('--------- reading image.out ----------')
        f = open(infile,'r')

        # read header info:
        iformat = int(f.readline())

        # nb of pixels
        im_nx, im_ny = tuple(np.array(f.readline().split(),dtype=int))

        # nb of wavelengths, can be different from unity for
        # multi-color images of gas emission
        nlam = int(f.readline())

        # pixel size in each direction in cm
        pixsize_x, pixsize_y = tuple(np.array(f.readline().split(),dtype=float))

        # read wavelength in microns
        lbda = np.zeros(nlam)
        for i in range(nlam):
            lbda[i] = float(f.readline())
        # convert lbda in velocity (km/s)
        if nlam > 1:
            lbda0 = lbda[nlam//2]  # implicitly gas RT
            vel_range = par.c * 1e-5 *(lbda-lbda0)/lbda0 # in km/s (c is in cm/s!)
            if par.verbose == 'Yes':
                print('range of wavelengths = ', lbda)
                print('range of velocities = ', vel_range)
            dv = vel_range[1]-vel_range[0]
        else:
            if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both':
                lbda0 = lbda[0]
            else:
                lbda0 = par.wavelength*1e3 # in microns

        print('lbda [microns] = ', lbda0)
                
        # next line in image.out is empty
        f.readline()               

        # calculate physical scales
        distance = par.distance * par.pc        # distance is in cm here
        pixsize_x_deg = 180.0*pixsize_x / distance / np.pi
        pixsize_y_deg = 180.0*pixsize_y / distance / np.pi

        # surface of a pixel in radian squared
        pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (np.pi/180.)**2
        
        # 1 Jansky converted in cgs x pixel surface in cm^2 (same
        # conversion whether RT in dust or gas lines)
        fluxfactor = 1.e23 * pixsurf_ster
        if par.verbose == 'Yes':
            print('fluxfactor = ',fluxfactor)
            
        # distance is in cm here
        distance = par.distance * par.pc
        pixsize_x  = pixsize_x_deg * distance * np.pi / 180.0
        
        # beam area in pixel^2
        mycdelt = pixsize_x/par.distance/par.au
        beam =  (np.pi/(4.*np.log(2.)))*par.bmaj*par.bmin/(mycdelt**2.)
        if par.verbose == 'Yes':
            print('beam = ',beam)
            
        # stdev lengths in pixel
        stdev_x = (par.bmaj/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt
        stdev_y = (par.bmin/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt
            
        # gas or dust continuum -> 1D array:
        if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both' or (par.RTdust_or_gas == 'dust' and par.polarized_scat == 'No'):
            images = np.loadtxt(infile, skiprows=5+nlam)

        # dust polarized emission -> 1D array:
        if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
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

            images = im.reshape(5*im_ny*im_nx)
            
        # keep track of image.out by writting its content in image.fits 
        if os.path.isfile('image.fits') == False:
                            
            print('--------- creating image.fits ----------')
            hdu = fits.PrimaryHDU()
            hdu.header['BITPIX'] = -32
            hdu.header['NAXIS'] = 2 
            hdu.header['NAXIS1'] = im_nx
            hdu.header['NAXIS2'] = im_ny
            hdu.header['EPOCH']  = 2000.0
            hdu.header['EQUINOX'] = 2000.0
            hdu.header['LONPOLE'] = 180.0
            hdu.header['CTYPE1'] = 'RA---SIN'
            hdu.header['CTYPE2'] = 'DEC--SIN'
            hdu.header['CRVAL1'] = float(0.0)
            hdu.header['CRVAL2'] = float(0.0)
            hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
            hdu.header['CDELT2'] = float(pixsize_y_deg)
            hdu.header['LBDAMIC'] = float(lbda0)
            hdu.header['CUNIT1'] = 'deg     '
            hdu.header['CUNIT2'] = 'deg     '
            hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
            hdu.header['CRPIX2'] = float((im_ny+1.)/2.)
            hdu.header['BUNIT'] = 'cgs'
            hdu.header['BTYPE'] = 'Intensity'
            hdu.header['BSCALE'] = float(1.0)
            hdu.header['BZERO'] = float(0.0)
            del hdu.header['EXTEND']
            hdu.data = images.astype('float32')
            hdu.writeto('image.fits', output_verify='fix', overwrite=False)

            # close image file image.out
            f.close()

            # now that image.fits is created, no need to keep image.out!
            os.system('rm -f image.out')
            
        else:
            # image.fits is already present, no need to keep image.out!
            os.system('rm -f image.out')

            
    # ----------
    # if image.out is no longer here (it has been deleted), we read image.fits file instead
    # ----------
    else:
        print('--------- reading image.fits ----------')
        f2 = fits.open('image.fits')
        images = f2[0].data
        hdr = f2[0].header
        f2.close()

        mydim = hdr['NAXIS']

        # Get dimensions from 1D array stored in image.fits:
        if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both':
            nlam = par.linenlam
            im_nx = int(np.sqrt(hdr['NAXIS1']/nlam))
            vel_range = -par.widthkms + 2.0*par.widthkms*np.arange(nlam)/(nlam-1.0)
        else:
            if par.polarized_scat == 'Yes':
                im_nx = int(np.sqrt(hdr['NAXIS1']/5.0))
            else:
                im_nx = int(np.sqrt(hdr['NAXIS1']))
        im_ny = im_nx

        lbda0 = hdr['LBDAMIC']

        # ----------
        # auxiliary quantities needed below for operations on gas images
        # ----------
        pixsize_x_deg = np.abs(hdr['CDELT1'])
        pixsize_y_deg = np.abs(hdr['CDELT2'])

        # we need to define dv again (velocity resolution in km/s)
        dv = par.widthkms / par.linenlam 
        
        # surface of a pixel in radian squared
        pixsurf_ster = pixsize_x_deg*pixsize_y_deg * (np.pi/180.)**2

        # 1 Jansky converted in cgs x pixel surface in cm^2 (same
        # conversion whether RT in dust or gas lines)
        fluxfactor = 1.e23 * pixsurf_ster
        if par.verbose == 'Yes':
            print('fluxfactor = ',fluxfactor)
            
        # distance is in cm here
        distance = par.distance * par.pc
        pixsize_x  = pixsize_x_deg * distance * np.pi / 180.0
        # beam area in pixel^2
        mycdelt = pixsize_x/par.distance/par.au
        beam =  (np.pi/(4.*np.log(2.)))*par.bmaj*par.bmin/(mycdelt**2.)
        if par.verbose == 'Yes':
            print('beam = ',beam)        
        # stdev lengths in pixel
        stdev_x = (par.bmaj/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt
        stdev_y = (par.bmin/(2.*np.sqrt(2.*np.log(2.)))) / mycdelt

            
    # ----------
    # Operations on images
    # ----------

    # - - - - - -
    # dust continuum RT calculations
    # - - - - - -
    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'No':
        im = images.reshape(im_ny,im_nx)
        # sometimes the intensity has a value at the origin that is
        # unrealistically large. We put it to zero at the origin, as
        # it should be in our disc model! (3pix x 3pix around)
        if par.central_binary != 'No':
            im[im_ny//2-1:im_ny//2+1,im_nx//2-1:im_nx//2+1] = 0.0
        if (('AddStarInRawImage' in open('params.dat').read()) and (par.AddStarInRawImage == 'Yes')):
            if par.central_binary == 'No':        
                # first find out star's radius in arcseconds:
                star_radius_in_arcseconds = par.rstar*7.0e8/1.5e11/par.distance
                # then find pixel size in arcseconds:
                pixel_size_in_arcseconds  = pixsize_x_deg*3600.0            
                # infer how many pixels are spawn by central star
                nb_pixels_spawn_by_star = round(star_radius_in_arcseconds/pixel_size_in_arcseconds)
                # ratio of integrated fluxes between star and disc
                star_to_disc_flux_ratio = par.star_to_disc_flux_ratio  # 0.1
                # finally get specific intensity of star emission in each pixel (assumed uniform)
                # np.sum(im) = disc integrated flux 
                # star_pixel x nb_pixels_spawn_by_star x nb_pixels_spawn_by_star = star integrated flux = star_to_disc_averaged_flux_ratio x disc integrated flux 
                star_pixel = star_to_disc_flux_ratio*np.sum(im)/(nb_pixels_spawn_by_star*nb_pixels_spawn_by_star)  # star ~ square!...
                # change im array accordingly
                for i in range(-nb_pixels_spawn_by_star,nb_pixels_spawn_by_star+1):
                    for j in range(-nb_pixels_spawn_by_star,nb_pixels_spawn_by_star+1):
                        pixdist = np.sqrt(i*i + j*j)
                        if pixdist <= nb_pixels_spawn_by_star:     # star ~ circle!...
                            im[im_ny//2-i,im_nx//2-j] = star_pixel   

            # if there's a central binary star, I will increase their intensity such that the total disc-to-stars flux ratio
            # equals desired value
            else:
                # first find out grid's inner edge radius in arcseconds:
                inner_edge_in_arcseconds = par.gas.rmed[0]*par.gas.culength/1.5e11/par.distance # 3.3 mas
                inner_edge_in_arcseconds *= 0.95*np.cos(par.inclination*np.pi/180.0) # 0.75 = safety factor! to avoid hitting the disc...
                # then find pixel size in arcseconds:
                pixel_size_in_arcseconds  = pixsize_x_deg*3600.0            
                # infer how many pixels are spawn by central star
                nb_pixels_spawn_by_inneredge = round(inner_edge_in_arcseconds/pixel_size_in_arcseconds)
                #print('inner_edge_in_arcseconds = ', inner_edge_in_arcseconds)
                #print('nb_pixels_spawn_by_inneredge = ', nb_pixels_spawn_by_inneredge)
                # change im array accordingly
                current_stars_intensity = 0.0
                for i in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                    for j in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                        pixdist = np.sqrt(i*i + j*j)
                        if pixdist <= nb_pixels_spawn_by_inneredge:
                            current_stars_intensity += im[im_ny//2-i,im_nx//2-j]
                total_disc_flux = np.sum(im)
                # print('stars current intensity = ', current_stars_intensity)
                # print('ratio wrt disc = ', current_stars_intensity/total_disc_flux)
                new_stars_intensity = 0.0
                for i in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                    for j in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                        pixdist = np.sqrt(i*i + j*j)
                        if pixdist <= nb_pixels_spawn_by_inneredge:
                            im[im_ny//2-i,im_nx//2-j] *= (par.star_to_disc_flux_ratio*total_disc_flux/current_stars_intensity)
                            new_stars_intensity += im[im_ny//2-i,im_nx//2-j]
                # print('stars new intensity = ', new_stars_intensity)
                # print('new ratio wrt disc = ', new_stars_intensity/total_disc_flux)

        if (('TurnOffStars' in open('params.dat').read()) and (par.TurnOffStars == 'Yes')):
            if par.central_binary == 'Yes':
                # first find out grid's inner edge radius in arcseconds:
                inner_edge_in_arcseconds = par.gas.rmed[0]*par.gas.culength/1.5e11/par.distance # 3.3 mas
                inner_edge_in_arcseconds *= 0.95*np.cos(par.inclination*np.pi/180.0) # 0.75 = safety factor! to avoid hitting the disc...
                # then find pixel size in arcseconds:
                pixel_size_in_arcseconds  = pixsize_x_deg*3600.0            
                # infer how many pixels are spawn by central star
                nb_pixels_spawn_by_inneredge = round(inner_edge_in_arcseconds/pixel_size_in_arcseconds)
                #print('inner_edge_in_arcseconds = ', inner_edge_in_arcseconds)
                #print('nb_pixels_spawn_by_inneredge = ', nb_pixels_spawn_by_inneredge)
                # change im array accordingly
                current_stars_intensity = 0.0
                for i in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                    for j in range(-nb_pixels_spawn_by_inneredge,nb_pixels_spawn_by_inneredge+1):
                        pixdist = np.sqrt(i*i + j*j)
                        if pixdist <= nb_pixels_spawn_by_inneredge:
                            im[im_ny//2-i,im_nx//2-j] = 0.0

        if (('TurnOffDisc' in open('params.dat').read()) and (par.TurnOffDisc == 'Yes')):
            if par.central_binary == 'Yes':
                # first find out grid's inner edge radius in arcseconds:
                inner_edge_in_arcseconds = par.gas.rmed[0]*par.gas.culength/1.5e11/par.distance # 3.3 mas
                inner_edge_in_arcseconds *= 0.95*np.cos(par.inclination*np.pi/180.0) # 0.75 = safety factor! to avoid hitting the disc...
                # then find pixel size in arcseconds:
                pixel_size_in_arcseconds  = pixsize_x_deg*3600.0            
                # infer how many pixels are spawn by central star
                nb_pixels_spawn_by_inneredge = round(inner_edge_in_arcseconds/pixel_size_in_arcseconds)
                # change im array accordingly
                for i in range(im_nx):
                    for j in range(im_ny):
                        pixdist = np.sqrt( (i-im_nx//2)*(i-im_nx//2) + (j-im_ny//2)*(j-im_ny//2) )
                        if pixdist > nb_pixels_spawn_by_inneredge:
                            im[i,j] = 0.0


    # - - - - - -
    # dust polarized RT calculations
    # - - - - - -
    if par.RTdust_or_gas == 'dust' and par.polarized_scat == 'Yes':
        im = images.reshape(5,im_ny,im_nx)

        for k in range(5):
            # sometimes the intensity has a value at the origin that
            # is unrealistically large. We put it to zero *around* the
            # origin, as it should be in our disc model!
            im[k,im_ny//2-1:im_ny//2+1,im_nx//2-1:im_nx//2+1] = 0.0
            if (('TestAddStar' in open('params.dat').read()) and (par.TestAddStar == 'Yes')):  # for test purpose: add the star light
                max_pixel = 5 * im.max()
                im[k,im_ny//2-1:im_ny//2,im_nx//2-1:im_nx//2] = max_pixel

    # - - - - - -
    # line or dust+line RT calculations
    # - - - - - -
    if par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both':
        
        intensity_in_each_channel = np.zeros((nlam,1,im_ny,im_nx))
        moment0 = np.zeros((im_ny,im_nx))
        moment1 = np.zeros((im_ny,im_nx))
        im0     = np.zeros((im_ny,im_nx))
            
        for i in range(nlam):
            print('channel '+str(i)+' out of '+str(nlam),end='\r')
            im_v = images[i*im_ny*im_nx:(i+1)*im_ny*im_nx]
            im = im_v.reshape(im_ny,im_nx)
            # sometimes the intensity has a value at the origin that
            # is unrealistically large, in particular when the
            # velocity relative to the systemic one is large. We put
            # the specific intensity to zero around the origin, as it
            # should be in our disc model!
            im[im_ny//2-1:im_ny//2+1,im_nx//2-1:im_nx//2+1] = 0.0
                
            # ---
            if (par.add_noise == 'Yes'):
                # noise standard deviation in cgs
                noise_dev_std_cgs = (par.noise_dev_std/fluxfactor) / np.sqrt(0.5*beam)  # 1D
                if i==0 and par.verbose == 'Yes':
                    print('rms noise level in cgs = ',noise_dev_std_cgs)
                # noise array
                noise_array = np.random.normal(0.0,noise_dev_std_cgs,size=im_ny*im_nx)
                noise_array = noise_array.reshape(im_ny,im_nx)
                im += noise_array
            # ---

            # dust continuum emission is removed by subtracting the
            # very first channel map (you need to make sure that the
            # very first channel map does not contain significant
            # information on gas emission...):
            if (par.RTdust_or_gas == 'both' and par.subtract_continuum == 'Yes'):
                if i==0:
                    im0 = im
                im = im - im0
                
            # keep track of beam-convolved specific intensity in each channel map
            if par.plot_tau == 'No':
                smooth = Gauss_filter(im, stdev_x, stdev_y, par.bpaangle, Plot=False) + 1e-20  # avoid division by zeros w/o noise...
                if par.intensity_inJyperpixel_inrawdatacube == 'No':
                    intensity_in_each_channel[i,0,:,:] = smooth*fluxfactor*beam  # in Jy/beam
                else:
                    intensity_in_each_channel[i,0,:,:] = im*fluxfactor      # in Jy/pixel
            else:
                intensity_in_each_channel[i,0,:,:] = im  # in this case im contains the optical depth (non convolved)
            # ---
            if par.moment_order == 0:
                moment0 += im*dv  # non convolved (done afterwards)
            if par.moment_order == 1:
                moment0 += smooth*dv
                moment1 += smooth*vel_range[i]*dv
            if par.moment_order > 1:
                sys.exit('moment map of order > 1 not implemented yet, I must exit!')
                
        # end loop over wavelengths
        if par.moment_order == 0:
            im = moment0   # non-convolved quantity
        if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:

            # -------------------------------------
            # calculate the 'classic' moment-1 map:
            # -------------------------------------
            im = moment1/moment0 # ratio of two beam-convolved quantities (convolution not redone afterwards)

            # -------------------------------------
            # calculate the residual moment-1 map:
            # work done by Valentin Oncle
            # CB: alternative could have been to deproject im by using the polar deprojection tools?
            # -------------------------------------

            # pixel size converted from degrees to arcseconds
            cdelt = float(-1.*pixsize_x_deg)*3600.0
            nx = im_nx
            ny = im_ny

            if ( (nx % 2) == 0):
                dpix = 0.5
            else:
                dpix = 0.0
            a0 = cdelt*(nx//2.-dpix)   # >0
            a1 = -cdelt*(nx//2.+dpix)  # <0
            d0 = -cdelt*(nx//2.-dpix)  # <0
            d1 = cdelt*(nx//2.+dpix)   # >0
            # da positive definite
            if ( (par.minmaxaxis != '#') and (par.minmaxaxis < abs(a0)) ):
                da = par.minmaxaxis
            else:
                da = np.maximum(abs(a0),abs(a1))

            # create a new image
            (Y,X) = (im_ny,im_nx)
            moment1_minus_vK = np.zeros((Y,X))

            # go through moment1_minus_vK[][] to subtract the keplerian velocity at each pixel from im[][]
            for y in range(Y):
                for x in range(X):

                    # calculate the distance r in [cm] of (x,y) from the star in the disc frame
                    alpha = (par.posangle+90.0)*math.pi/180

                    # (xd,yd) coordinates in the disc plane (rotation of alpha + rotation of par.inclination around the xd axis)
                    # ((x,y) being the coordinates in the sky plane)
                    xd = math.cos(alpha)*float(x) + math.sin(alpha)*float(y)
                    yd = (-math.sin(alpha)*float(x) + math.cos(alpha)*float(y))/math.cos(par.inclination*math.pi/180.0)

                    Xd = math.cos(alpha)*X + math.sin(alpha)*Y
                    Yd = (-math.sin(alpha)*X + math.cos(alpha)*Y)/math.cos(par.inclination*math.pi/180.0)

                    d = ((xd - Xd//2)**2 + ((yd - Yd//2))**2)**0.5
                    r = np.maximum(abs(a0),abs(a1)) * par.distance * par.au * d / (X*0.5)       # cm

                    # finally calculate the distance r_cu in the code units
                    r_cu = r/(par.gas.culength*100.0)
                    
                    # azimuthal coordinate in the disk plan
                    phi = math.atan2(((yd-Yd//2)),(xd-Xd//2+1e-20))

                    # for testing purpose: defining the Keplerian velocity above or below the midplane
                    # if (('TestVelocityNotMidplane' in open('params.dat').read()) and (par.TestVelocityNotMidplane == 'Yes')):
                    testtt = 'No'
                    if testtt == 'Yes':
                        hh = par.aspectratio * (r_cu)**(par.flaringindex) * r
                        eta = 3.0
                        fact = np.sqrt(r/np.sqrt(r**2 + (eta*hh)**2) - 3*((eta*hh)/r)**2)
                    else:
                        fact = 1.0

                    # substract the keplerian velocity at each [y,x]
                    if r_cu > par.gas.redge.min() and r_cu < par.gas.redge.max():
                        vK_obs = 1e-5 * math.sin(par.inclination*math.pi/180.0) * math.cos(phi) * (1e3*par.gas.cumass*par.G/r)**0.5 * fact       # compute v_K_app(x,y)
                        moment1_minus_vK[y,x] = im[y,x] - vK_obs


    # ----------
    # Finally write (modified) images in fits file
    # ----------
    # Fits header
    hdu = fits.PrimaryHDU()
    hdu.header['BITPIX'] = -32
    if par.polarized_scat == 'No':
        hdu.header['NAXIS'] = 2
    else:
        hdu.header['NAXIS'] = 3
    hdu.header['NAXIS1'] = im_nx
    hdu.header['NAXIS2'] = im_ny
    if par.polarized_scat == 'Yes':
        hdu.header['NAXIS3'] = 5
    hdu.header['EPOCH']  = 2000.0
    hdu.header['EQUINOX'] = 2000.0
    hdu.header['LONPOLE'] = 180.0
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    if par.polarized_scat == 'Yes':
        hdu.header['CTYPE3'] = 'STOKES: I, Q, U, V, P'
    hdu.header['CRVAL1'] = float(0.0)
    hdu.header['CRVAL2'] = float(0.0)
    hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
    hdu.header['CDELT2'] = float(pixsize_y_deg)
    hdu.header['LBDAMIC'] = float(lbda0)
    hdu.header['CUNIT1'] = 'deg     '
    hdu.header['CUNIT2'] = 'deg     '
    hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
    hdu.header['CRPIX2'] = float((im_ny+1.)/2.)
    if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:
        hdu.header['BUNIT'] = 'LOS velocity'
        hdu.header['BTYPE'] = 'km/s'
    else:
        if par.plot_tau == 'No':
            hdu.header['BUNIT'] = 'Jy/pixel'
            hdu.header['BTYPE'] = 'Intensity'
        else:
            hdu.header['BUNIT'] = '        '
            hdu.header['BTYPE'] = 'optical depth'
    hdu.header['BSCALE'] = float(1.0)
    hdu.header['BZERO'] = float(0.0)
    del hdu.header['EXTEND']

    # conversion of the intensity from erg/s/cm^2/Hz/steradian to Jy/pix
    if par.plot_tau == 'No' and par.moment_order != 1:
        im = im*fluxfactor

    hdu.data = im.astype('float32')
    hdu.writeto(par.outputfitsfile, output_verify='fix', overwrite=True)


    # write fits file for the residual moment-1 map:
    if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.moment_order == 1:

        hdu_2 = hdu
        hdu_2.data = moment1_minus_vK.astype('float32')

        # filename
        inbasename = os.path.basename('./'+par.outputfitsfile)
        # add beam information
        if par.log_colorscale == 'Yes':
            jybeamfileoutminusvK = re.sub('.fits', '_logYes' + '_bmaj'+str(par.bmaj) + '_bmin'+str(par.bmin) + '.fits', inbasename)
        else:
            jybeamfileoutminusvK = re.sub('.fits', '_bmaj'+str(par.bmaj) + '_bmin'+str(par.bmin) + '.fits', inbasename)

        jybeamfileoutminusvK=re.sub('.fits', '_JyBeam_residual.fits', jybeamfileoutminusvK)
        hdu_2.writeto(jybeamfileoutminusvK, output_verify='fix', overwrite=True)


    if par.plot_tau == 'No' and par.verbose == 'Yes':
        print("Total flux [Jy] = "+str(np.sum(hdu.data)))


    # if dust continuum emission has been subtracted, keep track of it
    # in a separate fits file
    if (par.RTdust_or_gas == 'both' and par.subtract_continuum == 'Yes'):
        # Fits header
        hdu = fits.PrimaryHDU()
        hdu.header['BITPIX'] = -32
        hdu.header['NAXIS'] = 2
        hdu.header['NAXIS1'] = im_nx
        hdu.header['NAXIS2'] = im_ny
        hdu.header['EPOCH']  = 2000.0
        hdu.header['EQUINOX'] = 2000.0
        hdu.header['LONPOLE'] = 180.0
        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['CRVAL1'] = float(0.0)
        hdu.header['CRVAL2'] = float(0.0)
        hdu.header['CDELT1'] = float(-1.*pixsize_x_deg)
        hdu.header['CDELT2'] = float(pixsize_y_deg)
        hdu.header['LBDAMIC'] = float(lbda0)
        hdu.header['CUNIT1'] = 'deg     '
        hdu.header['CUNIT2'] = 'deg     '
        hdu.header['CRPIX1'] = float((im_nx+1.)/2.)
        hdu.header['CRPIX2'] = float((im_ny+1.)/2.)
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['BTYPE'] = 'Intensity'
        hdu.header['BSCALE'] = float(1.0)
        hdu.header['BZERO'] = float(0.0)
        del hdu.header['EXTEND']
        # conversion of the intensity from erg/s/cm^2/Hz/steradian to Jy/pix
        if par.plot_tau == 'No' and par.moment_order != 1:
            im0 = im0*fluxfactor
        #
        hdu.data = im0.astype('float32')
        hdu.writeto(par.outputfitsfile_dust, output_verify='fix', overwrite=True)

        
    # ----------
    # finally, save entire intensity channels in another fits file (gas line
    # emission only)
    # ----------
    # case 1: application to CASA -> intensity in Jy/pixel, axis 4 = spectral
    if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.intensity_inJyperpixel_inrawdatacube == 'Yes':
        hdu3 = fits.PrimaryHDU()
        hdu3.header['BITPIX'] = -32
        hdu3.header['NAXIS'] = 4
        hdu3.header['NAXIS1'] = im_nx
        hdu3.header['NAXIS2'] = im_ny
        hdu3.header['NAXIS3'] = 1
        hdu3.header['NAXIS4'] = nlam
        hdu3.header['EPOCH']  = 2000.0
        hdu3.header['EQUINOX'] = 2000.0
        hdu3.header['LONPOLE'] = 180.0
        hdu3.header['CTYPE1'] = 'RA---SIN'
        hdu3.header['CTYPE2'] = 'DEC--SIN'
        hdu3.header['CTYPE3'] = 'STOKES'   
        hdu3.header['CTYPE4'] = 'FREQ-LSR'             
        hdu3.header['CRVAL1'] = float(0.0)
        hdu3.header['CRVAL2'] = float(0.0)
        hdu3.header['CRVAL3'] = float(1.0)  
        hdu3.header['CRVAL4'] = float(299792548/lbda0/1e-6)  # frequency in Hz     
        hdu3.header['CDELT1'] = float(-1.*pixsize_x_deg)
        hdu3.header['CDELT2'] = float(pixsize_y_deg)
        hdu3.header['CDELT3'] = float(1.0) 
        hdu3.header['CDELT4'] = float(1e9*dv/lbda0)               
        hdu3.header['LBDAMIC'] = float(lbda0)
        hdu3.header['CUNIT1'] = 'deg     '
        hdu3.header['CUNIT2'] = 'deg     '
        hdu3.header['CUNIT3'] = '        '
        hdu3.header['CUNIT4'] = 'Hz      '       
        hdu3.header['CRPIX1'] = float((im_nx+1.)/2.)
        hdu3.header['CRPIX2'] = float((im_ny+1.)/2.)
        hdu3.header['CRPIX3'] = float(1.0)
        hdu3.header['CRPIX4'] = float((nlam+1.)/2.)
        hdu3.header['BUNIT'] = 'JY/PIXEL'
        hdu3.header['BTYPE'] = 'Intensity'
        hdu3.header['BSCALE'] = float(1.0)
        hdu3.header['BZERO'] = float(0.0)
        del hdu3.header['EXTEND']
        hdu3.data = intensity_in_each_channel.astype('float32')
        hdu3.writeto(par.outputfitsfile_wholedatacube, output_verify='fix', overwrite=True)
        
    # case 2: application to bettermoments -> intensity in Jy/beam, axis 3 = spectral
    if (par.RTdust_or_gas == 'gas' or par.RTdust_or_gas == 'both') and par.intensity_inJyperpixel_inrawdatacube == 'No':
        hdu3 = fits.PrimaryHDU()
        hdu3.header['BITPIX'] = -32
        hdu3.header['NAXIS'] = 3
        hdu3.header['NAXIS1'] = im_nx
        hdu3.header['NAXIS2'] = im_ny
        hdu3.header['NAXIS3'] = nlam
        hdu3.header['EPOCH']  = 2000.0
        hdu3.header['EQUINOX'] = 2000.0
        hdu3.header['LONPOLE'] = 180.0
        hdu3.header['CTYPE1'] = 'RA---SIN'
        hdu3.header['CTYPE2'] = 'DEC--SIN'
        hdu3.header['CTYPE3'] = 'FREQ-LSR'
        hdu3.header['CRVAL1'] = float(0.0)
        hdu3.header['CRVAL2'] = float(0.0)
        hdu3.header['CRVAL3'] = float(299792548/lbda0/1e-6)  # frequency in Hz
        hdu3.header['CDELT1'] = float(-1.*pixsize_x_deg)
        hdu3.header['CDELT2'] = float(pixsize_y_deg)
        hdu3.header['CDELT3'] = float(1e9*dv/lbda0)
        hdu3.header['LBDAMIC'] = float(lbda0)
        hdu3.header['CUNIT1'] = 'deg     '
        hdu3.header['CUNIT2'] = 'deg     '
        hdu3.header['CUNIT3'] = 'Hz      '
        hdu3.header['CRPIX1'] = float((im_nx+1.)/2.)
        hdu3.header['CRPIX2'] = float((im_ny+1.)/2.)
        hdu3.header['CRPIX3'] = float((nlam+1.)/2.)
        hdu3.header['BUNIT'] = 'JY/BEAM'
        hdu3.header['BTYPE'] = 'Intensity'
        hdu3.header['BSCALE'] = float(1.0)
        hdu3.header['BZERO'] = float(0.0)
        del hdu3.header['EXTEND']
        hdu3.data = intensity_in_each_channel[:,0,:,:].astype('float32')
        hdu3.writeto(par.outputfitsfile_wholedatacube, output_verify='fix', overwrite=True)
    # ----------
