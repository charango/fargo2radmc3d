# import global variables
import par

import numpy as np
import os
import re
from copy import deepcopy
from astropy.io import fits
from astropy.wcs import WCS
import scipy as sp
from scipy.ndimage import map_coordinates
from scipy import ndimage


# -----------------------------------------------------
# Main routine to deproject cartesian flux maps onto polar maps (SC, SP)
# -----------------------------------------------------
def exec_polar_expansions(filename_source,workdir,PA,cosi,RA=False,DEC=False,alpha_min=False,Delta_min=False,XCheckInv=False,DoRadialProfile=True,ProfileExtractRadius=-1,DoAzimuthalProfile=False,PlotRadialProfile=True,zoomfactor=1.):

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
