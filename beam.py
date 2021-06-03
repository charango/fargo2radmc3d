import numpy as np

from astropy.convolution import convolve, convolve_fft

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
    
    result = convolve_fft(data, Z, boundary='fill', fill_value=0.0)
    return result
