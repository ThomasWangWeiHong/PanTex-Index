import math
import numpy as np
import numpy.ma as ma
import rasterio
from scipy.spatial import KDTree
from skimage import exposure
from skimage.feature import greycomatrix, greycoprops
from tqdm import tqdm



def grayscale_raster_creation(input_MSfile, output_filename):
    """ 
    This function creates a grayscale brightness image from an input image to be used for PanTex index calculation. For every pixel 
    in the input image, the intensity values from the red, green, blue channels are first obtained, and the maximum of these values 
    are then assigned as the pixel's intensity value, which would give the grayscale brightness image as mentioned earlier, as per 
    standard practice in the remote sensing academia and industry. It is assumed that the first three channels of the input image 
    correspond to the red, green and blue channels, irrespective of order.
    
    Inputs:
    - input_MSfile: File path of the input image that needs to be converted to grayscale brightness image
    - output_filename: File path of the grayscale brightness image that is to be written to file
    
    Outputs:
    - gray: Numpy array of grayscale brightness image of corresponding multi - channel input image
    
    """
    
    with rasterio.open(input_MSfile) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])[:, :, 0 : 3]
    
    gray = np.max(img, axis = 2).astype(metadata['dtype'])[np.newaxis, :, :]
    
    metadata['count'] = 1
    with rasterio.open(output_filename, 'w', **metadata) as dst:
        dst.write(gray)
    
    return gray



def PanTex_calculation_and_creation(input_filename, output_pantex_name, GL = 128, window = 35, LB = 15, UB = 85, 
                                    incrementation = 2, write = True):
    """ 
    This function is used to calculate the PanTex index for any given raster image. 
    It uses the default displacement vectors (shiftX, shiftY) : {(1,-2), (1, -1), (2, -1), (1, 0), (2, 0), 
    (0, 1), (1, 1), (2, 1), (0, 2), (1, 2)} as recommended by 'A Robust Built-Up Area Presence Index by 
    Anisotropic Rotation-Invariant Textural Measure' by Martino Pesaresi; Andrea Gerhardinger & Francois 
    Kayitakire (2008). 
    
    Inputs:
    - input_filename: String or file path of grayscale tif file to be used.
    - output_pantex_name: String or file path of generated PanTex feature map to be written.
    - GL: Number of gray levels to be used for calculations (helps to speed up calculations).
    - window: Size of the sliding window to be applied across the original grayscale image (odd numbers only).
    - LB: Lower percentile upon which to trim off histogram of grayscale values.
    - UB: Upper percentile upon which to trim off histogram of grayscale values.
    - incrementation: Stride of sliding window to be applied across the original grayscale image.
    - write: Boolean indicating whether calculated PanTex feature map should be written to file.
    
    Output:
    - pantex: Numpy array representing the PanTex feature map for input grayscale image.
    
    """
        
    if (window % 2 == 0) :
        raise ValueError('window size must be an odd number.')
    else :    
        buffer = int((window - 1) / 2)
    
    
    with rasterio.open(input_filename) as f:
        metadata = f.profile
        img = f.read(1)
    
    
    img_rescaled = rescale_intensity(np.clip(img, np.percentile(img, LB), np.percentile(img, UB)), 
                                     out_range = (0, GL - 1)).astype(metadata['dtype'])
    
    img_rescaled_padded = np.pad(img_rescaled, ((buffer, buffer), (buffer, buffer)), 
                                 mode = 'constant').astype(metadata['dtype'])  
    
            
    pantex = np.zeros((img.shape[0], img.shape[1]), dtype = np.float32)
    
    for alpha in tqdm(range(buffer, img_rescaled_padded.shape[0] - buffer, incrementation), mininterval = 600) :            
        for beta in range(buffer, img_rescaled_padded.shape[1] - buffer, incrementation) :                                                                                                                                   
            array = img_rescaled_padded[(alpha - buffer) : (alpha + buffer + 1), 
                                        (beta - buffer) : (beta + buffer + 1)].astype(int)
            
            g_1 = greycomatrix(array, [1], [np.pi / 4, - np.pi / 4, 0, - np.pi / 2], levels = GL, normed = True)
            g_2 = greycomatrix(array, [2], [0, - np.pi / 2, math.atan(0.5), - math.atan(0.5), math.atan(2), - math.atan(2)], 
                               levels = GL, normed = True)
            contrast_1 = greycoprops(g_1, prop = 'contrast')
            contrast_2 = greycoprops(g_2, prop = 'contrast')
            pantex[alpha - buffer, beta - buffer] = min(contrast_1.min(), contrast_2.min())
    
    if incrementation > 1 :
        print('Calculation of the PanTex index is complete. Conducting nearest neighbour interpolation now.')
    
        pantex = ma.masked_array(pantex, pantex == 0)
        x, y = np.mgrid[0 : pantex.shape[0], 0 : pantex.shape[1]]
        xygood = np.array((x[ ~ pantex.mask], y[ ~ pantex.mask])).T
        xybad = np.array((x[pantex.mask], y[pantex.mask])).T
        pantex[pantex.mask] = pantex[ ~ pantex.mask][KDTree(xygood).query(xybad)[1]]
                    
        print('Nearest neighbour interpolation has been completed. PanTex Index array ready to be written to file.')
    
    if write: 
        metadata['dtype'] = 'float32'
        with rasterio.open(output_pantex_name, 'w', **metadata) as dst:
            dst.write(pantex[np.newaxis, :, :])
    
    
    return pantex
