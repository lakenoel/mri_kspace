"""
code from
https://github.com/birogeri/kspace-explorer/blob/master/kspace.py
"""

import numpy as np
import nibabel as ni

try:
    import mkl_fft as m

    fftn = m.fftn
    ifftn = m.ifftn
except (ModuleNotFoundError, ImportError):
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
finally:
    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift

def convert_2_kspace(img):
    """Converts an MR image to kspace"""
    def np_fft(img: np.ndarray):
        """ Performs FFT function (image to kspace)
        Performs FFT function, FFT shift and stores the unmodified kspace data
        in a variable and also saves one copy for display and edit purposes.
        Parameters:
            img (np.ndarray): The NumPy ndarray to be transformed
            out (np.ndarray): Array to store output (must be same shape as img)
        """
        return fftshift(fftn(ifftshift(img)))
    
    # 2. Prepare kspace display - get magnitude then scale and normalise
    # K-space scaling: https://homepages.inf.ed.ac.uk/rbf/HIPR2/pixlog.htm
    kspace = np_fft(img)
    return kspace

def convert_2_image(kspace):
#def np_ifft(kspace: np.ndarray):
    """Performs inverse FFT function (kspace to [magnitude] image)
    Performs iFFT on the input data and updates the display variables for
    the image domain (magnitude) image and the kspace as well.
    Parameters:
        kspace (np.ndarray): Complex kspace ndarray
        out (np.ndarray): Array to store values
    """
    return np.absolute(fftshift(ifftn(ifftshift(kspace))))

def reduced_scan_percentage(kspace: np.ndarray, percentage: float):
    """Deletes a percentage of lines from the kspace in phase direction
    Deletes an equal number of lines from the top and bottom of kspace
    to only keep the specified percentage of sampled lines. For example if
    the image has 256 lines and percentage is 50.0 then 64 lines will be
    deleted from the top and bottom and 128 will be kept in the middle.
    Parameters:
        kspace (np.ndarray): Complex kspace data
        percentage (float): The percentage of lines sampled (0.0 - 100.0)
    """
    kspace = kspace.copy()
    if int(percentage) < 100:
        percentage_delete = 1 - percentage / 100
        lines_to_delete = round(percentage_delete * kspace.shape[0] / 2)
        if lines_to_delete:
            kspace[0:lines_to_delete] = 0
            kspace[-lines_to_delete:] = 0
    return kspace

def get_display(img, kspace, kscale=-3):
    def normalise(f: np.ndarray):
        """ Normalises array by "streching" all values to be between 0-255.
        Parameters:
            f (np.ndarray): input array
        """
        fmin = float(np.min(f))
        fmax = float(np.max(f))
        if fmax != fmin:
            coeff = fmax - fmin
            f[:] = np.floor((f[:] - fmin) / coeff * 255.)
    
        return f

    kspace_abs = np.absolute(kspace)  # get magnitude
    if np.any(kspace_abs):
        scaling_c = np.power(10., kscale)
        kspace_abs = np.log1p(kspace_abs * scaling_c)
        kspace_abs = normalise(kspace_abs)

    # 3. Obtain uint8 type arrays for QML display
    image_display_data = np.require(img, np.uint8)
    kspace_display_data = np.require(kspace_abs, np.uint8)
    return image_display_data, kspace_display_data


def get_reduced_scan(img_path, REDUCED_PERCENTAGE):
    img = ni.load(img_path).get_fdata()
    #n_layers = img.shape[-1]
    # convert each layer in the image to k-space
    
    #kspace_layers = [convert_2_kspace(img[:,:,layer]) for layer in range(n_layers)]
    #reduced_kspace = [reduced_scan_percentage(kspace_layer, REDUCED_PERCENTAGE) for kspace_layer in kspace_layers]

    #reduced_image_layers = [convert_2_image(reduced_kspace_layer) for reduced_kspace_layer in reduced_kspace]

    reduced_kspace = np.absolute(convert_2_kspace(img))
    return reduced_kspace
    #reduced_kspace = np.absolute(np.stack(reduced_kspace, axis=-1))
    #return reduced_kspace

    reduced_image = np.absolute(np.stack(reduced_image_layers, axis=-1))
    return reduced_image
