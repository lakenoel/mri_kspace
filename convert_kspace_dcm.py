"""
code from
https://github.com/birogeri/kspace-explorer/blob/master/kspace.py
"""

import numpy as np
import pydicom as dcm

try:
    import mkl_fft as m

    fft2 = m.fft2
    ifft2 = m.ifft2

except (ModuleNotFoundError, ImportError):
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
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
        return fftshift(fft2(ifftshift(img)))
    
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
    return np.absolute(fftshift(ifft2(ifftshift(kspace))))

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


def normalize(kspace, kscale=-3):
    """ Normalises array by "streching" all values to be between 0-255.
    Parameters:
        f (np.ndarray): input array
    """
    def normalise(f: np.ndarray):
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
    
    return kspace_abs

def get_reduced_image(slices, reduced_percentage):
    kspace_slices = [convert_2_kspace(s.pixel_array) for s in slices]
    reduced_kspace = [reduced_scan_percentage(s, reduced_percentage) for s in kspace_slices]
    reduced_scans = [convert_2_image(s) for s in reduced_kspace]
    reduced_image = np.expand_dims(np.stack(reduced_scans).transpose(1,2,0), axis=0)
    return reduced_image

def get_kspace(scan):#img_path):
    #img = dcm.read_file(img_path).pixel_array

    kspace = convert_2_kspace(scan)

    return normalize(kspace)


# def plot_layer(LAYER, REDUCED_PERCENTAGE=30):
#     fig, ax = plt.subplots(2,2)

#     T1_kspace = reduced_scan_percentage(T1_kspace_layers[LAYER].copy(), REDUCED_PERCENTAGE)
#     T1_reduced_kspace = reduced_scan_percentage(T1_kspace_layers[LAYER].copy(), REDUCED_PERCENTAGE)
#     T1_reduced_image = convert_2_image(T1_reduced_kspace)
#     T2_kspace = reduced_scan_percentage(T2_kspace_layers[LAYER].copy(), REDUCED_PERCENTAGE)
#     T2_reduced_kspace = reduced_scan_percentage(T2_kspace_layers[LAYER].copy(), REDUCED_PERCENTAGE)
#     T2_reduced_image = convert_2_image(T2_reduced_kspace)

#     T1_img_arr, T1_ksp_arr = get_display(T1_image_layers[LAYER], T1_kspace_layers[LAYER])
#     ax[0,0].imshow(T1_img_arr, cmap='gray')
#     ax[0,1].imshow(T1_ksp_arr, cmap='gray')

#     # reduced_img_arr, reduced_ksp_arr = get_display(reduced_image, reduced_kspace)
#     # ax[1,0].imshow(reduced_img_arr, cmap='gray')
#     # ax[1,1].imshow(reduced_ksp_arr, cmap='gray')

#     T2_img_arr, T2_ksp_arr = get_display(T2_image_layers[LAYER], T2_kspace_layers[LAYER])
#     ax[1,0].imshow(T2_img_arr, cmap='gray')
#     ax[1,1].imshow(T2_ksp_arr, cmap='gray')

#     axis_off(ax[0,0], ax[0,1], ax[1,0], ax[1,1]);


# plot_layer(LAYER)
