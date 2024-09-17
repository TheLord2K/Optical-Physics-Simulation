## CORONASIMULATE  Simulate coronagraph and Gerchberg-Saxton algorithm
#
# A simulation of a coronagraph and the Gerchberg-Saxton algorithm, in the
# context of NASA's Roman Space Telescope, developed to help teach ENCMP
# 100 Computer Programming for Engineers at the University of Alberta. The
# program saves output figures to PNG files for subsequent processing.
#
# Copyright (c) 2022, University of Alberta
# Electrical and Computer Engineering
# All rights reserved.
#
# Student name:Mohammad Khdeir
# Student CCID:Khdeir
# Others:
#
# To avoid plagiarism, list the names of persons, Version 0 author(s)
# excluded, whose code, words, ideas, or data you used. To avoid
# cheating, list the names of persons, excluding the ENCMP 100 lab
# instructor and TAs, who gave you compositional assistance.
#
# After each name, including your own name, enter in parentheses an
# estimate of the person's contributions in percent. Without these
# numbers, adding to 100%, follow-up questions will be asked.
#
# For anonymous sources, enter pseudonyms in uppercase, e.g., SAURON,
# followed by percentages as above. Email a link to or a copy of the
# source to the lab instructor before the assignment is due.
#
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load the input image
    im = load_image('300_26a_big-vlt-s.jpg')
    # Simulate the optical system with an occultation and phase aberration
    (im, Dphi) = optical_system(im, 300)
    # Run the Gerchberg-Saxton algorithm
    images = gerchberg_saxton(im, 10, Dphi)
    # Save the output images
    save_frames(images)

# Load an image and normalize its values to the range [0, 1].
# Parameters: name (str): The filename of the image.
# Returns: im (numpy.ndarray): The normalized image.
def load_image(name):
    im = plt.imread(name) / 255
    if len(im.shape) > 2:
        im = np.mean(im, axis=-1)
    im[im < 0] = 0
    im[im > 1] = 1
    return im
 
# Simulate the optical system with an occultation and phase aberration.
# Parameters:
# - im (numpy.ndarray): Input image.
# - width (int): Width of the occultation square.
# Returns:
# - im (numpy.ndarray): Output image after the optical system.
# - Dphi (numpy.ndarray): True phase aberration.
def optical_system(im, width):
    im = occult_square(im, width)
    (IMa, IMp) = dft2(im)
    rng = np.random.default_rng(12345)
    imR = rng.random(im.shape)
    (_, Dphi) = dft2(imR)
    im = idft2(IMa, IMp - Dphi)
    return im, Dphi

# Apply an occultation square to the center of the image.
# Parameters:
# - im (numpy.ndarray): Input image.
# - width (int): Width of the occultation square.
# Returns:
# - im (numpy.ndarray): Image with occultation applied.
def occult_square(im, width):
    center = np.array(im.shape) // 2
    half_width = width // 2
    im[center[0] - half_width: center[0] + half_width,
       center[1] - half_width: center[1] + half_width] = 0
    return im

# Compute the 2D discrete Fourier transform of a grayscale image.
# Parameters:
# - im (numpy.ndarray): Grayscale input image.
# Returns:
# - IMa (numpy.ndarray): Amplitude of the Fourier transform.
# - IMp (numpy.ndarray): Phase of the Fourier transform.
def dft2(im):
    IM = np.fft.rfft2(im)
    IMa = np.abs(IM)
    IMp = np.angle(IM)
    return IMa, IMp

# Compute the inverse 2D discrete Fourier transform.
# Parameters:
# - IMa (numpy.ndarray): Amplitude of the Fourier transform.
# - IMp (numpy.ndarray): Phase of the Fourier transform.
# Returns:
# - im (numpy.ndarray): Reconstructed grayscale image.
def idft2(IMa, IMp):
    IM = IMa * (np.cos(IMp) + 1j * np.sin(IMp))
    im = np.fft.irfft2(IM)
    im[im < 0] = 0
    im[im > 1] = 1
    return im

# Simulate the Gerchberg-Saxton algorithm.
# Parameters:
# - im (numpy.ndarray): Input image.
# - max_iters (int): Maximum number of iterations.
# - Dphi (numpy.ndarray): True phase aberration.
# Returns:
# - images (list): List of output images for each iteration.
def gerchberg_saxton(im, max_iters, Dphi):
    (IMa, IMp) = dft2(im)
    images = []
    for k in range(max_iters + 1):
        print(f"Iteration {k} of {max_iters}")
        alpha = k / max_iters
        corrected_phase = (1 - alpha) * IMp + alpha * (IMp + Dphi)
        im = idft2(IMa, corrected_phase)
        images.append(im)
    return images

# Save a series of images as PNG files.
# Parameters:
# - images (list): List of images to be saved.
def save_frames(images):
    max_iters = len(images) - 1
    for k in range(max_iters + 1):
        grey_image = np.stack([images[k]] * 3, axis=-1)  # Convert to 3-channel grayscale
        plt.imshow(grey_image, cmap='gray')  # Display in grayscale
        plt.title(f'Iteration {k} of {max_iters}')
        plt.xticks([])  # Hide x-axis ticks and labels
        plt.yticks([])  # Hide y-axis ticks and labels
        plt.savefig(f'coronagraph{k}.png', format='png', bbox_inches='tight', pad_inches=0)  # Save in grayscale
        plt.show()

main()


