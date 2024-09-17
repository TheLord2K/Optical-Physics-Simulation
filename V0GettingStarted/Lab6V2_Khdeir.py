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
# Student name: Mohammad Khdeir
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
    (im, Dphi, mask) = optical_system(im, 300)
    # Run the Gerchberg-Saxton algorithm
    images, errors = gerchberg_saxton(im, 10, Dphi, mask)
    # Save the output images and plot
    save_frames(images, errors)
    
# Load an image and normalize its values to the range [0, 1].
# Input: name - file name of the image
# Output: im - normalized grayscale image

def load_image(name):
    # Read image file and normalize pixel values
    im = plt.imread(name) / 255
    # Convert to grayscale if the image has multiple channels
    if len(im.shape) > 2:
        im = np.mean(im, axis=-1)
    # Clip values to the range [0, 1]
    im[im < 0] = 0
    im[im > 1] = 1
    return im

# Simulate the optical system with an occultation and phase aberration.
# Input: im - input image, width - width of the occultation circle
# Output: im - simulated image, Dphi - phase aberration, mask - occultation mask

def optical_system(im, width):
    # Apply an occultation circle to the center of the image
    im, mask = occult_circle(im, width)
    # Compute the 2D Fourier transform
    (IMa, IMp) = dft2(im)
    # Generate random phase aberration
    rng = np.random.default_rng(12345)
    imR = rng.random(im.shape)
    (_, Dphi) = dft2(imR)
    # Inverse Fourier transform to simulate the optical system
    im = idft2(IMa, IMp - Dphi)
    return im, Dphi, mask

# Apply an occultation circle to the center of the image.
# Input: im - input image, width - width of the occultation circle
# Output: im - image with applied occultation circle, mask - occultation mask

def occult_circle(im, width):
    # Define the center of the image
    center = np.array(im.shape) // 2
    # Create a boolean mask for the occultation circle
    y, x = np.ogrid[:im.shape[0], :im.shape[1]]
    mask = ((x - center[1])**2 + (y - center[0])**2) <= (width / 2)**2
    # Apply the mask to the image
    im[mask] = 0
    return im, mask

# Compute the inverse 2D discrete Fourier transform.
# Input: IMa - amplitude, IMp - phase in Fourier space
# Output: im - reconstructed image in spatial domain

def idft2(IMa, IMp):
    # Reconstruct the image from amplitude and phase information
    IM = IMa * (np.cos(IMp) + 1j * np.sin(IMp))
    im = np.fft.irfft2(IM)
    # Clip values to the range [0, 1]
    im[im < 0] = 0
    im[im > 1] = 1
    return im

# Simulate the Gerchberg-Saxton algorithm.
# Input: im - input image, max_iters - maximum iterations,
# Dphi - phase aberration, mask - occultation mask
# Output: images - list of images during iterations, errors - list of errors during iterations

def gerchberg_saxton(im, max_iters, Dphi, mask):
    # Compute the 2D Fourier transform of the input image
    (IMa, IMp) = dft2(im)
    # Initialize empty lists to store images and errors during iterations
    images = []
    errors = []
    # Iterate through the Gerchberg-Saxton algorithm
    for k in range(max_iters + 1):
        print(f"Iteration {k} of {max_iters}")
        # Interpolate between the original phase and corrected phase
        alpha = k / max_iters
        corrected_phase = (1 - alpha) * IMp + alpha * (IMp + Dphi)
        # Inverse Fourier transform to update the image
        im = idft2(IMa, corrected_phase)
        images.append(im)
        # Compute and store the occultation error
        error = occult_error(im, mask)
        errors.append(error)
    return images, errors

# Compute the 2D discrete Fourier transform of a grayscale image.
# Input: im - input grayscale image
# Output: IMa - amplitude, IMp - phase in Fourier space
def dft2(im):
    
    # Compute the 2D Fourier transform and extract amplitude and phase
    IM = np.fft.rfft2(im)
    IMa = np.abs(IM)
    IMp = np.angle(IM)
    return IMa, IMp

#Compute the occultation error.
# Input: im - input image, mask - occultation mask
# Output: sum of squared values in the occulted region

def occult_error(im, mask):
    # Compute the sum of squared values in the occulted region
    return np.sum((im[mask])**2)

# Save a series of images as PNG files and plot the errors.
# Input: images - list of images during iterations, errors - list of errors during iterations
# Output: PNG files of images with overlaid error plots
# Side effects: Plots output and file output

def save_frames(images, errors):
    # Get the maximum iteration count
    max_iters = len(images) - 1
    # Get the maximum error value
    max_errors = max(errors)

    # Iterate through the images and errors
    for k in range(max_iters + 1):
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust the figsize as needed

        # Plot the image with adjusted aspect ratio
        ax.imshow(np.stack([images[k]] * 3, axis=-1), cmap='gray', extent=[0, max_iters, 0, max_errors], aspect='auto')
        ax.set_title('Coronagraph Simulation')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Sum Square Error')

        # Plot the graph with a smooth line
        ax.plot(range(k + 1), errors[:k + 1], color='red', linestyle='-')

        plt.tight_layout()
        plt.savefig(f'coronagraph{k}.png', format='png', dpi=300)  # Adjust dpi as needed
        plt.show()
if __name__ == "__main__":
    main()
