"""
Veniamin Velikoretskikh
veniamin@pdx.edu
CS 410: Computer Vision & Deep Learning, Summer 2026
Programming Assignment #1 - Part 1

PART 1:
  (i)   Gaussian blur      smooths the image, removes noise/detail
  (ii)  DoG (gx, gy)        highlights vertical / horizontal edges
  (iii) Sobel (gx, gy, magnitude)  a sharper edge detector than DoG

"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




# Load the image and convert it to grayscale
def load_grayscale(path):
    img = Image.open(path).convert("L")  # "L" = 8-bit grayscale mode
    return np.asarray(img, dtype=np.float64)




#Convolution 
def convolve(image, kernel, pad=None):    
   
    kernel = np.asarray(kernel, dtype=np.float64)
    k_size = kernel.shape[0]

    
    # A 3x3 kernel needs pad=1, a 5x5 kernel needs pad=2
    if pad is None:
        pad = (k_size - 1) // 2

    # Adds a border of zeros around the image.
    padded = np.pad(image, pad_width=pad, mode="constant", constant_values=0)

    # Flip the kernel both vertically and horizontally (180 degree rotation).
    kernel_flipped = np.flipud(np.fliplr(kernel))

    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    # Slide the kernel over every pixel position in the original image size.
    for i in range(height):
        for j in range(width):
            neighborhood = padded[i:i + k_size, j:j + k_size]

            # Elementwise multiply the kernel with the neighborhood,
            # then sum everything up, this is the new pixel value.
            output[i, j] = np.sum(neighborhood * kernel_flipped)

    return output



# (i) Gaussian kernels
GAUSSIAN_3x3 = (1 / 16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
])

GAUSSIAN_5x5 = (1 / 273) * np.array([
    [1,  4,  7,  4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1,  4,  7,  4, 1],
])

# (ii) DoG kernels

DOG_GX = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

DOG_GY = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
])

# (iii) Sobel kernels

SOBEL_GX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

SOBEL_GY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
])


# A helper to plot "before / after" panels and save them 
def show_before_after(original, results, titles, suptitle, save_path):
    num_panels = len(results) + 1  # +1 for the original image
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, result_img, title in zip(axes[1:], results, titles):
        ax.imshow(result_img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)  # close instead of plt.show() so batch runs don't hang


# Run all of Part 1 on one image
def run_part1(image_path, tag):
    print(f"\n Running Part 1 on {image_path} ")
    img = load_grayscale(image_path)

    # (i) Gaussian smoothing 
    # pad=1 for the 3x3 kernel, pad=2 for the 5x5 kernel, per the handout.
    gauss_3x3 = convolve(img, GAUSSIAN_3x3, pad=1)
    gauss_5x5 = convolve(img, GAUSSIAN_5x5, pad=2)
    show_before_after(
        img, [gauss_3x3, gauss_5x5],
        ["3x3 Gaussian", "5x5 Gaussian"],
        f"Part 1(i) - Gaussian Filtering - {tag}",
        save_path=f"part1_{tag}_gaussian.png",
    )

    # (ii) DoG gx / gy 
    dog_x = convolve(img, DOG_GX, pad=1)
    dog_y = convolve(img, DOG_GY, pad=1)
    show_before_after(
        img, [dog_x, dog_y],
        ["DoG gx", "DoG gy"],
        f"Part 1(ii) - DoG Filtering - {tag}",
        save_path=f"part1_{tag}_dog.png",
    )

    # (iii) Sobel gx / gy / magnitude
    sobel_x = convolve(img, SOBEL_GX, pad=1)
    sobel_y = convolve(img, SOBEL_GY, pad=1)

    # Compute the magnitude of the Sobel gradient
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    show_before_after(
        img, [sobel_x, sobel_y, sobel_magnitude],
        ["Sobel gx", "Sobel gy", "Sobel magnitude"],
        f"Part 1(iii) - Sobel Filtering - {tag}",
        save_path=f"part1_{tag}_sobel.png",
    )



if __name__ == "__main__":
    run_part1("filter1_img.jpg", "filter1")
    run_part1("filter2_img.jpg", "filter2")