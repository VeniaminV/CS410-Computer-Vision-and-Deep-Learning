"""
Veniamin Velikoretskikh    veniamin@pdx.edu
CS 4/510: Computer Vision & Deep Learning, Summer 2026
Programming Assignment #2   Lucas-Kanade Optical Flow
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_grayscale(path):
    # convert to grayscale just in case the image is RGB
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def convolve(image, kernel, pad=None, flip=True):

    kernel = np.asarray(kernel, dtype=np.float64)
    k_size = kernel.shape[0]

    if pad is None:
        pad = (k_size - 1) // 2

    padded = np.pad(image, pad_width=pad, mode="constant", constant_values=0)

    # flip=False here because Gx/Gy give the wrong sign if you actually flip them
    if flip:
        kernel_used = np.flipud(np.fliplr(kernel))
    else:
        kernel_used = kernel

    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            patch = padded[i:i + k_size, j:j + k_size]
            output[i, j] = np.sum(patch * kernel_used)

    return output


# gradient kernels given in the assignment
GX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

GY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
])



# 3x3 kernel of all ones  convolving with this just adds up the 9
# values in each window
BOX_3x3 = np.ones((3, 3))


def lucas_kanade_optical_flow(frame1, frame2):
   
    # flip=False for the gradients or the sign comes out backwards.
    
    Ix = convolve(frame1, GX, pad=1, flip=False)
    Iy = convolve(frame1, GY, pad=1, flip=False)

    It = frame2 - frame1  # difference between the two frames


    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    Ixt = Ix * It
    Iyt = Iy * It


    # summing each of these over a 3x3 window is the same as convolving
    # with the box kernel above 
    Sxx = convolve(Ix2, BOX_3x3, pad=1)
    Syy = convolve(Iy2, BOX_3x3, pad=1)
    Sxy = convolve(Ixy, BOX_3x3, pad=1)
    Sxt = convolve(Ixt, BOX_3x3, pad=1)
    Syt = convolve(Iyt, BOX_3x3, pad=1)

    # solve the 2x2 system for Vx, Vy using the matrix inverse formula
    det = Sxx * Syy - Sxy * Sxy

    # skip flat regions to avoid divide-by-zero
    valid = np.abs(det) > 1e-6

    Vx = np.zeros_like(frame1)
    Vy = np.zeros_like(frame1)

    Vx[valid] = -(Syy[valid] * Sxt[valid] - Sxy[valid] * Syt[valid]) / det[valid]
    Vy[valid] = (Sxy[valid] * Sxt[valid] - Sxx[valid] * Syt[valid]) / det[valid]

    return Vx, Vy




def visualize_flow(Vx, Vy, title, save_path):
    magnitude = np.sqrt(Vx ** 2 + Vy ** 2)

    # clip the color range to the 2-98th percentile so a few outlier
    # pixels  don't wash out the plot
    def get_range(arr, center_zero=False):
        lo, hi = np.percentile(arr, [2, 98])
        if center_zero:
            m = max(abs(lo), abs(hi))
            return -m, m
        return lo, hi


    vx_lo, vx_hi = get_range(Vx, center_zero=True)
    vy_lo, vy_hi = get_range(Vy, center_zero=True)
    mag_lo, mag_hi = get_range(magnitude)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(Vx, cmap="coolwarm", vmin=vx_lo, vmax=vx_hi)
    axes[0].set_title("Vx")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Vy, cmap="coolwarm", vmin=vy_lo, vmax=vy_hi)
    axes[1].set_title("Vy")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)


    im2 = axes[2].imshow(magnitude, cmap="viridis", vmin=mag_lo, vmax=mag_hi)
    axes[2].set_title("Magnitude (sqrt(Vx^2 + Vy^2))")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)




def run_optical_flow(path_a, path_b, tag):
    print(f"\nrunning optical flow on {path_a}  {path_b}")

    frame_a = load_grayscale(path_a)
    frame_b = load_grayscale(path_b)

    Vx, Vy = lucas_kanade_optical_flow(frame_a, frame_b)

    print(f"Vx range: [{Vx.min():.3f}, {Vx.max():.3f}]")
    print(f"Vy range: [{Vy.min():.3f}, {Vy.max():.3f}]")

    visualize_flow(Vx, Vy, f"Lucas-Kanade Optical Flow - {tag}",
                   save_path=f"optical_flow_{tag}.png")

    return Vx, Vy


if __name__ == "__main__":
    run_optical_flow("frame1_a.png", "frame1_b.png", "pair1")
    run_optical_flow("frame2_a.png", "frame2_b.png", "pair2")