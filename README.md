# CS 410 – Assignment 2: Lucas-Kanade Optical Flow

Programming assignment for CS 4/510: Computer Vision & Deep Learning (Summer 2026, PSU). Implements the Lucas-Kanade method for estimating optical flow between pairs of grayscale frames, built from scratch with numpy (no OpenCV flow functions).

## What it does

1. Loads a pair of grayscale frames.
2. Computes image gradients (Ix, Iy) using the Sobel-style Gx/Gy kernels from the assignment, and the temporal gradient (It) as the pixel difference between frames.
3. Builds the local 2x2 structure tensor for each pixel by summing Ix², Iy², and Ix·Iy over a 3x3 window (done via convolution with a box kernel).
4. Solves the 2x2 least-squares system at each pixel to get the flow vectors (Vx, Vy), skipping flat/low-texture regions where the system is singular.
5. Visualizes Vx, Vy, and flow magnitude as heatmaps and saves them to disk.

The convolution itself (`convolve()`) is also written by hand rather than using a library function, so it doubles as a basic 2D convolution implementation with configurable padding and kernel flipping.

## Files

- `CS410_Computer_Vission_and_Deep_learning_Assignment_2_Optic_Flow.py` — main script
- `CS 410 Computer Vision and Deep Learning Assignment 2 write up .pdf` — writeup
- `frame1_a.png`, `frame1_b.png`, `frame2_a.png`, `frame2_b.png` — input frame pairs
- `optical_flow_pair1.png`, `optical_flow_pair2.png` — output visualizations

## Requirements

```
numpy
pillow
matplotlib
```

## Usage

Run from the repo directory (frame images need to be present alongside the script):

```bash
python CS410_Computer_Vission_and_Deep_learning_Assignment_2_Optic_Flow.py
```

This runs Lucas-Kanade on both frame pairs (`pair1`, `pair2`) and saves `optical_flow_pair1.png` / `optical_flow_pair2.png`, each showing Vx, Vy, and flow magnitude side by side.

## Notes

- The gradient kernels are applied with `flip=False` since flipping them (as a "true" convolution would) flips the sign of the resulting gradients.
- The color range on each plot is clipped to the 2nd–98th percentile so a few outlier pixels don't wash out the visualization.
