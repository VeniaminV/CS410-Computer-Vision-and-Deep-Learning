 # CS 4/510: Computer Vision & Deep Learning — Programming Assignment #1

**Author:** Veniamin Velikoretskikh (veniamin@pdx.edu)
**Course:** CS 4/510, Summer 2026 — Portland State University
**Instructor:** A. Rhodes

Three-part computer vision assignment: convolution-based image filtering, K-means
clustering, and SIFT feature matching — all core algorithms implemented from
scratch in NumPy per the assignment's requirements.

## Contents

```
.
├── part1_filters.py       # Part 1: Gaussian / DoG / Sobel filtering
├── part2_kmeans.py        # Part 2: K-Means clustering (2D + RGB color space)
├── part3_sift.py          # Part 3: SIFT keypoints + custom NN matching
├── PA1_writeup.pdf        # Write-up with all figures and analysis
└── README.md
```

## Requirements

- Python 3
- `numpy`
- `pillow` (PIL)
- `matplotlib`
- `opencv-python` (Part 3 only, for SIFT)

Install everything with:

```bash
pip install numpy pillow matplotlib opencv-python
```

## Data files

Place these in the same directory as the scripts before running:

| File | Used by |
|---|---|
| `filter1_img.jpg`, `filter2_img.jpg` | Part 1 |
| `510_cluster_dataset.txt` | Part 2(i) |
| `Kmean_img1.jpg`, `Kmean_img2.jpg` | Part 2(ii) |
| `SIFT1_img.jpg`, `SIFT2_img.jpg` | Part 3 |

## Usage

Each part runs standalone:

```bash
python3 part1_filters.py
python3 part2_kmeans.py
python3 part3_sift.py
```

Each script saves its output figures (PNGs) to the working directory.

## What's implemented from scratch vs. library calls

Per the assignment rules, the *main algorithm* in each part is hand-written;
standard libraries (NumPy, plotting, image I/O) are used for supporting work.

| Part | Written from scratch | Library used |
|---|---|---|
| 1 | 2D convolution (`convolve`) with zero-padding, kernel flip | image I/O (PIL), plotting (matplotlib) |
| 2 | K-Means (init, assign, update, convergence, r-restarts, SSE) | none for the algorithm itself |
| 3 | Nearest-neighbor matching (L2 distance) + top-10% pruning | SIFT detection (`cv2.SIFT_create`) — explicitly permitted by the assignment |

## Part summaries

**Part 1 — Filtering:** Explicit 2D convolution applied with Gaussian (3x3, 5x5),
DoG (gx/gy), and Sobel (gx/gy/magnitude) kernels on two grayscale test images.

**Part 2 — K-Means:** From-scratch K-means with random-point initialization and
r=10 random restarts (keeping lowest SSE). Run on a 2D dataset (K=2,3,4) and
directly on RGB pixel values for color segmentation (K=5 and K=10 on two images).

**Part 3 — SIFT:** SIFT keypoints detected via OpenCV; nearest-neighbor matching
between two images' descriptors (minimum L2 distance) and pruning to the top 10%
of matches, both implemented by hand.

