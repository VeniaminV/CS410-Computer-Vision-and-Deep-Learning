"""
Veniamin Velikoretskikh
veniamin@pdx.edu
CS 410: Computer Vision & Deep Learning, Summer 2026
Programming Assignment #1 - Part 2

  (i)  K-Means from scratch on a 2D dataset (3 Gaussian blobs).
       Run K=2, 3, 4. For each K, try r=10 random restarts and keep
       the restart with the lowest sum-of-squares error (SSE).
  (ii) Run the SAME K-means function in 3D (R, G, B) color space on
       two real images, to segment them by color. K=5 for image 1,
       K=10 for image 2.

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def load_dataset(path):
    return np.loadtxt(path)


# Run K-means clustering on data (shape: N x D, where D=2 for the Gaussian dataset, D=3 for RGB pixels).
def kmeans(data, k, max_iters=100, seed=None, track_history=False):
   
    rng = np.random.default_rng(seed)
    n_points = data.shape[0]

    # Pick k random data points to be a starting centroids
    initial_indices = rng.choice(n_points, size=k, replace=False)
    centroids = data[initial_indices].copy()

    history = []
    labels = np.zeros(n_points, dtype=int)

    for iteration in range(max_iters):
        # For every point, compute its distance to every centroid, then
        # pick the closest one.

        diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_distances = np.sum(diffs ** 2, axis=2)  

        new_labels = np.argmin(sq_distances, axis=1)  # closest centroid per point

        
        # Move each centroid to the average position of the points
        # currently assigned to it.
        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(k):
            members = data[new_labels == cluster_id]
            if len(members) > 0:
                new_centroids[cluster_id] = members.mean(axis=0)
            else:
                # Edge case: a cluster lost all its points. Keep its old
                # centroid rather than crashing on an empty mean.
                new_centroids[cluster_id] = centroids[cluster_id]

        if track_history:
            history.append((new_centroids.copy(), new_labels.copy()))


        # If no point changed which cluster it belongs to, then we're done
        if np.array_equal(new_labels, labels) and iteration > 0:
            centroids = new_centroids
            labels = new_labels
            break

        centroids = new_centroids
        labels = new_labels

 
    # SSE = sum of squared distances from each point to its assigned centroid
    sse = 0.0
    for cluster_id in range(k):
        members = data[labels == cluster_id]
        sse += np.sum((members - centroids[cluster_id]) ** 2)

    if track_history:
        return centroids, labels, sse, history
    return centroids, labels, sse


def kmeans_best_of_r(data, k, r=10, max_iters=100, seed=None):
    """
     run the whole algorithm r independent times
    (each with a different random initialization) and just keep whichever
    run ended with the lowest SSE.
    """
    rng = np.random.default_rng(seed)
    best_sse = np.inf
    best_centroids = None
    best_labels = None

    for run in range(r):
        # Give each restart its own random seed derived from the master rng,
        # so restarts don't all pick the same initial points.
        run_seed = int(rng.integers(0, 1_000_000))
        centroids, labels, sse = kmeans(data, k, max_iters=max_iters, seed=run_seed)
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_sse


# (i) run on the 2D Gaussian dataset
# 
def plot_clusters(data, centroids, labels, k, title, save_path):
    """2D scatter plot: points colored by cluster, centroids marked with X."""
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=200,
               edgecolors="white", linewidths=1.5, label="centroids")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


# Show a handful of snapshots of the clustering as it evolves across iterations
def plot_iteration_progress(data, history, k, save_path, num_snapshots=4):
    
    total_iters = len(history)
   
    snapshot_indices = np.linspace(0, total_iters - 1, num_snapshots, dtype=int)
    snapshot_indices = sorted(set(snapshot_indices))

    fig, axes = plt.subplots(1, len(snapshot_indices), figsize=(5 * len(snapshot_indices), 5))
    if len(snapshot_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, snapshot_indices):
        centroids, labels = history[idx]
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=150,
                   edgecolors="white", linewidths=1.5)
        ax.set_title(f"Iteration {idx + 1}")

    fig.suptitle(f"K-Means progress over iterations (K={k})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def run_part2i(dataset_path, r=10, seed=42):
    print("\n--- Part 2(i): K-Means on 2D Gaussian dataset ---")
    data = load_dataset(dataset_path)

    sse_results = {}
    for k in [2, 3, 4]:
        centroids, labels, sse = kmeans_best_of_r(data, k, r=r, seed=seed)
        sse_results[k] = sse
        print(f"K={k}: best SSE over {r} restarts = {sse:.4f}")

        plot_clusters(
            data, centroids, labels, k,
            title=f"K-Means Result (K={k}), SSE={sse:.2f}",
            save_path=f"part2_kmeans_K{k}_result.png",
        )

      
        _, _, _, history = kmeans(data, k, seed=seed, track_history=True)
        plot_iteration_progress(
            data, history, k,
            save_path=f"part2_kmeans_K{k}_iterations.png",
        )

    print("\nSummary of SSE by K:")
    for k, sse in sse_results.items():
        print(f"  K={k}: SSE = {sse:.4f}")

    return sse_results


# (ii)  color clustering on real images (RGB = 3D points)

def load_rgb_pixels(path):
   
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)
    return pixels, (h, w)


def segment_image_by_color(image_path, k, r=5, seed=42, save_path=None):
    """
    Run K-means in RGB space, then rebuild an image where every pixel
    is replaced by the color of its assigned cluster's centroid. 
    """
    print(f"\nRunning K-means (K={k}) on {image_path} ...")
    pixels, (h, w) = load_rgb_pixels(image_path)

  
    centroids, labels, sse = kmeans(pixels, k, max_iters=50, seed=seed)
    print(f"  Final SSE: {sse:.2f}")

    # Rebuild the image: every pixel becomes the color of its cluster center.
    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(h, w, 3).astype(np.uint8)

    original_img = np.asarray(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(segmented_img)
    axes[1].set_title(f"K-Means Color Segmentation (K={k})")
    axes[1].axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)

    return segmented_img, sse


def run_part2ii():
    print("\n Part 2(ii): K-Means color clustering on images")
    segment_image_by_color(
        "Kmean_img1.jpg", k=5,
        save_path="part2_kmean_img1_K5.png",
    )
    segment_image_by_color(
        "Kmean_img2.jpg", k=10,
        save_path="part2_kmean_img2_K10.png",
    )


if __name__ == "__main__":
    run_part2i("510_cluster_dataset.txt", r=10, seed=42)
    run_part2ii()