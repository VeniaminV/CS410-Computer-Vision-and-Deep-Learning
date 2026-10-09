"""
Veniamin Velikoretskikh
veniamin@pdx.edu
CS 410: Computer Vision & Deep Learning, Summer 2026
Programming Assignment #1 - Part 3

  (i) Detect SIFT keypoints in two images 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_sift_features(image_path):
    
    img_bgr = cv2.imread(image_path)
    img_color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    return img_color, img_gray, keypoints, descriptors


# Draw the detected keypoints on top of the image. Using DRAW_RICH_KEYPOINTS 
def visualize_keypoints(img_color, keypoints, title, save_path):
    
    img_with_kp = cv2.drawKeypoints(
        img_color, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_with_kp)
    ax.set_title(f"{title} ({len(keypoints)} keypoints)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


"""
For every descriptor (keypoint) in image 1, find which descriptor in
    image 2 is closest to it in L2 (Euclidean) distance, over all 128
    dimensions of the SIFT vector.
"""
def find_nearest_neighbor_matches(descriptors1, descriptors2):
    
    n1 = descriptors1.shape[0]
    matches = []

    for i in range(n1):
        # Distance from THIS descriptor in image 1 to every descriptor
        # in image 2, all at once (vectorized subtraction + norm).
        diffs = descriptors2 - descriptors1[i]          
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))  

        best_j = np.argmin(distances)       # index of the closest match in image 2
        best_dist = distances[best_j]

        matches.append((i, best_j, best_dist))

    return matches



#Sort all matches by distance, smallest (best) first, and keep only the closest 10%.
def keep_top_10_percent(matches):
    
    sorted_matches = sorted(matches, key=lambda m: m[2])  # sort by distance
    n_keep = max(1, int(round(len(sorted_matches) * 0.10)))
    return sorted_matches[:n_keep]



# Visualize the surviving matches with connecting lines
def visualize_matches(img1, kp1, img2, kp2, matches, title, save_path):
    
    dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=float(d))
                for (i, j, d) in matches]

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    match_img = cv2.drawMatches(
        img1_bgr, kp1, img2_bgr, kp2, dmatches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(match_img)
    ax.set_title(f"{title} ({len(matches)} matches shown)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)



def run_part3(path1, path2):
    print("\nPart 3: SIFT keypoints + custom NN matching ")

    img1_color, img1_gray, kp1, desc1 = detect_sift_features(path1)
    img2_color, img2_gray, kp2, desc2 = detect_sift_features(path2)
    print(f"Image 1: {len(kp1)} keypoints found")
    print(f"Image 2: {len(kp2)} keypoints found")

    # Visualize all detected keypoints on each image
    visualize_keypoints(img1_color, kp1, "SIFT Keypoints - Image 1",
                        save_path="part3_sift1_keypoints.png")
    visualize_keypoints(img2_color, kp2, "SIFT Keypoints - Image 2",
                        save_path="part3_sift2_keypoints.png")

    # Nearest-neighbor matching 
    print("Computing nearest-neighbor matches ")
    all_matches = find_nearest_neighbor_matches(desc1, desc2)
    print(f"Total matches computed: {len(all_matches)}")

    # Keep only the closest 10% 
    top_matches = keep_top_10_percent(all_matches)
    print(f"Top 10% kept: {len(top_matches)} matches")
    print(f"Distance range of kept matches: "
          f"{top_matches[0][2]:.2f} (best) to {top_matches[-1][2]:.2f} (worst kept)")

    
    visualize_matches(
        img1_color, kp1, img2_color, kp2, top_matches,
        title="Top 10% SIFT Keypoint Matches",
        save_path="part3_top10pct_matches.png",
    )


if __name__ == "__main__":
    run_part3("SIFT1_img.jpg", "SIFT2_img.jpg")