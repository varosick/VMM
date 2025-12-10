import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

from src.extract_sift import DATA_DIR

DESCRIPTORS_FILE = os.path.join(DATA_DIR, "descriptors.pkl")  # Path to stored SIFT descriptors
KMEANS_FILE = os.path.join(DATA_DIR, "kmeans_model.pkl")      # Output file for trained k-means model

K = 700  # Size of the visual vocabulary (number of visual words)


def main():
    # Load precomputed SIFT descriptors for all images.
    with open(DESCRIPTORS_FILE, "rb") as f:
        descriptors_dict = pickle.load(f)

    all_samples = []  # Will store sampled descriptors for training k-means

    print("Sampling descriptors...")

    # For each image, select up to 200 random descriptors.
    # This avoids using millions of descriptors and stabilizes k-means training.
    for des in descriptors_dict.values():
        if des.shape[0] > 200:
            # Randomly sample 200 descriptor rows without replacement
            idx = np.random.choice(des.shape[0], 200, replace=False)
            sample = des[idx]
        else:
            # If the image has fewer than 200 descriptors, use all of them
            sample = des

        all_samples.append(sample)

    # Combine selected descriptors into a single matrix
    all_samples = np.vstack(all_samples)
    print("Total sampled descriptors:", all_samples.shape)

    print("Training k-means...")

    # batch_size controls how many points are processed per iteration.
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2000, verbose=1)
    kmeans.fit(all_samples)

    # Save the trained model for use in BoW construction
    with open(KMEANS_FILE, "wb") as f:
        pickle.dump(kmeans, f)

    print("K-means model saved:", KMEANS_FILE)


if __name__ == "__main__":
    main()
