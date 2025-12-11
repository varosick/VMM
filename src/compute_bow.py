import pickle
import numpy as np
from src.build_dictionary import K
import os

from src.extract_sift import DATA_DIR

DESCRIPTORS_FILE = os.path.join(DATA_DIR, "descriptors.pkl")
KMEANS_FILE = os.path.join(DATA_DIR, "kmeans_model.pkl")
IDF_FILE = os.path.join(DATA_DIR, "idf.npy")
BOW_VECTORS_FILE = os.path.join(DATA_DIR, "bow_vectors.pkl")


def main():
    # Load precomputed SIFT descriptors
    with open(DESCRIPTORS_FILE, "rb") as f:
        descriptors_dict = pickle.load(f)

    # Load previously trained k-means visual vocabulary
    with open(KMEANS_FILE, "rb") as f:
        kmeans = pickle.load(f)

    bow_vectors = {}
    df = np.zeros(K) # Document frequency for each visual word

    print("Building BoW histograms...")

    # Build raw TF histograms
    for filename, des in descriptors_dict.items():

        # Assign each descriptor to the closest visual word
        labels = kmeans.predict(des)

        # Count occurrences of each visual word to build a histogram
        hist, _ = np.histogram(labels, bins=np.arange(K + 1))

        # L1 normalization (make histogram sum to 1)
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()

        bow_vectors[filename] = hist

        # Count how many images contain each visual word at least once
        df += (hist > 0).astype(int)

    # Compute IDF
    N = len(bow_vectors) # Total number of images
    # Standard IDF formula with smoothing
    idf = np.log((N + 1) / (df + 1))

    # Save IDF to file so search.py can use it later
    np.save(IDF_FILE, idf)
    print("IDF saved:", IDF_FILE)


    # Apply TF-IDF and L2 normalize
    print("Applying TF-IDF...")

    for filename in bow_vectors:
        tfidf = bow_vectors[filename] * idf # Multiply TF by IDF

        # L2 normalization for cosine similarity compatibility
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        bow_vectors[filename] = tfidf

    # Save final representation of all images
    with open(BOW_VECTORS_FILE, "wb") as f:
        pickle.dump(bow_vectors, f)

    print("TF-IDF BoW vectors saved:", BOW_VECTORS_FILE)


if __name__ == "__main__":
    main()
