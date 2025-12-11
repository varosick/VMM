import pickle
import numpy as np
import json
from src.extract_sift import extract_sift_from_image
from src.build_dictionary import K
import os

from src.extract_sift import DATA_DIR

KMEANS_FILE = os.path.join(DATA_DIR, "kmeans_model.pkl")
IDF_FILE = os.path.join(DATA_DIR, "idf.npy")
BOW_VECTORS_FILE = os.path.join(DATA_DIR, "bow_vectors.pkl")


def search_similar_image(query_image):
    """
    Given a query image, this function computes its TF-IDF BoW vector and compares it
    to the database vectors using cosine similarity. It returns json of the top-10 most similar images.
    """

    # Extract SIFT descriptors from the query image
    des_query = extract_sift_from_image(query_image)

    # If the image has no detectable SIFT features, similarity search is impossible
    if des_query is None:
        print("No SIFT features found in query image")
        return

    # Load pretrained models
    with open(KMEANS_FILE, "rb") as f:
        kmeans = pickle.load(f) # k-means model used for descriptor quantization

    idf = np.load(IDF_FILE) # IDF vector for TF-IDF weighting

    with open(BOW_VECTORS_FILE, "rb") as f:
        bow_db = pickle.load(f) # Dictionary: {filename: TF-IDF vector}


    # Build BoW for the query image

    # Assign each SIFT descriptor to a visual word (cluster index)
    labels = kmeans.predict(des_query)

    # Count occurrences of each visual word
    hist, _ = np.histogram(labels, bins=np.arange(K + 1))

    # L1 normalization (convert counts to frequencies)
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()

    # TF-IDF weighting
    tfidf = hist * idf

    # L2 normalization (required for cosine similarity)
    norm = np.linalg.norm(tfidf)
    if norm > 0:
        tfidf = tfidf / norm


    # Compare query vector to database vectors
    results = []

    for filename, bow in bow_db.items():
        # Cosine similarity = dot product (because both vectors are L2-normalized)
        score = np.dot(tfidf, bow)
        results.append((filename, score))

    # Sort by similarity score in descending order
    results.sort(key=lambda x: -x[1])

    images = {fname : score for fname, score in results[:10]}
    return json.dumps(images)

