import cv2
import os
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DESCRIPTORS = os.path.join(DATA_DIR, "descriptors.pkl")


def extract_sift_from_image(path):
    """
    Extracts SIFT descriptors from a single image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale (required for SIFT)

    # Initialize SIFT detector with a maximum of 2000 features per image.
    # Limiting the number of features helps keep descriptors consistent across images.
    sift = cv2.SIFT_create(nfeatures=2000)

    # Extract keypoints and descriptors.s
    kp, des = sift.detectAndCompute(img, None)

    return des


def main():
    """
    Iterates through all images in IMAGES_DIR, extracts SIFT descriptors for each image,
    and stores them in a dictionary. The dictionary is then saved to a pickle file.
    """
    descriptors = {}

    for filename in os.listdir(IMAGES_DIR):
        path = os.path.join(IMAGES_DIR, filename)

        print(f"Extracting SIFT from {filename} ...")

        des = extract_sift_from_image(path)

        # Only save entries that contain valid descriptor data
        if des is not None:
            descriptors[filename] = des

    # Save all descriptors to a pickle file for future processing
    with open(OUTPUT_DESCRIPTORS, "wb") as f:
        pickle.dump(descriptors, f)

    print("SIFT descriptors saved to", OUTPUT_DESCRIPTORS)


if __name__ == "__main__":
    main()
