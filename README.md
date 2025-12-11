# Bag of Features

This project implements an image similarity search system using the **Bag of Features** model with:

- SIFT features  
- k-means visual vocabulary
- TF-IDF weighted histograms  
- Cosine similarity  

Backend: **FastAPI**  
Frontend: **React/Vite**

---

# Project Structure

```
VMM/
│
├── backend/
├── frontend/
│
├── data/
│   ├── descriptors.pkl
│   ├── kmeans_model.pkl
│   ├── idf.npy
│   └── bow_vectors.pkl
│
├── images/          
│
├── src/
│   ├── extract_sift.py
│   ├── build_dictionary.py
│   ├── compute_bow.py
│   └── search.py
│
├── uploads/
├── requirements.txt
└── README.md
```

---

# Algorithm Overview

## 1. Extracting SIFT Descriptors
### Each image from /images is converted into SIFT descriptors using:
```python
sift = cv2.SIFT_create(nfeatures=2000)
kp, des = sift.detectAndCompute(img, None)

descriptors[filename] = des
```
### This means:

* At most 2000 keypoints are extracted.
* Each keypoint becomes a 128-dimensional descriptor.
* Output: matrix des of shape (N, 128).
* Descriptors saved to `data/descriptors.pkl`.

## 2. Building Visual Vocabulary (k-means)

### Instead of using all descriptors (which may be millions), the code samples only 200 random descriptors per image:

```python
idx = np.random.choice(des.shape[0], 200, replace=False)
sample = des[idx]
```

### All samples are concatenated:
```python
all_samples = np.vstack(all_samples)
```

### Then k-means with K = 700 clusters is trained:
```python
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=2000, verbose=1)
kmeans.fit(all_samples)
```
### Each centroid of k-means becomes a visual word.
The vocabulary is saved as: `data/kmeans_model.pkl`
Training k-means:

```python
kmeans = MiniBatchKMeans(n_clusters=700, batch_size=2000)
```

## 3. Constructing TF-IDF BoW

### Quantizing descriptors into visual words:

```python
labels = kmeans.predict(des)
```
This transforms each descriptor into an integer cluster index [0 … K-1].


### Building the histogram:

```python
hist, _ = np.histogram(labels, bins=np.arange(K + 1))
```

### L1 normalization:

```python
hist = hist.astype(float)
hist /= hist.sum()
```
This ensures all histograms are comparable across images.

### Computing IDF:

```python
idf = np.log((N + 1) / (df + 1))
```
Where df is updated using:
```python
df += (hist > 0).astype(int)
```

### TF-IDF + L2 normalization
```python
tfidf = hist * idf
tfidf = tfidf / np.linalg.norm(tfidf)
```

The final vectors are saved to: `data/bow_vectors.pkl`

## 4. Searching for Similar Images
Given a query image, the function search_similar_image() performs the entire retrieval pipeline.

### Extract SIFT from quer
```python
des_query = extract_sift_from_image(query_image)
```
### Quantize query descriptor
```python
labels = kmeans.predict(des_query)
```
### Build query histogra
```python
hist, _ = np.histogram(labels, bins=np.arange(K + 1))
```
### L1 normalizatio
```python
hist /= hist.sum()
```
### Apply TF-ID
```python
tfidf = hist * idf
```
### L2 normalizatio
```python
tfidf = tfidf / np.linalg.norm(tfidf)
```
### Cosine similarity with database
This line performs the actual similarity computation:
```python
score = np.dot(tfidf, bow)
```
Because all vectors are L2-normalized:
```python
cosine_similarity = dot product
```
### Sort results
```python
results.sort(key=lambda x: -x[1])
```

This ranking is based on TF-IDF cosine similarity.

# Backend (FastAPI)

Handles uploads, calls search function, returns similarity results.
```python
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = search_similar_image(file_path)

    return JSONResponse({"uploaded_file": file_path, "result": result})
```

# Frontend (React)

Allows user to upload an image and displays similar images returned by backend.

---

# Installation

```bash
pip install -r requirements.txt
```

Build index:

```bash
python src/extract_sift.py
python src/build_dictionary.py
python src/compute_bow.py
```

Run backend:

```bash
uvicorn backend.main:app --reload
```

Run frontend:

```bash
npm install
npm run dev
```
