# CELL 3: SPECTRAL VARIANTS (BIRCH & BISECTING)
from sklearn.cluster import Birch, BisectingKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import Nystroem

def run_nystroem_variant(X_data, n_clusters=5, algorithm='birch'):
    """
    Runs Spectral Clustering using Nyström Embedding paired with 
    scalable alternatives to K-Means (BIRCH or Bisecting K-Means).
    
    Parameters:
    - algorithm: 'birch' (Great for outliers) or 'bisecting' (Great for hierarchy)
    """
    print(f"--- Starting Spectral Clustering ({algorithm.upper()}) on {len(X_data)} customers ---")
    
    # 1. Scaling (Required for Nyström/Spectral)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Nyström Embedding (The "Spectral" Part)
    # Maps data to 100-dim non-linear spectral space
    print("Approximating Spectral Embeddings...")
    nystroem = Nystroem(
        kernel='rbf', 
        gamma=None,       # Defaults to 1/n_features
        n_components=100, # 100 landmarks
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Select Clustering Algorithm
    if algorithm == 'birch':
        # BIRCH: Builds a tree (CF Tree) to compress data. 
        # threshold: Radius of the subcluster. 0.01 is a good starting point for scaled data.
        # branching_factor: Max children per node. 50 is standard for large data.
        clusterer = Birch(
            n_clusters=n_clusters,
            threshold=0.01, 
            branching_factor=50
        )
    elif algorithm == 'bisecting':
        # Bisecting K-Means: Divisive hierarchical clustering.
        # bisecting_strategy='largest_cluster': Splits the biggest blob first (good for imbalanced data)
        clusterer = BisectingKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            bisecting_strategy='largest_cluster',
            random_state=42
        )
    else:
        raise ValueError("Algorithm must be 'birch' or 'bisecting'")

    # 4. Pipeline Execution
    pipeline = Pipeline([
        ('spectral_embedding', nystroem),
        ('clusterer', clusterer)
    ])
    
    print(f"Fitting {algorithm.upper()} to Spectral Embeddings...")
    pipeline.fit(X_scaled)
    labels = pipeline.predict(X_scaled)
    
    print("Clustering Complete.")
    return labels, pipeline, X_scaled

# Usage Examples:
# labels_birch, model_birch, X_sc = run_nystroem_variant(X_clean, n_clusters=5, algorithm='birch')


#####################################################33


# CELL 4: RBF SAMPLER PIPELINE (Alternative Embedding)
from sklearn.kernel_approximation import RBFSampler

def run_rff_spectral(X_data, n_clusters=5):
    """
    Performs Spectral Clustering using Random Fourier Features (RBF Sampler).
    This is a Monte-Carlo alternative to Nyström that is mathematically 
    independent of the data distribution (smoother embedding).
    """
    print(f"--- Starting RBF Sampler Spectral Clustering on {len(X_data)} customers ---")
    
    # 1. Scaling (Crucial for RBF Sampler)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. RBF Sampler Embedding (The "Random Fourier" Part)
    # Approximates the RBF kernel using Monte Carlo sampling.
    # gamma=1.0: Standard for 0-1 scaled data. Lower (e.g. 0.1) = looser clusters.
    print("Approximating Kernel Map using Random Fourier Features...")
    rbf_feature = RBFSampler(
        gamma=1.0, 
        n_components=100, 
        random_state=42
    )
    
    # 3. Clustering (Standard K-Means works well on RFF embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # 4. Pipeline Execution
    pipeline = Pipeline([
        ('rff_embedding', rbf_feature),
        ('clusterer', kmeans)
    ])
    
    print("Fitting K-Means to RBF Embeddings...")
    pipeline.fit(X_scaled)
    labels = pipeline.predict(X_scaled)
    
    print("Clustering Complete.")
    return labels, pipeline, X_scaled

# Usage:
# labels_rff, model_rff, X_sc = run_rff_spectral(X_clean, n_clusters=5)
# labels_bisect, model_bisect, X_sc = run_nystroem_variant(X_clean, n_clusters=5, algorithm='bisecting')
