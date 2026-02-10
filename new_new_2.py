import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def run_gmm_covariance_clustering(X_data, n_clusters=5):
    """
    Performs Gaussian Mixture Model clustering on the FULL dataset.
    Learns a specific Covariance Matrix for each cluster (Elliptical shapes).
    """
    print(f"--- Starting GMM (Covariance) Clustering on {len(X_data)} rows ---")
    
    # 1. Scaling
    # GMM is sensitive to scale, though less so than K-Means. 
    # MinMaxScaler (0-1) is generally safe for marketing data.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Configure GMM
    # covariance_type='full': Allows each cluster to have its own shape (ellipse)
    # reg_covar=1e-5: Adds a tiny bit of noise to prevent "Singular Matrix" crashes
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full', 
        random_state=42,
        reg_covar=1e-5,  
        n_init=3,        # Run 3 times and take best result
        verbose=1        # Show progress log
    )
    
    # 3. Fit on FULL Data
    print("Fitting GMM (This learns the covariance structures)...")
    gmm.fit(X_scaled)
    
    # 4. Predict
    labels = gmm.predict(X_scaled)
    # GMM also gives probabilities (Soft Clustering)
    probs = gmm.predict_proba(X_scaled)
    
    print("Clustering Complete.")
    print(f"Converged in {gmm.n_iter_} iterations.")
    
    return labels, probs, gmm, X_scaled

# Usage:
# labels_gmm, probs, model_gmm, X_sc = run_gmm_covariance_clustering(df_t_clean, n_clusters=7)   

from sklearn.mixture import BayesianGaussianMixture

def run_bayesian_gmm_clustering(X_data, n_clusters=10):
    """
    Performs Variational Bayesian GMM clustering on the FULL dataset.
    - Robust to overfitting (Auto-detects effective cluster count).
    - Uses Dirichlet Process Prior.
    """
    print(f"--- Starting Bayesian GMM on {len(X_data)} rows ---")
    
    # 1. Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Configure BGMM
    # weight_concentration_prior: Controls how 'sparse' the weights are.
    # Lower (e.g., 0.01) encourages fewer active clusters.
    bgmm = BayesianGaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        weight_concentration_prior=0.01,
        random_state=42,
        reg_covar=1e-5,
        n_init=1,
        verbose=1
    )
    
    # 3. Fit on FULL Data
    print("Fitting Bayesian GMM (Inference step)...")
    bgmm.fit(X_scaled)
    
    # 4. Predict
    labels = bgmm.predict(X_scaled)
    probs = bgmm.predict_proba(X_scaled)
    
    # Check effective number of clusters (clusters with non-zero weight)
    active_clusters = sum(bgmm.weights_ > 0.01) # Threshold 1%
    print(f"Clustering Complete. Effective Clusters Found: {active_clusters}/{n_clusters}")
    
    return labels, probs, bgmm, X_scaled

# Usage:
# labels_bgmm, probs_b, model_bgmm, X_sc_b = run_bayesian_gmm_clustering(df_t_clean, n_clusters=10)    

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score

def run_scalable_affinity_propagation(X_data, n_representatives=2000, damping=0.9):
    """
    Runs Affinity Propagation on 1.5M+ rows by first compressing data 
    into 'n_representatives' micro-clusters.
    
    Parameters:
    - n_representatives: Number of micro-clusters to summarize the data (e.g. 2000).
                         Higher = More detail, but slower AP.
    - damping: AP parameter (0.5 to 1.0). Controls oscillation. 0.9 is safer for convergence.
    """
    print(f"--- Starting Scalable Affinity Propagation on {len(X_data)} rows ---")
    
    # 1. Scaling (AP is distance-based)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Compression Step (Using Full Data)
    # We condense 1.5M rows into 'n_representatives' centroids.
    # This ensures we use information from the FULL dataset.
    print(f"Compressing data into {n_representatives} representatives...")
    mbk = MiniBatchKMeans(
        n_clusters=n_representatives, 
        batch_size=4096,
        random_state=42,
        n_init=3
    )
    # We fit on the FULL data to find the best representatives
    mbk_labels = mbk.fit_predict(X_scaled) 
    representatives = mbk.cluster_centers_
    
    # 3. Run Affinity Propagation on Representatives
    # Now we cluster the *centroids*, which allows AP to find the "Exemplars"
    # without the N^2 memory crash.
    print("Running Affinity Propagation on representatives...")
    ap = AffinityPropagation(
        damping=damping,
        random_state=42,
        max_iter=400,
        preference=None # None = median similarity (automatically determines cluster count)
    )
    
    ap.fit(representatives)
    
    # 4. Map Labels Back to Full Data
    # mbk_labels maps: User -> Micro-Cluster (0 to 1999)
    # ap.labels_ maps: Micro-Cluster -> Final AP Cluster
    print("Propagating labels to full population...")
    
    # We use the micro-cluster label to look up the final AP cluster label
    final_labels = ap.labels_[mbk_labels]
    
    n_clusters_found = len(set(final_labels))
    print(f"Affinity Propagation converged. Found {n_clusters_found} clusters.")
    
    return final_labels, ap, X_scaled

# Usage:
# Note: AP determines the number of clusters AUTOMATICALLY. You do not set n_clusters.
# labels_ap, model_ap, X_sc = run_scalable_affinity_propagation(df_t_clean, n_representatives=2500)

# Validation
# validate_clusters_fast(X_sc, labels_ap)  

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

def run_scalable_kmeans(X_data, n_clusters=5):
    """
    Fits K-Means on the FULL dataset using Mini-Batch processing.
    Memory efficient and fast for 1.5M+ rows.
    """
    print(f"--- Starting Scalable K-Means on {len(X_data)} customers ---")
    
    # 1. Scale Data (K-Means requires 0-1 scaling)
    # If X_data is already scaled, you can comment this out
    if np.max(X_data) > 1.05 or np.min(X_data) < -0.05:
        print("Scaling data...")
        scaler = MinMaxScaler()
        X_processed = scaler.fit_transform(X_data)
    else:
        X_processed = X_data

    # 2. Initialize MiniBatch K-Means
    # batch_size=4096: Processes 4k rows at a time (Memory Safe)
    # n_init=10: Runs the initialization 10 times to find the best start
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=4096,
        random_state=42,
        n_init=10,
        verbose=0
    )
    
    # 3. Fit on FULL Data
    print("Fitting model to full dataset...")
    kmeans.fit(X_processed)
    
    # 4. Predict Labels
    labels = kmeans.labels_
    
    print(f"K-Means Complete. Inertia: {kmeans.inertia_:,.0f}")
    return labels, kmeans, X_processed

# Usage:
# labels_kmeans, model_kmeans, X_scaled = run_scalable_kmeans(df_t_clean, n_clusters=5)   
