# CELL 1: CLUSTERING PIPELINE
import umap
import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

def perform_clustering(df_clean, final_features):
    """
    Executes the High-Performance Clustering Pipeline.
    1. UMAP: Compresses 50+ features to 5 dimensions (preserving structure).
    2. HDBSCAN: Finds dense clusters and isolates noise (-1).
    """
    print(f"Starting clustering on {len(df_clean)} customers using {len(final_features)} features...")
    
    # 1. Prepare Data
    X = df_clean[final_features].values
    
    # 2. UMAP Dimensionality Reduction
    # n_neighbors=30: Balances local detail (micro-segments) with global structure.
    # min_dist=0.0: Allows points to clump tightly (good for distinct segments).
    umap_reducer = umap.UMAP(
        n_neighbors=30,
        n_components=5, 
        min_dist=0.0,
        metric='manhattan', # Manhattan is better for high-dim sparse data
        random_state=42
    )
    embedding = umap_reducer.fit_transform(X)
    print("UMAP Dimensionality Reduction complete.")
    
    # 3. HDBSCAN Density Clustering
    # min_cluster_size: Smallest group to consider a "Segment" (e.g., 1% of pop).
    # min_samples: Higher values make it more conservative (more points labeled as noise).
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=500, # Adjust based on your total N (e.g., N/200)
        min_samples=50, 
        metric='euclidean', 
        cluster_selection_method='eom',
        prediction_data=True
    )
    labels = clusterer.fit_predict(embedding)
    
    # Attach results back to dataframe
    df_results = df_clean.copy()
    df_results['Cluster'] = labels
    df_results['Probabilities'] = clusterer.probabilities_
    
    # Count Clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Found {n_clusters} Clusters.")
    print(f"Noise Points Detected: {n_noise} ({n_noise/len(X):.1%})")
    
    return df_results, embedding, labels

# Usage:
# df_clustered, embedding, labels = perform_clustering(df_clean, final_features)


# CELL 2: VALIDATION METRICS
def validate_clusters(X_data, labels):
    """
    Calculates technical validity scores.
    """
    # Filter out Noise (-1) for fair scoring
    mask = labels != -1
    X_valid = X_data[mask]
    labels_valid = labels[mask]
    
    if len(set(labels_valid)) < 2:
        print("Validation Failed: Less than 2 clusters found (excluding noise).")
        return
    
    # 1. Silhouette Score (Shape tightness)
    # Range: -1 to 1. Higher is better. >0.4 is generally good for behavioral data.
    # Note: Using a sample of 10k points for speed if dataset is huge
    sample_size = min(10000, len(X_valid))
    sil_score = silhouette_score(X_valid, labels_valid, sample_size=sample_size)
    
    print(f"--- Cluster Validation ---")
    print(f"Silhouette Score (Density): {sil_score:.3f}")
    
    # 2. Size Distribution Check
    counts = pd.Series(labels).value_counts().sort_index()
    print("\nCluster Sizes:")
    print(counts)
    
    # Warning if one cluster dominates > 80% of data (Implies failed separation)
    if counts.max() / len(labels) > 0.8:
        print("\nWARNING: One cluster dominates >80% of data. Try lowering min_cluster_size.")
        
# Usage:
# validate_clusters(df_clean[final_features].values, labels)

# CELL 3: BUSINESS INSIGHTS PROFILER
def get_cluster_insights(df_clustered, profiling_features):
    """
    Generates a 'Snake Plot' style table comparing each cluster to the global average.
    """
    # 1. Calculate Global Means (The Baseline)
    global_means = df_clustered[profiling_features].mean()
    
    # 2. Calculate Cluster Means
    # Group by Cluster and take mean of original (non-logged) features for readability
    cluster_profiles = df_clustered.groupby('Cluster')[profiling_features].mean()
    
    # 3. Calculate "Lift" (How much higher/lower is this cluster than average?)
    # Lift = Cluster_Mean / Global_Mean. 
    # 1.0 = Average. 2.0 = 2x Average. 0.5 = Half Average.
    lift_table = cluster_profiles.div(global_means).round(2)
    
    # 4. Add Counts and Response Rates (if available)
    lift_table['Count'] = df_clustered['Cluster'].value_counts()
    
    # If you have the target/response column, add it to see which cluster performs best
    # Assuming 'OFFER_STATUS' or similar exists where 1 = Responded
    # lift_table['Response_Rate'] = df_clustered.groupby('Cluster')['IS_RESPONDER'].mean()

    return lift_table.sort_values('Count', ascending=False)

# Usage Configuration
# Select RAW features (Dollars, Points), NOT the log-transformed ones for readability.
profile_cols = [
    'LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 
    'POINTS_BALANCE', 'TOTAL_RDM_SINCE_2020',
    'ratio_burn_rate', 'tenure_months' 
]
# insights = get_cluster_insights(df_clustered, profile_cols)
# print(insights)


