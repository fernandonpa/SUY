import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# SUB-CLUSTERING DRILL-DOWN FUNCTION
# ======================================================
def run_sub_clustering_analysis(df_engineered, df_raw, target_cluster_id, features_to_use, n_sub_clusters=3):
    """
    Isolates a parent cluster, re-scales data, runs sub-clustering,
    and returns detailed insights for the new micro-segments.
    
    Parameters:
    - df_engineered: Dataframe with Log/Ratio features (used for clustering model)
    - df_raw: Original Dataframe with real $ amounts (used for insight profiling)
    - target_cluster_id: The specific cluster ID you want to split (e.g., 11)
    - features_to_use: List of feature names (candidate_pool)
    """
    print(f"--- Drilling Down into Cluster {target_cluster_id} ---")
    
    # 1. Isolate the Target Population (Engineered Data)
    # Ensure 'Cluster' column exists in your df_engineered
    if 'Cluster' not in df_engineered.columns:
        raise ValueError("df_engineered must have a 'Cluster' column from the main model.")
        
    target_indices = df_engineered[df_engineered['Cluster'] == target_cluster_id].index
    
    if len(target_indices) == 0:
        print(f"Error: Cluster {target_cluster_id} is empty.")
        return None, None
        
    X_sub = df_engineered.loc[target_indices, features_to_use].copy()
    
    # 2. Isolate Raw Data (for Profiling later)
    sub_df_raw = df_raw.loc[target_indices].copy()
    
    print(f"Target Population Size: {len(X_sub):,}")
    
    # 3. Re-Scaling (The "Zoom In" Effect) - CRITICAL STEP
    # We fit the scaler ONLY on this cluster's data to maximize variance detection
    scaler = MinMaxScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    
    # 4. Run Sub-Clustering
    # BisectingKMeans is often superior for splitting homogeneous groups
    print(f"Running Sub-Clustering (k={n_sub_clusters})...")
    clusterer = BisectingKMeans(n_clusters=n_sub_clusters, random_state=42)
    sub_labels = clusterer.fit_predict(X_sub_scaled)
    
    # 5. Attach Labels to Raw Data for Profiling
    sub_df_raw['Sub_Cluster'] = sub_labels
    
    # 6. Generate Deep Insights
    # We call your EXISTING robust function 'get_cluster_deep_insights'
    # Note: We pass sub_df_raw because we want insights on real $ amounts, not logs
    print("Generating Insights Table...")
    insights_table, raw_stats = get_cluster_deep_insights(sub_df_raw, sub_labels)
    
    # Rename index to avoid confusion with main clusters
    insights_table.index = [f"{target_cluster_id}.{i}" for i in insights_table.index]
    
    return sub_df_raw, insights_table

# ======================================================
# EXECUTION EXAMPLE
# ======================================================

# Configuration
TARGET_ID = 11  # The cluster you want to split
N_SUB = 4       # How many sub-segments to find

# 1. Ensure your engineered dataframe has the main cluster labels
# (Assuming 'labels' is your result from the main spectral model)
# df_engineered['Cluster'] = labels 

# 2. Run the Drill-Down
# df_engineered = The dataframe with log_SPEND, ratio_burn_rate, etc.
# df_train = The dataframe with original LAST_12MTH_SPEND, etc.
sub_df_11, sub_insights_11 = run_sub_clustering_analysis(
    df_engineered=df_t_clean,     # Use the CLEANED/ENGINEERED data for math
    df_raw=df_train,              # Use the RAW data for stats
    target_cluster_id=TARGET_ID,
    features_to_use=candidate_pool, # Using the exact same features
    n_sub_clusters=N_SUB
)

# 3. Display Results
print(f"\nMicro-Segments for Cluster {TARGET_ID}:")
display(sub_insights_11.style.background_gradient(cmap='YlGn'))
