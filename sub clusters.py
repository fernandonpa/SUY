from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_sub_clustering(df_original, df_labeled, target_cluster_id, n_sub_clusters=3, sub_features=None):
    """
    Isolates a specific cluster and breaks it down into sub-segments.
    
    Parameters:
    - df_original: The raw data (for profiling)
    - df_labeled: The dataframe containing the 'Cluster' column from Step 1
    - target_cluster_id: The ID of the cluster you want to split (e.g., 11)
    - sub_features: (Optional) List of features specific for this sub-layer. 
                    If None, uses the original candidate_pool.
    """
    print(f"--- Sub-Clustering Target: Cluster {target_cluster_id} ---")
    
    # 1. Isolate the Data
    # Get indices of the target cluster
    target_indices = df_labeled[df_labeled['Cluster'] == target_cluster_id].index
    
    # Extract the relevant subset from the engineered/scaled data
    # NOTE: It's best to re-scale because the range of values in this subset 
    # is different from the global range.
    sub_df = df_original.loc[target_indices].copy()
    
    print(f"Population Size: {len(sub_df):,}")
    
    # 2. Define Features for Sub-Clustering
    # If no specific features provided, use a default behavioral set
    if sub_features is None:
        # Default to behavior/ratios if not specified
        sub_features = [
            'ratio_burn_rate', 'ratio_spend_velocity', 
            'log_POINTS_BALANCE', 'MES_ENG_SCORE'
        ]
        
    print(f"Features used for splitting: {sub_features}")
    
    # 3. Re-Engineering / Re-Scaling (Crucial for Local Variance)
    # We essentially repeat the engineering/scaling just for this group
    # Assumes 'sub_df' is the Raw Data. If it's already engineered, skip engineer_features_v2
    
    # Quick Scaling
    X_sub = sub_df[sub_features].fillna(0)
    scaler = MinMaxScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    
    # 4. Run Sub-Clustering (K-Means is usually fine here)
    # We use K-Means because we are now looking for tighter variances within a group
    kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
    sub_labels = kmeans.fit_predict(X_sub_scaled)
    
    # 5. Attach Results
    sub_df['Sub_Cluster'] = sub_labels
    
    # 6. Profile the Sub-Clusters
    print("\n--- Sub-Cluster Profiles ---")
    summary = sub_df.groupby('Sub_Cluster')[sub_features].mean()
    
    # Add count
    summary['Count'] = sub_df['Sub_Cluster'].value_counts()
    
    return sub_df, summary

# Usage:
# 1. Define features that distinguish users WITHIN this group
# Example: If Cluster 11 is "High Spenders", split them by "Loyalty Behavior"
# distinctive_features = ['ratio_burn_rate', 'flag_digital_active', 'D_MONTHONBOOK']

# 2. Run
# sub_df, sub_profile = run_sub_clustering(
#     df_original=df_train,   # Use Raw Data
#     df_labeled=df_train,    # Data with 'Cluster' col
#     target_cluster_id=11,   # The cluster to split
#     n_sub_clusters=3,       # How many micro-segments?
#     sub_features=distinctive_features
# )

# display(sub_profile)


# CELL: SUB-CLUSTERING WITH ORIGINAL FEATURES

def run_sub_clustering_original_features(df_original, df_labeled, target_cluster_id, original_features, n_sub_clusters=3):
    """
    Sub-clusters a specific group using the SAME features used in the main model.
    """
    print(f"--- Sub-Clustering Cluster {target_cluster_id} using ORIGINAL Features ---")
    
    # 1. Filter Data to the Target Cluster
    target_indices = df_labeled[df_labeled['Cluster'] == target_cluster_id].index
    sub_df = df_original.loc[target_indices].copy()
    
    print(f"Population: {len(sub_df):,}")
    
    # 2. Prepare Features (Re-use the original list)
    # We must RE-SCALE because the min/max within this cluster is different from the global min/max
    X_sub = sub_df[original_features].fillna(0)
    
    scaler = MinMaxScaler()
    X_sub_scaled = scaler.fit_transform(X_sub)
    
    # 3. Run Sub-Clustering (K-Means is perfect for this)
    kmeans = KMeans(n_clusters=n_sub_clusters, random_state=42, n_init=10)
    sub_labels = kmeans.fit_predict(X_sub_scaled)
    
    # 4. Attach & Profile
    sub_df['Sub_Cluster'] = sub_labels
    
    # Profiling: Group by the new sub-clusters to see differences
    profile = sub_df.groupby('Sub_Cluster')[original_features].mean()
    profile['Count'] = sub_df['Sub_Cluster'].value_counts()
    
    return sub_df, profile

# ==========================================
# EXECUTION
# ==========================================

# 1. Identify your "High Value" or "Target" cluster from previous steps
# Let's assume Cluster 2 is the one you want to break down
target_id = 2 

# 2. Run using the 'candidate_pool' (The list you created earlier)
# Ensure 'df_train' (or whatever your main df is) has the raw values for these columns
df_sub, sub_profile = run_sub_clustering_original_features(
    df_original=df_train,   # Raw dataframe
    df_labeled=df_train,    # Dataframe that has the 'Cluster' column
    target_cluster_id=target_id,
    original_features=candidate_pool, # <--- PASSING THE ORIGINAL LIST
    n_sub_clusters=3
)

print("\nSub-Cluster Insights:")
display(sub_profile.style.background_gradient(cmap='Greens'))
