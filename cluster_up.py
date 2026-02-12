# Install necessary libraries if not present
!pip install smote_variants scikit-learn-extra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smote_variants as sv

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.kernel_approximation import Nystroem
from sklearn_extra.cluster import KMedoids  # Requires scikit-learn-extra
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. PREPARE DATA
# Assuming 'df' is your DataFrame and 'target_applied' is your label
# X = df.drop('target_applied', axis=1)
# y = df['target_applied']

# Separate Majority (0) and Minority (1)
X_maj = X[y == 0].values
y_maj = y[y == 0].values

X_min = X[y == 1].values
y_min = y[y == 1].values

print(f"Majority Samples: {len(X_maj)}")
print(f"Minority Samples (to cluster): {len(X_min)}")

# 2. SCALE DATA (Critical for Clustering)
scaler = StandardScaler()
X_min_scaled = scaler.fit_transform(X_min)


def evaluate_clustering_methods(X_data, k_range):
    results = []
    
    print("Evaluating Clustering Methods...")
    
    for k in k_range:
        # Define Algorithms
        algorithms = {
            'K-Means': KMeans(n_clusters=k, random_state=42, n_init=10),
            'K-Medoids': KMedoids(n_clusters=k, random_state=42),
            'GMM': GaussianMixture(n_components=k, random_state=42),
            'Agnes (Agglomerative)': AgglomerativeClustering(n_clusters=k), # Default is Ward linkage
            'Nystroem+KMeans': 'pipeline' # Handled specifically below
        }

        for name, algo in algorithms.items():
            try:
                # Handle Special Case: Nystroem + KMeans
                if name == 'Nystroem+KMeans':
                    feature_map = Nystroem(gamma=.2, random_state=42, n_components=100)
                    X_transformed = feature_map.fit_transform(X_data)
                    kmeans_nystroem = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans_nystroem.fit_predict(X_transformed)
                
                # Handle Standard Cases
                else:
                    if name == 'GMM':
                        labels = algo.fit_predict(X_data)
                    else:
                        labels = algo.fit_predict(X_data)

                # Calculate Score
                score = silhouette_score(X_data, labels)
                results.append({
                    'Method': name,
                    'K': k,
                    'Score': score,
                    'Labels': labels
                })
            except Exception as e:
                print(f"Skipped {name} k={k}: {e}")

    return pd.DataFrame(results)

# Run Evaluation for 3 to 7 clusters
df_results = evaluate_clustering_methods(X_min_scaled, range(3, 8))

# Find Best Result
best_run = df_results.loc[df_results['Score'].idxmax()]

print("\n--- Best Clustering Configuration ---")
print(f"Method: {best_run['Method']}")
print(f"Clusters: {best_run['K']}")
print(f"Silhouette Score: {best_run['Score']:.4f}")

# Display Top 5 Results
print("\nTop 5 Configurations:")
print(df_results.sort_values('Score', ascending=False).head(5)[['Method', 'K', 'Score']])  


# 1. Retrieve Best Labels
minority_cluster_labels = best_run['Labels']

# 2. Visualize the Clusters (Optional, using PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_min_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=minority_cluster_labels, cmap='viridis', alpha=0.6)
plt.title(f"Class 1 Structure: {best_run['Method']} (K={best_run['K']})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster ID')
plt.show()

# 3. Add Cluster Labels to Original Minority Data
# We will use this to split them during oversampling
df_min_clustered = pd.DataFrame(X_min, columns=X.columns) # Assuming X is dataframe
df_min_clustered['cluster'] = minority_cluster_labels


# Initialize Lists to hold final data
X_final_list = [X_maj] # Start with original majority
y_final_list = [y_maj] # Start with original majority labels

# Calculate how many samples we want TOTAL (e.g., 1:1 ratio)
target_total_minority = len(X_maj) 
current_total_minority = len(X_min)
required_synthetic = target_total_minority - current_total_minority

# Calculate how many to generate PER CLUSTER
# We use proportional distribution (larger clusters get more synthetic samples)
# Or you can use Uniform (target_total / n_clusters) to boost rare types.
# Here we use Uniform to ensure representation of all sub-types.
n_clusters = best_run['K']
samples_per_cluster_target = int(target_total_minority / n_clusters)

print(f"Targeting ~{samples_per_cluster_target} total samples per cluster (Original + Synthetic)")

for cluster_id in range(n_clusters):
    # 1. Get Data for this Cluster
    cluster_data = df_min_clustered[df_min_clustered['cluster'] == cluster_id].drop('cluster', axis=1).values
    
    n_cluster_samples = len(cluster_data)
    
    # 2. Add Original Cluster Data to Final List
    X_final_list.append(cluster_data)
    y_final_list.append(np.ones(n_cluster_samples)) # Label 1
    
    # 3. Setup Oversampling for this Cluster
    # "Lee" needs Majority class to calculate boundary safety.
    # We construct a temporary dataset: (Original Majority) + (Current Cluster)
    X_temp = np.vstack([X_maj, cluster_data])
    y_temp = np.hstack([np.zeros(len(X_maj)), np.ones(len(cluster_data))])
    
    # Check if cluster is big enough for 'Lee' (needs neighbors)
    if n_cluster_samples < 5:
        print(f"Cluster {cluster_id} too small ({n_cluster_samples}). Using Random Oversampling.")
        # Fallback to simple duplication
        n_needed = samples_per_cluster_target - n_cluster_samples
        if n_needed > 0:
            indices = np.random.choice(n_cluster_samples, n_needed, replace=True)
            X_final_list.append(cluster_data[indices])
            y_final_list.append(np.ones(n_needed))
        continue

    # 4. Apply Lee Oversampling
    # We want to increase this cluster to 'samples_per_cluster_target'
    # Lee sampler usually balances 50/50 or strict ratio. 
    # We control it via 'proportion' roughly.
    try:
        oversampler = sv.Lee(random_state=42)
        
        # We sample, then extract ONLY the NEW synthetic points
        X_samp_temp, y_samp_temp = oversampler.sample(X_temp, y_temp)
        
        # Identify Synthetic Samples (Those that are not in X_temp)
        # Since Lee returns (Majority + Original Minority + Synthetic), 
        # and we know the count of (Majority + Original Minority)...
        n_original_temp = len(X_temp)
        X_synthetic = X_samp_temp[n_original_temp:]
        y_synthetic = y_samp_temp[n_original_temp:]
        
        # Determine how many we actually need to reach our target
        n_needed = samples_per_cluster_target - n_cluster_samples
        
        if len(X_synthetic) > 0 and n_needed > 0:
            # If Lee generated too many, sample down. If too few, take all.
            if len(X_synthetic) > n_needed:
                indices = np.random.choice(len(X_synthetic), n_needed, replace=False)
                X_synthetic = X_synthetic[indices]
                y_synthetic = y_synthetic[indices]
            
            X_final_list.append(X_synthetic)
            y_final_list.append(y_synthetic)
            print(f"Cluster {cluster_id}: Added {len(X_synthetic)} synthetic samples using Lee.")
            
    except Exception as e:
        print(f"Cluster {cluster_id} failed Lee sampling: {e}")


# 1. Concatenate everything
X_resampled = np.vstack(X_final_list)
y_resampled = np.hstack(y_final_list)

# 2. Shuffle (Important for training)
shuffler = np.random.permutation(len(X_resampled))
X_resampled = X_resampled[shuffler]
y_resampled = y_resampled[shuffler]

# 3. Final Check
print("\n--- Final Dataset Stats ---")
print(f"Original Class 1: {len(X_min)}")
print(f"Resampled Class 1: {sum(y_resampled == 1)}")
print(f"Class 0: {sum(y_resampled == 0)}")
print(f"Ratio: 1:{sum(y_resampled == 0) / sum(y_resampled == 1):.2f}")


import pandas as pd
import numpy as np

def get_cluster_deep_insights(df_original, cluster_labels):
    """
    Calculates detailed business metrics for each cluster of APPLIED customers.
    """
    # 1. Setup: Merge Clusters back to Raw Data
    df = df_original.copy()
    df['Cluster'] = cluster_labels
    
    # 2. Define Helper for Categorical Mode (Most common value)
    def get_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else "Unknown"
    
    # --- PRE-CALCULATIONS ---
    
    # A. Activity Flags (Robust to 1/0 or Y/N)
    # Checks if value is 'Y', 'y', 1, or > 0
    def is_active(x):
        return 1 if x in ['Y', 'y', 1, '1'] or (isinstance(x, (int, float)) and x > 0) else 0

    df['is_online'] = df['LTM_ONLINE_ACTIVE'].apply(is_active)
    df['is_email'] = df['LTM_EMAIL_ACTIVE'].apply(is_active)
    df['is_txn_active'] = df['LTM_TRAN_ACTIVE'].apply(is_active)
    
    # B. Approval Status Flag (Target Variable)
    # Assumes 'APPLICATION_STATUS' contains 'Approved' text
    df['is_approved'] = df['APPLICATION_STATUS'].astype(str).str.lower().str.contains('approved').astype(int)
    
    # 3. AGGREGATION
    summary = df.groupby('Cluster').agg({
        # 1. Size & Share
        'LYL_ID_NO': 'count',
        
        # 2. Financials (Mean)
        'LAST_12MTH_SPEND': 'mean',
        'TOTAL_EARNED_SINCE_2020': 'mean',
        'POINTS_BALANCE': 'mean',
        
        # 3. Demographics (Mode)
        'AGE_RANGE': get_mode,
        'INCOME_RANGE': get_mode, # or CENSUS_INCOME_RANGE
        
        # 4. Engagement (Percent Active)
        'is_online': 'mean',
        'is_email': 'mean',
        
        # 5. Success Metric (Approval Rate)
        'is_approved': 'mean'
    }).rename(columns={'LYL_ID_NO': 'Cluster_Size'})
    
    # 4. DERIVED METRICS & FORMATTING
    
    # A. Cluster Share (What % of total applicants fall into this cluster?)
    total_applicants = summary['Cluster_Size'].sum()
    summary['Share_of_Applicants'] = (summary['Cluster_Size'] / total_applicants) * 100
    
    # B. Format Percentages (0.5 -> 50.0)
    cols_to_pct = ['is_online', 'is_email', 'is_approved']
    for col in cols_to_pct:
        summary[col] = summary[col] * 100
        
    # C. Renaming for clarity
    summary = summary.rename(columns={
        'is_approved': 'Approval_Rate (%)',
        'is_online': 'Online_Active (%)',
        'is_email': 'Email_Active (%)',
        'LAST_12MTH_SPEND': 'Avg_Annual_Spend ($)',
        'POINTS_BALANCE': 'Avg_Points_Bal'
    })
    
    # D. Final Column Ordering
    final_view = summary[[
        'Cluster_Size', 
        'Share_of_Applicants',
        'Approval_Rate (%)',
        'Avg_Annual_Spend ($)', 
        'Avg_Points_Bal',
        'Online_Active (%)',
        'Email_Active (%)',
        'AGE_RANGE', 
        'INCOME_RANGE'
    ]].round(2).sort_values('Approval_Rate (%)', ascending=False)
    
    return final_view
