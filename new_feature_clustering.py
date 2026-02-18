# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from hdbscan import HDBSCAN
from sklearn_extra.cluster import KMedoids

# Preprocessing & metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)

# For Spectral Clustering
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

# Set display options
pd.set_option('display.max_columns', None)
np.random.seed(42)

###########################
# Load your preprocessed data
df = pd.read_parquet('your_data_path.parquet')

# Create target variable for validation
df['applied'] = df['APPLICATION_STATUS'].notna().astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Applied rate: {df['applied'].mean():.4f}")

############################3

def engineer_marketing_features_enhanced(df):
    """
    Enhanced feature engineering with recency and interactions
    """
    data = df.copy()
    
    # --- SAFETY: Handle negative values ---
    safe_earn = data['TOTAL_EARNED_SINCE_2020'].clip(lower=0)
    safe_spend_12m = data['LAST_12MTH_SPEND'].clip(lower=0)
    safe_spend_3m = data['LAST_3MTH_SPEND'].clip(lower=0)
    safe_spend_36m = data['LAST_36MTH_SPEND'].clip(lower=0)
    
    # --- 1. RECENCY FEATURES (CRITICAL) ---
    # Approximate days since last spend using spend presence
    data['flag_active_3m'] = (safe_spend_3m > 0).astype(int)
    data['flag_active_12m'] = (safe_spend_12m > 0).astype(int)
    data['flag_dormant'] = ((safe_spend_12m == 0) & (safe_spend_36m > 0)).astype(int)
    
    # --- 2. BEHAVIORAL RATIOS ---
    # Burn rate (Redeem/Earn)
    data['ratio_burn_rate'] = data['TOTAL_RDM_SINCE_2020'] / (safe_earn + 1)
    
    # Spend velocity (trend flag is more robust than ratio)
    data['flag_spend_increasing'] = ((safe_spend_3m * 4) > safe_spend_12m).astype(int)
    
    # Active burner (earns AND redeems)
    data['flag_active_burner'] = (
        (data['LTM_DLR_EARNED'] > 0) & 
        (data['LTM_RDM'] > 0)
    ).astype(int)
    
    # --- 3. ENGAGEMENT SCORES ---
    # Email engagement composite
    data['email_engagement_score'] = (
        data['LTM_EMAIL_ACTIVE'] * 0.5 + 
        data['L18M_EMAIL_ACTIVE'] * 0.3 + 
        data['L24M_EMAIL_ACTIVE'] * 0.2
    )
    
    # Digital engagement flag
    if data['LTM_ONLINE_ACTIVE'].dtype == 'object':
        data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE'].apply(
            lambda x: 1 if x == 'Y' else 0
        )
    else:
        data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE']
    
    # Omnichannel user
    data['flag_omnichannel'] = (
        (data['flag_digital_active'] == 1) & 
        (data['LTM_TRAN_ACTIVE'] > 0)
    ).astype(int)
    
    # --- 4. LOG TRANSFORMATIONS (Winsorized) ---
    from scipy.stats import mstats
    
    cols_to_winsorize = [
        'LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 
        'POINTS_BALANCE', 'LTM_DLR_EARNED'
    ]
    
    for col in cols_to_winsorize:
        # Winsorize at 95th percentile
        data[f'{col}_winsorized'] = mstats.winsorize(
            data[col], limits=[0, 0.05]
        )
        # Log transform
        data[f'log_{col}'] = np.log1p(
            data[f'{col}_winsorized'].clip(lower=0)
        )
    
    # --- 5. INTERACTION FEATURES ---
    # Engagement × Spend
    data['engage_spend_score'] = (
        data['MES_ENG_SCORE'] * np.log1p(safe_spend_12m)
    )
    
    # Spend to Income ratio
    data['spend_to_income_ratio'] = (
        safe_spend_12m / (data['INCOME_RANGE_ENC'] + 1)
    )
    
    # Points hoarder flag
    data['flag_high_points'] = (data['POINTS_BALANCE'] > 10000).astype(int)
    
    # --- 6. FINAL SAFETY ---
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    return data

# Apply feature engineering
df_engineered = engineer_marketing_features_enhanced(df)
print("Enhanced features created successfully")

############################


# MINIMAL OPTIMAL FEATURE SET (10 features max for imbalanced clustering)
optimal_features = [
    # BEHAVIORAL (6 features - highest signal)
    'flag_active_3m',              # Recency proxy
    'log_LAST_12MTH_SPEND',        # Monetary magnitude
    'flag_spend_increasing',       # Trend
    'ratio_burn_rate',             # Engagement type
    'email_engagement_score',      # Response proxy
    'flag_active_burner',          # Active vs passive
    
    # DEMOGRAPHIC (3 features - context)
    'AGE_RANGE_ENC',
    'INCOME_RANGE_ENC',
    'D_MONTHONBOOK',               # Tenure
    
    # ADVANCED (1 feature - interaction)
    'engage_spend_score'
]

# Create clustering dataset
X_cluster = df_engineered[optimal_features].copy()

print(f"Clustering features: {len(optimal_features)}")
print(f"Dataset shape: {X_cluster.shape}")
print("\nFeature statistics:")
print(X_cluster.describe())

##########################################3

def prepare_data_for_clustering(X, scaler_type='robust'):
    """
    Prepares data with:
    1. Outlier clipping (99th percentile)
    2. Scaling (Robust or Power Transform)
    3. Returns scaled data
    """
    X_prep = X.copy()
    
    # 1. Clip extreme outliers at 99th percentile
    for col in X_prep.columns:
        upper_limit = X_prep[col].quantile(0.99)
        X_prep[col] = X_prep[col].clip(upper=upper_limit)
    
    # 2. Fill any remaining NaN/inf
    X_prep = X_prep.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Scale based on type
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X_prep)
    
    return X_scaled, scaler

# Prepare data
X_scaled, scaler = prepare_data_for_clustering(X_cluster, scaler_type='power')

print("Data prepared for clustering")
print(f"Scaled shape: {X_scaled.shape}")

###############################################
def run_advanced_clustering_suite(X_scaled, n_clusters=5):
    """
    Runs multiple advanced clustering algorithms
    Returns: Dictionary of {algorithm_name: labels}
    """
    results = {}
    
    print("Running Advanced Clustering Suite...\n")
    
    # 1. KMeans (baseline)
    print("1. K-Means...")
    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=20, 
        max_iter=500, 
        random_state=42
    )
    results['KMeans'] = kmeans.fit_predict(X_scaled)
    
    # 2. Gaussian Mixture Model (GMM)
    print("2. GMM...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        max_iter=200,
        n_init=10,
        random_state=42
    )
    results['GMM'] = gmm.fit_predict(X_scaled)
    
    # 3. Bayesian GMM (handles uncertainty better)
    print("3. Bayesian GMM...")
    bgmm = BayesianGaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        max_iter=200,
        n_init=5,
        random_state=42
    )
    results['BGMM'] = bgmm.fit_predict(X_scaled)
    
    # 4. Spectral Clustering (graph-based)
    print("4. Spectral Clustering...")
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=15,
        assign_labels='kmeans',
        random_state=42
    )
    results['Spectral'] = spectral.fit_predict(X_scaled)
    
    # 5. HDBSCAN (density-based, auto clusters)
    print("5. HDBSCAN...")
    hdbscan = HDBSCAN(
        min_cluster_size=int(len(X_scaled) * 0.01),  # 1% of data
        min_samples=10,
        metric='euclidean',
        cluster_selection_epsilon=0.5
    )
    results['HDBSCAN'] = hdbscan.fit_predict(X_scaled)
    
    # 6. OPTICS (density-based, hierarchical)
    print("6. OPTICS...")
    optics = OPTICS(
        min_samples=50,
        xi=0.05,
        min_cluster_size=0.02
    )
    results['OPTICS'] = optics.fit_predict(X_scaled)
    
    # 7. Agglomerative (hierarchical)
    print("7. Agglomerative...")
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    results['Agglomerative'] = agg.fit_predict(X_scaled)
    
    # 8. K-Medoids (more robust to outliers than K-Means)
    print("8. K-Medoids...")
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric='euclidean',
        init='k-medoids++',
        max_iter=300,
        random_state=42
    )
    results['KMedoids'] = kmedoids.fit_predict(X_scaled) 

  # 5. Agglomerative Clustering (Hierarchical)
    print("5. Agglomerative Clustering...")
    # 'ward' linkage minimizes the variance of the clusters being merged.
    # It usually leads to the most balanced cluster sizes.
    agglomerative = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward', 
        metric='euclidean'
    )
    results['Agglomerative'] = agglomerative.fit_predict(X_scaled)
    
    print("\nAll algorithms completed!")
    return results

# Run all clustering algorithms
clustering_results = run_advanced_clustering_suite(X_scaled, n_clusters=5)

###########################

# Team's feature set for Calinski-Harabasz score
cols_check = [
    'LAST_3MTH_SPEND', 'LAST_12MTH_SPEND', 'LAST_36MTH_SPEND', 
    'LTM_RDM', 'LTM_DLR_EARNED', 'POINTS_BALANCE',
    'MES_ENG_SCORE', 'LTM_ONLINE_ACTIVE', 'L24M_EMAIL_ACTIVE', 
    'TOTAL_RDM_SINCE_2020', 'AGE_RANGE_ENC',
    'CENSUS_INCOME_RANGE_ENC', 'L6M_REDEEMER', 
    'LTM_TRAN_ACTIVE', 'INCOME_RANGE_ENC', 'RFM_SEGMENT_ENC'
]

# Prepare validation features
X_validation = df_engineered[cols_check].copy()

# Handle LTM_ONLINE_ACTIVE if it's categorical
if X_validation['LTM_ONLINE_ACTIVE'].dtype == 'object':
    X_validation['LTM_ONLINE_ACTIVE'] = X_validation['LTM_ONLINE_ACTIVE'].apply(
        lambda x: 1 if x == 'Y' else 0
    )

# Scale validation features
X_val_scaled, _ = prepare_data_for_clustering(X_validation, scaler_type='power')

def calculate_clustering_metrics(X_val_scaled, labels, algorithm_name):
    """
    Calculate comprehensive metrics on TEAM'S feature set
    """
    # Filter out noise points (label = -1 for HDBSCAN/OPTICS)
    valid_mask = labels >= 0
    X_filtered = X_val_scaled[valid_mask]
    labels_filtered = labels[valid_mask]
    
    n_clusters = len(np.unique(labels_filtered))
    
    # Skip if only 1 cluster or all noise
    if n_clusters < 2:
        return {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters,
            'calinski_harabasz': 0,
            'silhouette': 0,
            'davies_bouldin': 999,
            'note': 'Invalid clustering (< 2 clusters)'
        }
    
    metrics = {
        'algorithm': algorithm_name,
        'n_clusters': n_clusters,
        'calinski_harabasz': calinski_harabasz_score(X_filtered, labels_filtered),
        'silhouette': silhouette_score(X_filtered, labels_filtered),
        'davies_bouldin': davies_bouldin_score(X_filtered, labels_filtered),
        'noise_points': np.sum(labels == -1),
        'noise_pct': np.mean(labels == -1) * 100
    }
    
    return metrics

# Calculate metrics for all algorithms
metrics_results = []
for algo_name, labels in clustering_results.items():
    metrics = calculate_clustering_metrics(X_val_scaled, labels, algo_name)
    metrics_results.append(metrics)

# Create results DataFrame
metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.sort_values('calinski_harabasz', ascending=False)

print("\n" + "="*80)
print("CLUSTERING PERFORMANCE METRICS (On Team's Feature Set)")
print("="*80)
print(metrics_df.to_string(index=False))
print("\nBest Algorithm (by Calinski-Harabasz):", metrics_df.iloc[0]['algorithm'])

#####################################

def analyze_cluster_business_metrics(df_original, labels, algorithm_name):
    """
    Analyzes cluster quality based on business metrics
    """
    df_analysis = df_original.copy()
    
    # Filter out noise
    valid_mask = labels >= 0
    df_analysis = df_analysis[valid_mask].copy()
    labels_filtered = labels[valid_mask]
    
    df_analysis['Cluster'] = labels_filtered
    
    # Calculate metrics per cluster
    summary = df_analysis.groupby('Cluster').agg({
        'applied': ['count', 'sum', 'mean'],
        'LAST_12MTH_SPEND': 'mean',
        'POINTS_BALANCE': 'mean',
        'MES_ENG_SCORE': 'mean'
    }).round(4)
    
    summary.columns = [
        'Size', 'Applied_Count', 'Applied_Rate',
        'Avg_Spend_12M', 'Avg_Points', 'Avg_Engagement'
    ]
    
    # Calculate lift (how much better than baseline)
    baseline_rate = df_analysis['applied'].mean()
    summary['Lift'] = (summary['Applied_Rate'] / baseline_rate).round(2)
    
    # Sort by applied rate descending
    summary = summary.sort_values('Applied_Rate', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"{algorithm_name} - BUSINESS METRICS BY CLUSTER")
    print(f"{'='*80}")
    print(summary)
    print(f"\nBaseline Applied Rate: {baseline_rate:.4f}")
    print(f"Best Cluster Lift: {summary['Lift'].max():.2f}x")
    
    return summary

# Analyze top 3 algorithms
top_algos = metrics_df.head(3)['algorithm'].tolist()

business_results = {}
for algo in top_algos:
    labels = clustering_results[algo]
    summary = analyze_cluster_business_metrics(df_engineered, labels, algo)
    business_results[algo] = summary

######################################

def validate_cluster_balance(labels, algorithm_name):
    """
    Validates cluster size distribution
    Alerts if micro-clusters (<1%) exist
    """
    valid_labels = labels[labels >= 0]
    unique, counts = np.unique(valid_labels, return_counts=True)
    
    total = len(valid_labels)
    distribution = pd.DataFrame({
        'Cluster': unique,
        'Count': counts,
        'Percentage': (counts / total * 100).round(2)
    }).sort_values('Count', ascending=False)
    
    print(f"\n{algorithm_name} - Cluster Size Distribution:")
    print(distribution.to_string(index=False))
    
    # Alert if any cluster < 1%
    if (distribution['Percentage'] < 1.0).any():
        print("\n⚠ WARNING: Micro-clusters detected (<1% of data)")
    
    return distribution

# Check balance for top algorithms
for algo in top_algos:
    validate_cluster_balance(clustering_results[algo], algo)

#######################################

from scipy.stats import mode

def create_ensemble_clusters(clustering_results, algorithms_to_ensemble, n_final_clusters=5):
    """
    Creates consensus clustering from multiple algorithms
    Uses majority voting
    """
    print("Creating Ensemble Clustering...")
    
    # Stack labels from selected algorithms
    label_matrix = np.column_stack([
        clustering_results[algo] for algo in algorithms_to_ensemble
    ])
    
    # For each sample, find the most common cluster assignment
    # This is a simplified approach - use with caution
    # Better: Use co-association matrix approach
    
    # Re-cluster the label matrix itself
    from sklearn.cluster import KMeans
    
    # Treat each algorithm's labels as features
    # Then cluster THOSE patterns
    ensemble_kmeans = KMeans(
        n_clusters=n_final_clusters,
        n_init=20,
        random_state=42
    )
    
    ensemble_labels = ensemble_kmeans.fit_predict(label_matrix)
    
    print(f"Ensemble created from: {algorithms_to_ensemble}")
    
    return ensemble_labels

# Create ensemble from top 3 performing algorithms
ensemble_labels = create_ensemble_clusters(
    clustering_results,
    algorithms_to_ensemble=top_algos,
    n_final_clusters=5
)

# Evaluate ensemble
ensemble_metrics = calculate_clustering_metrics(
    X_val_scaled, 
    ensemble_labels, 
    'Ensemble'
)

print("\nEnsemble Metrics:")
print(pd.DataFrame([ensemble_metrics]))

# Business metrics for ensemble
ensemble_business = analyze_cluster_business_metrics(
    df_engineered, 
    ensemble_labels, 
    'Ensemble'
)

#######################################

# Select best performing algorithm
best_algo = metrics_df.iloc[0]['algorithm']
best_labels = clustering_results[best_algo]

# Add to original dataframe
df_final = df_engineered.copy()
df_final['Cluster'] = best_labels
df_final['Cluster_Algorithm'] = best_algo

# Export
output_path = '/mnt/user-data/outputs/clustered_marketing_data.parquet'
df_final.to_parquet(output_path, index=False)

print(f"\nBest clustering algorithm: {best_algo}")
print(f"Results exported to: {output_path}")
print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Clusters created: {len(np.unique(best_labels[best_labels >= 0]))}")
