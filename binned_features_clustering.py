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
from sklearn.cluster import SpectralClustering

# Preprocessing & metrics
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)

# Set display options
pd.set_option('display.max_columns', None)
np.random.seed(42)

print("Libraries imported successfully")

################################
# Load your data
df = pd.read_parquet('your_data_path.parquet')

# Create target variable
df['applied'] = df['APPLICATION_STATUS'].notna().astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Applied rate: {df['applied'].mean():.4f}")
print(f"Applied count: {df['applied'].sum()}")

######################################


# Load your data
df = pd.read_parquet('your_data_path.parquet')

# Create target variable
df['applied'] = df['APPLICATION_STATUS'].notna().astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Applied rate: {df['applied'].mean():.4f}")
print(f"Applied count: {df['applied'].sum()}")

######################################

def create_binned_features(df):
    """
    Creates binned versions of continuous features
    Based on your image analysis showing heavy concentration in low bins
    """
    data = df.copy()
    
    # --- 1. SPEND BINNING (Based on your distribution images) ---
    
    # LAST_12MTH_SPEND bins (Image shows 0-5000 has majority)
    data['LAST_12MTH_SPEND_BIN'] = pd.cut(
        data['LAST_12MTH_SPEND'],
        bins=[0, 100, 500, 1000, 2000, 5000, 10000, np.inf],
        labels=['0-100', '100-500', '500-1K', '1K-2K', '2K-5K', '5K-10K', '10K+'],
        include_lowest=True
    )
    
    # LAST_3MTH_SPEND bins
    data['LAST_3MTH_SPEND_BIN'] = pd.cut(
        data['LAST_3MTH_SPEND'],
        bins=[0, 50, 200, 500, 1000, 2000, np.inf],
        labels=['0-50', '50-200', '200-500', '500-1K', '1K-2K', '2K+'],
        include_lowest=True
    )
    
    # LAST_36MTH_SPEND bins
    data['LAST_36MTH_SPEND_BIN'] = pd.cut(
        data['LAST_36MTH_SPEND'],
        bins=[0, 500, 2000, 5000, 10000, 20000, np.inf],
        labels=['0-500', '500-2K', '2K-5K', '5K-10K', '10K-20K', '20K+'],
        include_lowest=True
    )
    
    # --- 2. POINTS BALANCE BINNING ---
    data['POINTS_BALANCE_BIN'] = pd.cut(
        data['POINTS_BALANCE'],
        bins=[0, 100, 500, 1000, 5000, 10000, np.inf],
        labels=['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K+'],
        include_lowest=True
    )
    
    # --- 3. EARNED/REDEEMED BINNING ---
    data['LTM_DLR_EARNED_BIN'] = pd.cut(
        data['LTM_DLR_EARNED'],
        bins=[0, 50, 200, 500, 1000, 2000, np.inf],
        labels=['0-50', '50-200', '200-500', '500-1K', '1K-2K', '2K+'],
        include_lowest=True
    )
    
    data['LTM_RDM_BIN'] = pd.cut(
        data['LTM_RDM'],
        bins=[0, 50, 200, 500, 1000, 2000, np.inf],
        labels=['0-50', '50-200', '200-500', '500-1K', '1K-2K', '2K+'],
        include_lowest=True
    )
    
    data['TOTAL_RDM_SINCE_2020_BIN'] = pd.cut(
        data['TOTAL_RDM_SINCE_2020'],
        bins=[0, 200, 1000, 3000, 5000, 10000, np.inf],
        labels=['0-200', '200-1K', '1K-3K', '3K-5K', '5K-10K', '10K+'],
        include_lowest=True
    )
    
    # --- 4. ENGAGEMENT SCORE BINNING ---
    data['MES_ENG_SCORE_BIN'] = pd.cut(
        data['MES_ENG_SCORE'],
        bins=[0, 30, 60, 90, 120, 150, np.inf],
        labels=['0-30', '30-60', '60-90', '90-120', '120-150', '150+'],
        include_lowest=True
    )
    
    # --- 5. TENURE BINNING (MONTHONBOOK) ---
    data['TENURE_BIN'] = pd.cut(
        data['D_MONTHONBOOK'],
        bins=[0, 6, 12, 24, 36, 60, np.inf],
        labels=['0-6m', '6-12m', '12-24m', '24-36m', '36-60m', '60m+'],
        include_lowest=True
    )
    
    # --- 6. INCOME BINNING (Already encoded, create ranges) ---
    # Assuming INCOME_RANGE_ENC is 0-10 scale
    data['INCOME_BIN'] = pd.cut(
        data['INCOME_RANGE_ENC'],
        bins=[-1, 2, 4, 6, 8, np.inf],
        labels=['Low', 'Low-Mid', 'Mid', 'Mid-High', 'High'],
        include_lowest=True
    )
    
    # --- 7. RFM SEGMENT BINNING ---
    # Assuming RFM_SEGMENT_ENC is numeric
    data['RFM_BIN'] = pd.cut(
        data['RFM_SEGMENT_ENC'],
        bins=[-1, 3, 6, 9, np.inf],
        labels=['Low-Value', 'Mid-Value', 'High-Value', 'Premium'],
        include_lowest=True
    )
    
    return data

# Create binned features
df_binned = create_binned_features(df)

print("\nBinned features created:")
binned_cols = [col for col in df_binned.columns if '_BIN' in col]
for col in binned_cols:
    print(f"  - {col}: {df_binned[col].nunique()} bins")


###############################################

def engineer_features_from_bins(df):
    """
    Creates clustering features from binned data
    EXCLUDES: Age and Location features as per requirement
    """
    data = df.copy()
    
    # --- Convert categorical bins to ordinal encoding ---
    from sklearn.preprocessing import OrdinalEncoder
    
    binned_features = [col for col in data.columns if '_BIN' in col]
    
    # Encode bins to numeric
    for col in binned_features:
        # Fill NaN with 'Missing' category first
        data[col] = data[col].fillna('Missing').astype(str)
        
        # Get unique categories
        categories = sorted(data[col].unique())
        
        # Create ordinal encoding
        encoding_map = {cat: idx for idx, cat in enumerate(categories)}
        data[f'{col}_ENCODED'] = data[col].map(encoding_map)
    
    # --- Binary Activity Flags ---
    data['flag_active_3m'] = (data['LAST_3MTH_SPEND'] > 0).astype(int)
    data['flag_active_12m'] = (data['LAST_12MTH_SPEND'] > 0).astype(int)
    data['flag_earner'] = (data['LTM_DLR_EARNED'] > 0).astype(int)
    data['flag_redeemer'] = (data['LTM_RDM'] > 0).astype(int)
    
    # Email engagement
    if data['LTM_EMAIL_ACTIVE'].dtype == 'object':
        data['flag_email_active'] = data['LTM_EMAIL_ACTIVE'].apply(
            lambda x: 1 if x in ['Y', 1, True] else 0
        )
    else:
        data['flag_email_active'] = data['LTM_EMAIL_ACTIVE']
    
    # Online engagement
    if data['LTM_ONLINE_ACTIVE'].dtype == 'object':
        data['flag_online_active'] = data['LTM_ONLINE_ACTIVE'].apply(
            lambda x: 1 if x in ['Y', 1, True] else 0
        )
    else:
        data['flag_online_active'] = data['LTM_ONLINE_ACTIVE']
    
    # --- Behavioral Ratios (calculated from original, not binned) ---
    safe_earn = data['TOTAL_EARNED_SINCE_2020'].clip(lower=0)
    safe_spend_12m = data['LAST_12MTH_SPEND'].clip(lower=0)
    safe_spend_3m = data['LAST_3MTH_SPEND'].clip(lower=0)
    
    # Burn rate
    data['ratio_burn_rate'] = data['TOTAL_RDM_SINCE_2020'] / (safe_earn + 1)
    
    # Spend acceleration
    data['ratio_spend_velocity'] = (safe_spend_3m * 4) / (safe_spend_12m + 1)
    
    # Active burner
    data['flag_active_burner'] = (
        (data['LTM_DLR_EARNED'] > 0) & 
        (data['LTM_RDM'] > 0)
    ).astype(int)
    
    # --- Clean up ---
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    return data

# Apply feature engineering
df_features = engineer_features_from_bins(df_binned)
print("Feature engineering completed")

####################################3

# EXPERIMENT 1: Pure Binned Features (Categorical bins encoded)
binned_encoded_features = [
    # SPENDING BINS
    'LAST_12MTH_SPEND_BIN_ENCODED',
    'LAST_3MTH_SPEND_BIN_ENCODED',
    'LAST_36MTH_SPEND_BIN_ENCODED',
    
    # EARNING/REDEEMING BINS
    'LTM_DLR_EARNED_BIN_ENCODED',
    'LTM_RDM_BIN_ENCODED',
    'TOTAL_RDM_SINCE_2020_BIN_ENCODED',
    
    # ENGAGEMENT BINS
    'MES_ENG_SCORE_BIN_ENCODED',
    'POINTS_BALANCE_BIN_ENCODED',
    
    # TENURE BIN
    'TENURE_BIN_ENCODED',
    
    # RFM BIN
    'RFM_BIN_ENCODED',
    
    # INCOME BIN (NOT age, this is spending capacity)
    'INCOME_BIN_ENCODED'
]

# EXPERIMENT 2: Hybrid (Bins + Behavioral Flags + Ratios)
hybrid_features = [
    # BINNED FEATURES
    'LAST_12MTH_SPEND_BIN_ENCODED',
    'LAST_3MTH_SPEND_BIN_ENCODED',
    'LTM_RDM_BIN_ENCODED',
    'MES_ENG_SCORE_BIN_ENCODED',
    'POINTS_BALANCE_BIN_ENCODED',
    'TENURE_BIN_ENCODED',
    
    # BEHAVIORAL FLAGS
    'flag_active_3m',
    'flag_active_12m',
    'flag_earner',
    'flag_redeemer',
    'flag_email_active',
    'flag_online_active',
    'flag_active_burner',
    
    # RATIOS
    'ratio_burn_rate',
    'ratio_spend_velocity'
]

# EXPERIMENT 3: Minimal High-Signal Bins Only
minimal_binned_features = [
    'LAST_12MTH_SPEND_BIN_ENCODED',     # Primary spending
    'LTM_RDM_BIN_ENCODED',               # Redemption behavior
    'MES_ENG_SCORE_BIN_ENCODED',         # Engagement
    'TENURE_BIN_ENCODED',                 # Lifecycle
    'flag_active_3m',                     # Recency
    'flag_active_burner',                 # Behavior type
    'ratio_burn_rate'                     # Burn pattern
]

print("Feature sets defined:")
print(f"\n1. Pure Binned: {len(binned_encoded_features)} features")
print(f"2. Hybrid: {len(hybrid_features)} features")
print(f"3. Minimal: {len(minimal_binned_features)} features")

# Verify NO age/location features included
age_location_keywords = ['AGE', 'STATE', 'CITY', 'ZIP', 'GEOGRAPHIC', 'PRIMARY_STATE']
for feature_set in [binned_encoded_features, hybrid_features, minimal_binned_features]:
    violations = [f for f in feature_set if any(kw in f.upper() for kw in age_location_keywords)]
    if violations:
        print(f"\nWARNING: Age/Location features detected: {violations}")
    else:
        print(f"✓ No age/location features detected")

################################################33333

def prepare_binned_data_for_clustering(df, feature_list, scaler_type='robust'):
    """
    Prepares binned feature data for clustering
    """
    # Extract features
    X = df[feature_list].copy()
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    # Clip outliers at 99th percentile (for ratio features)
    for col in X.columns:
        if X[col].max() > 100:  # Only clip if large values exist
            upper_limit = X[col].quantile(0.99)
            X[col] = X[col].clip(upper=upper_limit)
    
    # Scale
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = PowerTransformer(method='yeo-johnson')
    
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Prepare all three experiment datasets
print("Preparing datasets...\n")

X_binned, scaler_binned = prepare_binned_data_for_clustering(
    df_features, binned_encoded_features, scaler_type='robust'
)
print(f"Experiment 1 (Pure Binned): {X_binned.shape}")

X_hybrid, scaler_hybrid = prepare_binned_data_for_clustering(
    df_features, hybrid_features, scaler_type='robust'
)
print(f"Experiment 2 (Hybrid): {X_hybrid.shape}")

X_minimal, scaler_minimal = prepare_binned_data_for_clustering(
    df_features, minimal_binned_features, scaler_type='robust'
)
print(f"Experiment 3 (Minimal): {X_minimal.shape}")


############################################################3

def run_clustering_experiment(X_scaled, experiment_name, n_clusters_range=[4, 5, 6, 7]):
    """
    Runs multiple clustering algorithms with different n_clusters
    Returns best configuration
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}\n")
    
    algorithms = {
        'KMeans': lambda n: KMeans(n_clusters=n, n_init=20, max_iter=500, random_state=42),
        'GMM': lambda n: GaussianMixture(n_components=n, covariance_type='full', max_iter=200, n_init=10, random_state=42),
        'BGMM': lambda n: BayesianGaussianMixture(n_components=n, covariance_type='full', max_iter=200, n_init=5, random_state=42),
        'KMedoids': lambda n: KMedoids(n_clusters=n, metric='euclidean', init='k-medoids++', max_iter=300, random_state=42),
        'Agglomerative': lambda n: AgglomerativeClustering(n_clusters=n, linkage='ward')
    }
    
    for algo_name, algo_func in algorithms.items():
        for n_clusters in n_clusters_range:
            print(f"Running {algo_name} with {n_clusters} clusters...")
            
            # Fit
            model = algo_func(n_clusters)
            labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                ch_score = calinski_harabasz_score(X_scaled, labels)
                sil_score = silhouette_score(X_scaled, labels)
                db_score = davies_bouldin_score(X_scaled, labels)
            else:
                ch_score, sil_score, db_score = 0, 0, 999
            
            results.append({
                'Algorithm': algo_name,
                'N_Clusters': n_clusters,
                'Calinski_Harabasz': ch_score,
                'Silhouette': sil_score,
                'Davies_Bouldin': db_score,
                'Labels': labels
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'Labels'} 
        for r in results
    ])
    results_df = results_df.sort_values('Calinski_Harabasz', ascending=False)
    
    print(f"\nTop 5 Configurations:")
    print(results_df.head().to_string(index=False))
    
    # Get best configuration
    best_idx = results_df.index[0]
    best_config = results[best_idx]
    
    return results_df, best_config, results

# Run all experiments
exp1_results, exp1_best, exp1_all = run_clustering_experiment(
    X_binned, 
    "Pure Binned Features", 
    n_clusters_range=[4, 5, 6, 7]
)

exp2_results, exp2_best, exp2_all = run_clustering_experiment(
    X_hybrid, 
    "Hybrid (Bins + Flags + Ratios)", 
    n_clusters_range=[4, 5, 6, 7]
)

exp3_results, exp3_best, exp3_all = run_clustering_experiment(
    X_minimal, 
    "Minimal High-Signal Bins", 
    n_clusters_range=[4, 5, 6, 7]
)


#####################################################

# Team's validation features (EXCLUDING age/location)
cols_check_no_age_location = [
    'LAST_3MTH_SPEND', 'LAST_12MTH_SPEND', 'LAST_36MTH_SPEND', 
    'LTM_RDM', 'LTM_DLR_EARNED', 'POINTS_BALANCE',
    'MES_ENG_SCORE', 'LTM_ONLINE_ACTIVE', 'L24M_EMAIL_ACTIVE', 
    'TOTAL_RDM_SINCE_2020', 
    'CENSUS_INCOME_RANGE_ENC',  # Keep income (spending capacity)
    'L6M_REDEEMER', 'LTM_TRAN_ACTIVE', 
    'INCOME_RANGE_ENC', 'RFM_SEGMENT_ENC'
]

# Prepare validation features
X_validation = df_features[cols_check_no_age_location].copy()

# Handle categorical
if X_validation['LTM_ONLINE_ACTIVE'].dtype == 'object':
    X_validation['LTM_ONLINE_ACTIVE'] = X_validation['LTM_ONLINE_ACTIVE'].apply(
        lambda x: 1 if x in ['Y', 1, True] else 0
    )

# Scale
X_val_scaled, _ = prepare_binned_data_for_clustering(
    X_validation, 
    cols_check_no_age_location, 
    scaler_type='power'
)

def validate_on_team_features(labels, X_val_scaled, config_name):
    """
    Validates clustering on team's feature set
    """
    valid_mask = labels >= 0
    X_filtered = X_val_scaled[valid_mask]
    labels_filtered = labels[valid_mask]
    
    if len(np.unique(labels_filtered)) < 2:
        return None
    
    ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
    sil_score = silhouette_score(X_filtered, labels_filtered)
    db_score = davies_bouldin_score(X_filtered, labels_filtered)
    
    return {
        'Config': config_name,
        'Calinski_Harabasz': ch_score,
        'Silhouette': sil_score,
        'Davies_Bouldin': db_score
    }

# Validate best configurations
validation_results = []

val1 = validate_on_team_features(
    exp1_best['Labels'], 
    X_val_scaled, 
    f"Exp1: {exp1_best['Algorithm']}-{exp1_best['N_Clusters']}"
)
if val1: validation_results.append(val1)

val2 = validate_on_team_features(
    exp2_best['Labels'], 
    X_val_scaled, 
    f"Exp2: {exp2_best['Algorithm']}-{exp2_best['N_Clusters']}"
)
if val2: validation_results.append(val2)

val3 = validate_on_team_features(
    exp3_best['Labels'], 
    X_val_scaled, 
    f"Exp3: {exp3_best['Algorithm']}-{exp3_best['N_Clusters']}"
)
if val3: validation_results.append(val3)

print("\n" + "="*80)
print("VALIDATION ON TEAM'S FEATURE SET (No Age/Location)")
print("="*80)
val_df = pd.DataFrame(validation_results)
print(val_df.to_string(index=False))


##############################################


def analyze_binned_cluster_business_metrics(df, labels, config_name):
    """
    Analyzes business performance by cluster
    """
    df_analysis = df.copy()
    
    valid_mask = labels >= 0
    df_analysis = df_analysis[valid_mask].copy()
    labels_filtered = labels[valid_mask]
    
    df_analysis['Cluster'] = labels_filtered
    
    # Business metrics
    summary = df_analysis.groupby('Cluster').agg({
        'applied': ['count', 'sum', 'mean'],
        'LAST_12MTH_SPEND': 'mean',
        'LTM_RDM': 'mean',
        'POINTS_BALANCE': 'mean',
        'MES_ENG_SCORE': 'mean',
        'LTM_EMAIL_ACTIVE': 'mean',
        'LTM_TRAN_ACTIVE': 'mean'
    }).round(4)
    
    summary.columns = [
        'Size', 'Applied_Count', 'Applied_Rate',
        'Avg_Spend_12M', 'Avg_Redeem', 'Avg_Points', 
        'Avg_Engagement', 'Avg_Email_Active', 'Avg_Transactions'
    ]
    
    baseline_rate = df_analysis['applied'].mean()
    summary['Lift'] = (summary['Applied_Rate'] / baseline_rate).round(2)
    
    summary = summary.sort_values('Applied_Rate', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"{config_name} - BUSINESS METRICS")
    print(f"{'='*80}")
    print(summary.to_string())
    print(f"\nBaseline Applied Rate: {baseline_rate:.4f}")
    print(f"Best Cluster Applied Rate: {summary['Applied_Rate'].max():.4f}")
    print(f"Best Cluster Lift: {summary['Lift'].max():.2f}x")
    print(f"Best Cluster Size: {summary.iloc[0]['Size']:.0f} ({summary.iloc[0]['Size']/len(df_analysis)*100:.1f}%)")
    
    return summary

# Analyze all experiments
bus1 = analyze_binned_cluster_business_metrics(
    df_features, 
    exp1_best['Labels'], 
    f"Exp1: {exp1_best['Algorithm']}-{exp1_best['N_Clusters']}"
)

bus2 = analyze_binned_cluster_business_metrics(
    df_features, 
    exp2_best['Labels'], 
    f"Exp2: {exp2_best['Algorithm']}-{exp2_best['N_Clusters']}"
)

bus3 = analyze_binned_cluster_business_metrics(
    df_features, 
    exp3_best['Labels'], 
    f"Exp3: {exp3_best['Algorithm']}-{exp3_best['N_Clusters']}"
)


#########################################################3333

def profile_clusters_with_bins(df, labels, config_name):
    """
    Profiles clusters using the binned features
    Shows dominant bin categories per cluster
    """
    df_profile = df.copy()
    
    valid_mask = labels >= 0
    df_profile = df_profile[valid_mask].copy()
    labels_filtered = labels[valid_mask]
    
    df_profile['Cluster'] = labels_filtered
    
    # Get all bin columns (categorical, not encoded)
    bin_cols = [col for col in df_profile.columns if '_BIN' in col and '_ENCODED' not in col]
    
    print(f"\n{'='*80}")
    print(f"{config_name} - CLUSTER PROFILES (Dominant Bins)")
    print(f"{'='*80}\n")
    
    for cluster_id in sorted(df_profile['Cluster'].unique()):
        cluster_data = df_profile[df_profile['Cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        print(f"Cluster {cluster_id} (n={cluster_size}, {cluster_size/len(df_profile)*100:.1f}%):")
        print("-" * 60)
        
        for col in bin_cols[:5]:  # Show top 5 bin features
            mode_value = cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else 'N/A'
            mode_pct = (cluster_data[col] == mode_value).mean() * 100
            print(f"  {col:30s}: {mode_value:15s} ({mode_pct:.1f}%)")
        
        # Business metrics
        applied_rate = cluster_data['applied'].mean()
        avg_spend = cluster_data['LAST_12MTH_SPEND'].mean()
        print(f"  {'Applied Rate':30s}: {applied_rate:.4f}")
        print(f"  {'Avg 12M Spend':30s}: ${avg_spend:.2f}")
        print()

# Profile all experiments
profile_clusters_with_bins(
    df_features, 
    exp1_best['Labels'], 
    f"Exp1: {exp1_best['Algorithm']}-{exp1_best['N_Clusters']}"
)

profile_clusters_with_bins(
    df_features, 
    exp2_best['Labels'], 
    f"Exp2: {exp2_best['Algorithm']}-{exp2_best['N_Clusters']}"
)

profile_clusters_with_bins(
    df_features, 
    exp3_best['Labels'], 
    f"Exp3: {exp3_best['Algorithm']}-{exp3_best['N_Clusters']}"
)


#############################################

def visualize_cluster_distributions(labels_dict, df):
    """
    Visualizes cluster size and applied rate distributions
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Binned Clustering Experiments - Comparison', fontsize=16)
    
    experiments = [
        ('Exp1: Pure Binned', exp1_best['Labels']),
        ('Exp2: Hybrid', exp2_best['Labels']),
        ('Exp3: Minimal', exp3_best['Labels'])
    ]
    
    for idx, (exp_name, labels) in enumerate(experiments):
        # Filter valid labels
        valid_mask = labels >= 0
        labels_filtered = labels[valid_mask]
        df_filtered = df[valid_mask].copy()
        df_filtered['Cluster'] = labels_filtered
        
        # Cluster sizes
        ax1 = axes[0, idx]
        cluster_sizes = df_filtered['Cluster'].value_counts().sort_index()
        cluster_sizes.plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title(f'{exp_name}\nCluster Sizes')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Count')
        
        # Applied rates
        ax2 = axes[1, idx]
        applied_rates = df_filtered.groupby('Cluster')['applied'].mean().sort_index()
        applied_rates.plot(kind='bar', ax=ax2, color='coral')
        ax2.axhline(y=df['applied'].mean(), color='red', linestyle='--', label='Baseline')
        ax2.set_title(f'{exp_name}\nApplied Rates by Cluster')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Applied Rate')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/binned_clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to /mnt/user-data/outputs/binned_clustering_comparison.png")

# Create visualization
visualize_cluster_distributions(
    {
        'Exp1': exp1_best['Labels'],
        'Exp2': exp2_best['Labels'],
        'Exp3': exp3_best['Labels']
    },
    df_features
)


###########################################3

def generate_experiment_summary():
    """
    Generates comprehensive summary of all experiments
    """
    summary_data = []
    
    for exp_name, best_config, bus_metrics in [
        ('Exp1: Pure Binned', exp1_best, bus1),
        ('Exp2: Hybrid', exp2_best, bus2),
        ('Exp3: Minimal', exp3_best, bus3)
    ]:
        summary_data.append({
            'Experiment': exp_name,
            'Algorithm': best_config['Algorithm'],
            'N_Clusters': best_config['N_Clusters'],
            'Calinski_Harabasz': best_config['Calinski_Harabasz'],
            'Silhouette': best_config['Silhouette'],
            'Best_Cluster_Applied_Rate': bus_metrics['Applied_Rate'].max(),
            'Best_Cluster_Lift': bus_metrics['Lift'].max(),
            'Best_Cluster_Size': bus_metrics.iloc[0]['Size'],
            'Best_Cluster_Size_Pct': bus_metrics.iloc[0]['Size'] / bus_metrics['Size'].sum() * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*100)
    print("BINNED CLUSTERING EXPERIMENTS - FINAL SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False))
    
    # Recommendation
    best_exp = summary_df.loc[summary_df['Best_Cluster_Lift'].idxmax()]
    
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)
    print(f"Best Configuration: {best_exp['Experiment']}")
    print(f"Algorithm: {best_exp['Algorithm']}")
    print(f"Number of Clusters: {best_exp['N_Clusters']:.0f}")
    print(f"Best Cluster Applied Rate: {best_exp['Best_Cluster_Applied_Rate']:.4f}")
    print(f"Lift vs Baseline: {best_exp['Best_Cluster_Lift']:.2f}x")
    print(f"Best Cluster Size: {best_exp['Best_Cluster_Size']:.0f} ({best_exp['Best_Cluster_Size_Pct']:.1f}%)")
    print("="*100)
    
    return summary_df, best_exp

summary_df, best_experiment = generate_experiment_summary()
