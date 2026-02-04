# CELL 1: SETUP & SANITIZATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import umap

# Set plot style
sns.set(style="whitegrid")

def clean_data_for_spectral(df, features):
    """
    Prepares data specifically for Spectral Clustering.
    1. Fills Missing Values
    2. Clips Extreme Outliers (to prevent infinite distance calculations)
    3. Scales to 0-1 (Required for the RBF Kernel in Spectral Clustering)
    """
    X = df[features].copy()
    
    # 1. Fill NaN with 0 (Assuming NaN = No Activity)
    X = X.fillna(0)
    
    # 2. Handle Infinities (from Log transforms)
    X = X.replace([np.inf, -np.inf], 0)
    
    # 3. Clip extremes (Safety net for RBF kernel stability)
    # Caps values at the 99th percentile to stop massive outliers from distorting the graph
    for col in X.columns:
        limit = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=limit)
        
    return X

# Usage:
# features_to_use = [List of features selected in previous step]
# X_clean = clean_data_for_spectral(df, features_to_use)
# print("Data sanitized and ready.")

##########################################

# CELL 2: SCALABLE SPECTRAL PIPELINE
def run_advanced_spectral(X_data, n_clusters=5):
    """
    Performs Spectral Clustering scaled for 1.8M rows.
    Uses Nystroem approximation to create 100 spectral dimensions.
    """
    print(f"--- Starting Spectral Clustering on {len(X_data)} customers ---")
    
    # 1. Scaling
    # Spectral clustering relies on distance, so features must be on same scale (0-1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Nyström Approximation (The "Advanced" Step)
    # Maps data to a non-linear spectral embedding space.
    # n_components=100: Creates 100 new 'spectral features' that capture complex topology.
    print("Approximating Spectral Embeddings (This may take 2-5 mins)...")
    nystroem = Nystroem(
        kernel='rbf', 
        gamma=None,       # Defaults to 1/n_features
        n_components=100, # 100 landmarks is sufficient for structure
        random_state=42,
        n_jobs=-1         # Use all CPU cores
    )
    
    # 3. Clustering
    # We cluster the *Spectral Embeddings*, not the raw data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # 4. Pipeline Execution
    pipeline = Pipeline([
        ('spectral_embedding', nystroem),
        ('clusterer', kmeans)
    ])
    
    pipeline.fit(X_scaled)
    labels = pipeline.predict(X_scaled)
    
    print("Clustering Complete.")
    return labels, pipeline, X_scaled

# Usage:
# labels, model, X_scaled = run_advanced_spectral(X_clean, n_clusters=5)
# df['Cluster'] = labels


############################################


# CELL 3: FAST VALIDATION
def validate_clusters_fast(X_scaled, labels):
    """
    Validates clusters using Variance Ratio.
    Higher Score = Better defined clusters.
    """
    print("--- Validation Metrics ---")
    
    # 1. Calinski-Harabasz Score
    # Measures how tight the clusters are vs. how far apart they are
    ch_score = calinski_harabasz_score(X_scaled, labels)
    print(f"Calinski-Harabasz Score: {ch_score:,.2f} (Higher is better)")
    
    # 2. Cluster Size Check
    # Ensures we didn't just create one giant cluster and 4 tiny ones
    counts = pd.Series(labels).value_counts().sort_index()
    print("\nCluster Size Distribution:")
    print(counts)
    
    # Alert if any cluster is < 1% of population
    if counts.min() < (len(labels) * 0.01):
        print("WARNING: Found micro-clusters (<1%). Consider reducing n_clusters.")

# Usage:
# validate_clusters_fast(X_scaled, labels)

#########################################################

# CELL 4: VISUALIZATION
def plot_spectral_insights(df, features_for_profiling, labels):
    """
    Generates UMAP Projection and Persona Heatmap.
    """
    # Create a copy for plotting to avoid messing up original data
    plot_df = df.copy()
    plot_df['Cluster'] = labels
    
    # --- A. UMAP PROJECTION (Sampled) ---
    print("Generating Topology Plot...")
    
    # Sample 10k points for speed (1.8M is too slow for UMAP plotting)
    sample_df = plot_df.sample(n=10000, random_state=42)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(sample_df[features_for_profiling])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=sample_df['Cluster'], 
        palette='tab10', 
        s=10, 
        alpha=0.6
    )
    plt.title('Spectral Cluster Topology (UMAP Projection)', fontsize=14)
    plt.show()

    # --- B. PERSONA HEATMAP (Snake Plot) ---
    print("Generating Persona Heatmap...")
    
    # Group by Cluster
    cluster_means = plot_df.groupby('Cluster')[features_for_profiling].mean()
    
    # Normalize Columns (0-1) so we can compare Spend vs Points
    # 0 = Lowest average in that category, 1 = Highest average
    scaler = MinMaxScaler()
    heatmap_data = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        linewidths=.5
    )
    plt.title('Cluster Personas (Normalized Intensity)', fontsize=14)
    plt.ylabel("Cluster ID")
    plt.show()

# Usage:
# Select RAW features (Real dollars/points) for the heatmap so it's readable
# profile_features = ['LAST_12MTH_SPEND', 'POINTS_BALANCE', 'ratio_burn_rate', 'tenure_months']
# plot_spectral_insights(df, profile_features, labels)


################################################


# ======================================================
# 4. INSIGHT GENERATION: The "Lift" Report
# ======================================================
def generate_cluster_insights(df_result, features):
    """
    Prints a narrative for each cluster based on Lift Analysis.
    Lift > 1.2 = High, Lift < 0.8 = Low.
    """
    global_mean = df_result[features].mean()
    cluster_stats = df_result.groupby('Cluster')[features].mean()
    
    # Calculate Lift (Cluster Mean / Global Mean)
    lift = cluster_stats.div(global_mean)
    
    print("\n--- CLUSTER INSIGHTS (Interpretation) ---")
    for cluster_id in lift.index:
        print(f"\n[ CLUSTER {cluster_id} ]")
        
        # Identify Defining Characteristics
        high_traits = lift.loc[cluster_id][lift.loc[cluster_id] > 1.2].index.tolist()
        low_traits = lift.loc[cluster_id][lift.loc[cluster_id] < 0.8].index.tolist()
        
        print(f"  • DOMINANT TRAITS (High): {', '.join(high_traits)}")
        print(f"  • WEAK TRAITS (Low):      {', '.join(low_traits)}")
        print(f"  • Size: {len(df_result[df_result['Cluster']==cluster_id]):,}")


#############################################################

# ======================================================
# 3. VISUALIZATION: Topology & Profiles
# ======================================================

def plot_cluster_visualizations(df_result, features_for_profiling, sample_size=10000):
    """
    Generates:
    1. UMAP Projection (Visual separation)
    2. Heatmap Profile (Business Persona)
    """
    # --- A. UMAP VISUALIZATION (Sampled for Speed) ---
    print("Generating UMAP Projection...")
    
    # Sample data to keep plotting fast (10k points is enough for visual check)
    if len(df_result) > sample_size:
        df_sample = df_result.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_result
        
    # Run UMAP on the *Raw Selected Features* of the sample
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df_sample[selected_features])
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=df_sample['Cluster'], 
        palette='tab10', 
        s=10, 
        alpha=0.6
    )
    plt.title('Cluster Separation (UMAP Projection)', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- B. BUSINESS HEATMAP (The "Snake Plot" Alternative) ---
    print("Generating Insight Heatmap...")
    
    # Calculate Mean of each feature per cluster
    # Use NON-LOG features for readability (e.g., Real Dollars, not Log Dollars)
    cluster_means = df_result.groupby('Cluster')[features_for_profiling].mean()
    
    # Normalize by Column (Min-Max) so we can compare Spend vs Points on one chart
    # 0 = Lowest in that category, 1 = Highest in that category
    scaler_viz = MinMaxScaler()
    heatmap_data = pd.DataFrame(
        scaler_viz.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="RdYlGn", 
        linewidths=.5
    )
    plt.title('Cluster Personas (Normalized 0-1)', fontsize=14)
    plt.ylabel("Cluster ID")
    plt.show()

# Usage Configuration:
# profile_cols = ['LAST_12MTH_SPEND', 'POINTS_BALANCE', 'ratio_burn_rate', 'tenure_months']
# plot_cluster_visualizations(df_spectral, profile_cols)


################################################################

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

def run_direct_spectral_with_propagation(df, features, n_clusters=5, sample_size=25000):
    """
    Runs standard Spectral Clustering on a sample, then propagates labels 
    to the full 1.8M dataset using KNN.
    """
    print(f"--- Strategy: Direct Spectral (N={sample_size}) + Propagation ---")
    
    # 1. Prepare Data
    X = df[features].fillna(0)
    
    # Clip outliers & Scale (Required for RBF Affinity)
    for col in X.columns:
        limit = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=limit)
        
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. SAMPLING: Create a Representative Subset
    # We use indices to track which rows are sampled
    sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    
    print(f"Running Standard Spectral Clustering on {sample_size} customers...")
    
    # 3. DIRECT SPECTRAL CLUSTERING (The Heavy Lift)
    # affinity='nearest_neighbors' is sparse and faster than 'rbf' for this size
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors', # Sparse graph is memory-friendly for 25k
        n_neighbors=10,
        assign_labels='kmeans',
        random_state=42,
        n_jobs=-1
    )
    
    # This step is slow but manageable for 25k rows
    sample_labels = spectral.fit_predict(X_sample)
    print("Spectral Clustering on sample complete.")
    
    # 4. PROPAGATION: Label the Remaining 1.78M Rows
    print("Propagating labels to full population (KNN)...")
    
    # Train a KNN classifier to "learn" the Spectral Cluster boundaries
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    knn.fit(X_sample, sample_labels)
    
    # Predict clusters for the ENTIRE dataset
    # This effectively "projects" the spectral logic to the full 1.8M
    full_labels = knn.predict(X_scaled)
    
    # 5. Validation (on the sample, since full silhouette is too slow)
    sil_score = silhouette_score(X_sample, sample_labels)
    print(f"Sample Silhouette Score: {sil_score:.3f}")
    
    # Attach results
    df_result = df.copy()
    df_result['Cluster'] = full_labels
    
    return df_result, sample_labels

# Usage
# df_final, labels = run_direct_spectral_with_propagation(df_clean, selected_features)


#############################################

# CELL 1: PROBABILISTIC SPECTRAL PIPELINE
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def run_spectral_probabilistic(X_data, n_clusters=5, method='gmm'):
    """
    Performs Scalable Spectral Clustering using Probabilistic Models (GMM/BGMM).
    Returns Labels AND the Probability Matrix (Soft Clustering).
    
    Parameters:
    - method: 'gmm' (Standard) or 'bgmm' (Bayesian - robust to overfitting)
    """
    print(f"--- Starting Spectral {method.upper()} on {len(X_data)} customers ---")
    
    # 1. Scaling (Crucial for Nystroem)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Nyström Embedding (The "Spectral" Part)
    # Maps complex data to 100-dim spectral space where GMM works beautifully
    print("Approximating Spectral Embeddings...")
    nystroem = Nystroem(
        kernel='rbf', 
        n_components=100, 
        random_state=42, 
        n_jobs=-1
    )
    
    # 3. Define the Probabilistic Clusterer
    if method == 'gmm':
        # Standard GMM
        clusterer = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full', # Allows elliptical clusters (better than KMeans spheres)
            n_init=3,               # Run 3 times to find best fit
            random_state=42
        )
    elif method == 'bgmm':
        # Bayesian GMM (Dirichlet Process)
        # weight_concentration_prior: Controls how "sparse" the weights are.
        # Lower value (e.g. 0.01) encourages using fewer clusters than n_clusters.
        clusterer = BayesianGaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            weight_concentration_prior=0.01, 
            n_init=1,
            random_state=42
        )
    
    # 4. Pipeline Execution
    # We fit the Nystroem first to transform data, then fit GMM on the result
    X_embedded = nystroem.fit_transform(X_scaled)
    
    print(f"Fitting {method.upper()} to Spectral Embeddings...")
    clusterer.fit(X_embedded)
    
    # Get Hard Labels (The highest probability cluster)
    labels = clusterer.predict(X_embedded)
    
    # Get Soft Probabilities (Matrix of shape [n_samples, n_clusters])
    probs = clusterer.predict_proba(X_embedded)
    
    print("Clustering Complete.")
    return labels, probs, X_embedded, clusterer

# Usage Example:
# labels_gmm, probs_gmm, X_embed, model = run_spectral_probabilistic(X_clean, n_clusters=5, method='gmm')


#################################

# CELL 2: VISUALIZING PROBABILITIES
import matplotlib.pyplot as plt
import seaborn as sns
import umap

def plot_probabilistic_insights(X_embedded, labels, probs, sample_size=10000):
    """
    Plots the UMAP topology colored by:
    1. Cluster ID
    2. Prediction Confidence (How sure is the model?)
    """
    # Calculate "Confidence" (Max probability for each user)
    # If max_prob is 0.99, the model is sure. If 0.35, the user is a "fence sitter".
    confidence = np.max(probs, axis=1)
    
    # Sample for plotting speed
    if len(X_embedded) > sample_size:
        indices = np.random.choice(len(X_embedded), sample_size, replace=False)
        X_samp = X_embedded[indices]
        labels_samp = labels[indices]
        conf_samp = confidence[indices]
    else:
        X_samp = X_embedded
        labels_samp = labels
        conf_samp = confidence

    # Run UMAP on the Embeddings for 2D projection
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_2d = reducer.fit_transform(X_samp)
    
    # --- Plot 1: The Clusters (GMM) ---
    plt.figure(figsize=(18, 7))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=umap_2d[:,0], y=umap_2d[:,1], hue=labels_samp, palette='tab10', s=10, alpha=0.6)
    plt.title("GMM Cluster Assignments", fontsize=14)
    
    # --- Plot 2: The Uncertainty (Who are the edge cases?) ---
    plt.subplot(1, 2, 2)
    sc = plt.scatter(umap_2d[:,0], umap_2d[:,1], c=conf_samp, cmap='RdYlGn', s=10, alpha=0.5)
    plt.colorbar(sc, label='Probability Confidence (1.0 = Sure, 0.2 = Unsure)')
    plt.title("Model Confidence (Green=Core Users, Red=Fence Sitters)", fontsize=14)
    
    plt.show()

# Usage:
# plot_probabilistic_insights(X_embed, labels_gmm, probs_gmm)


###################################################

import pandas as pd
import numpy as np

def engineer_features_v2(df):
    """
    Advanced Feature Engineering for Clustering.
    - Adds Winsorization (Outlier Capping)
    - Creates Interaction Terms (Spend x Engagement)
    - One-Hot Encodes Geography
    """
    data = df.copy()
    
    # --- 1. HANDLING CATEGORICAL (STATE) ---
    # One-Hot Encode the 5-category bucketized state
    # prefix='GEO' creates cols like: GEO_East, GEO_West, etc.
    if 'PRIMARY_STATE_CD_BUCKETIZED' in data.columns:
        data = pd.get_dummies(data, columns=['PRIMARY_STATE_CD_BUCKETIZED'], prefix='GEO', dtype=int)
    
    # --- 2. THE "HURDLE" FLAGS (Binary State Detectors) ---
    # Activity Flags
    data['flag_is_active_12m'] = (data['LAST_12MTH_SPEND'] > 0).astype(int)
    data['flag_recent_buyer_3m'] = (data['LAST_3MTH_SPEND'] > 0).astype(int)
    
    # Dormant Flag (Critical for "Lapsed" detection on whiteboard)
    # Active in lifetime, but 0 in last 12 months
    data['flag_dormant'] = (
        (data['TOTAL_EARNED_SINCE_2020'] > 0) & 
        (data['LAST_12MTH_SPEND'] == 0)
    ).astype(int)
    
    # Omnichannel Flag (Digital + Transaction Active)
    # Checks if user is active in BOTH Online and Transactions
    is_online = data['LTM_ONLINE_ACTIVE'].apply(lambda x: 1 if x == 'Y' else 0) if data['LTM_ONLINE_ACTIVE'].dtype == 'object' else data['LTM_ONLINE_ACTIVE']
    data['flag_omnichannel'] = ((is_online == 1) & (data['LTM_TRAN_ACTIVE'] > 0)).astype(int)

    # --- 3. WINSORIZATION & LOGS (Robust Magnitude) ---
    # We cap outliers at 99% percentile BEFORE logging to prevent cluster distortion
    magnitude_cols = ['LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 'POINTS_BALANCE', 'LTM_DLR_EARNED']
    
    for col in magnitude_cols:
        # 1. Clip negative values to 0
        cleaned_col = data[col].clip(lower=0)
        
        # 2. Winsorize (Cap at 99th percentile)
        # We calculate the cap on the valid data
        cap_limit = cleaned_col.quantile(0.99)
        cleaned_col = cleaned_col.clip(upper=cap_limit)
        
        # 3. Log Transform
        data[f'log_{col}'] = np.log1p(cleaned_col)

    # --- 4. BEHAVIORAL RATIOS & INTERACTIONS ---
    # Spend Velocity (Trend): Is spend accelerating?
    # (3M avg) vs (12M avg). >1.0 means increasing.
    safe_spend_12m = data['LAST_12MTH_SPEND'].clip(lower=0)
    safe_spend_3m = data['LAST_3MTH_SPEND'].clip(lower=0)
    data['ratio_spend_velocity'] = (safe_spend_3m * 4) / (safe_spend_12m + 1)
    
    # Burn Rate: Are they hoarding?
    data['ratio_burn_rate'] = data['TOTAL_RDM_SINCE_2020'] / (data['TOTAL_EARNED_SINCE_2020'].clip(lower=0) + 1)
    
    # Interaction: Engagement x Spend (The "Value Score")
    # Combines Psychographic (MES Score) with Behavioral (Spend)
    # Higher score = Highly engaged High spender (Best Cluster)
    data['score_engage_spend'] = data['MES_ENG_SCORE'] * data['log_LAST_12MTH_SPEND']

    # --- 5. CLEANUP ---
    # Drop infinite/NaNs created by ratios
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    return data



#################################################


candidate_pool = [
    # --- 1. DEMOGRAPHIC ---
    'AGE_RANGE_ENC',
    'INCOME_RANGE_ENC',
    'HOME_MARKET_VALUE_RANGE_ENC',
    
    # --- 2. GEOGRAPHIC (One-Hot Encoded) ---
    # We include the specific OHE columns generated by the function
    # You might need to check your actual column names after running the function
    # These are placeholders; typically pandas generates 'Col_Value'
    # Use: [c for c in df.columns if c.startswith('GEO_')]
    
    # --- 3. PSYCHOGRAPHIC ---
    'MES_ENG_SCORE',
    'score_engage_spend',          # NEW: Interaction feature
    
    # --- 4. BEHAVIORAL: FLAGS (The Hurdles) ---
    'flag_is_active_12m',          # Active vs Inactive
    'flag_recent_buyer_3m',        # Recency
    'flag_dormant',                # NEW: Lapsed detection
    'flag_omnichannel',            # NEW: Channel Usage
    'flag_high_points',            # Whales
    
    # --- 5. BEHAVIORAL: MAGNITUDE (The Scale) ---
    'log_LAST_12MTH_SPEND',        # Main Spend Metric
    'log_POINTS_BALANCE',          # Liability Metric
    'log_LTM_DLR_EARNED',          # Earning Velocity
    
    # --- 6. BEHAVIORAL: RATIOS (The Trend) ---
    'ratio_spend_velocity',        # Acceleration
    'ratio_burn_rate',             # Hoarding vs Burning
    'D_YEAR_ON_YEAR_TREND_FOR_SPEND', # Provided Trend
    'D_MONTHONBOOK'                # Tenure
]

