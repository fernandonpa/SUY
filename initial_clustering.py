# CELL 1: SETUP & LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler

# Setting display options for clearer output analysis
pd.set_option('display.max_columns', None)
np.random.seed(42)

print("Libraries Loaded. Ready for Feature Engineering.")


# CELL 2: DOMAIN-DRIVEN FEATURE ENGINEERING
def engineer_marketing_features(df):
    """
    Transforms raw marketing data into clustering-ready features based on 
    'Activity Cliffs' and Whiteboard Logic.
    """
    data = df.copy()
    
    # --- A. THE "HURDLE" FLAGS (Binary) ---
    # Logic: Capture the 'Step Change' in response rates seen in Images 1, 2, 6.
    
    # 1. Active Spender Flag (Image 1 & 2)
    # Separates the 'Dead' (0.3% response) from the 'Alive' (>6% response)
    data['flag_is_active_12m'] = (data['LAST_12MTH_SPEND'] > 0).astype(int)
    data['flag_recent_buyer'] = (data['LAST_3MTH_SPEND'] > 0).astype(int)
    
    # 2. Points Hoarder Flag (Image 6)
    # Threshold > 10,000 based on the 12.4% response rate spike in your analysis
    data['flag_high_points'] = (data['POINTS_BALANCE'] > 10000).astype(int)
    
    # 3. Digital Engagement (Whiteboard 'Usage' Branch)
    # Assumes 'LTM_ONLINE_ACTIVE' is 'Y'/'N' or 1/0
    if data['LTM_ONLINE_ACTIVE'].dtype == 'object':
         data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE'].apply(lambda x: 1 if x == 'Y' else 0)
    else:
         data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE']

    # --- B. BEHAVIORAL RATIOS (Continuous) ---
    # Logic: Capture Velocity and Wallet Share
    
    # 1. Burn Rate (Redeem vs Earn)
    # Distinguishes "Hoarders" (High Balance, Low Burn) from "Burners"
    # Adding 1 to denominator to avoid division by zero
    data['ratio_burn_rate'] = data['TOTAL_RDM_SINCE_2020'] / (data['TOTAL_EARNED_SINCE_2020'] + 1)
    
    # 2. Spend Velocity (Recent vs Annual)
    # (3M Spend * 4) / 12M Spend. 
    # > 1.0 means spending is accelerating. < 1.0 means slowing down (Lapsing).
    data['ratio_spend_velocity'] = (data['LAST_3MTH_SPEND'] * 4) / (data['LAST_12MTH_SPEND'] + 1)

    # --- C. LOG TRANSFORMATIONS (Magnitude) ---
    # Logic: Financial data follows Power Law. Log-transform creates Gaussian-like distributions for clustering.
    cols_to_log = ['LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 'POINTS_BALANCE', 'LTM_DLR_EARNED']
    for col in cols_to_log:
        data[f'log_{col}'] = np.log1p(data[col])
        
    return data

# Usage Example:
# df_engineered = engineer_marketing_features(raw_df)
# print("Feature Engineering Complete.")

# CELL 3: STATISTICAL & ML FEATURE SELECTION
def select_optimal_features(df, candidate_cols):
    """
    1. Removes Highly Correlated Features (Statistical).
    2. Uses 'Proxy' Random Forest to find features that distinguish real data from noise (ML).
    """
    X = df[candidate_cols].fillna(0)
    
    # --- STEP 1: STATISTICAL CORRELATION FILTER ---
    # If two features are 95% correlated, the model doesn't need both.
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    X_reduced = X.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} correlated features: {to_drop}")
    
    # --- STEP 2: ML 'PROXY' SELECTION (Structure vs Noise) ---
    # We train a model to distinguish Real Data vs Shuffled Data.
    # Features that help the model win are the ones holding the 'Cluster Structure'.
    
    X_real = X_reduced.copy()
    X_real['is_real'] = 1
    
    X_shadow = X_reduced.copy()
    # Shuffle each column to break the structure
    for col in X_shadow.columns:
        X_shadow[col] = np.random.permutation(X_shadow[col].values)
    X_shadow['is_real'] = 0
    
    # Combine
    X_combined = pd.concat([X_real, X_shadow])
    y = X_combined['is_real']
    X_train = X_combined.drop('is_real', axis=1)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y)
    
    # Select features better than 'Median' importance
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    selected_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_indices].tolist()
    
    # Plotting Importances
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10,6))
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title("Feature Importance for Clustering Structure")
    plt.show()
    
    return selected_features

# Usage Example:
# best_features = select_optimal_features(df_engineered, candidate_list)

# CELL 4: EXECUTION & PREPARATION
# 1. Define your initial candidate pool (Numeric Only)
# Combine the Raw features you have with the New ones we made
candidate_pool = [
    # -- Engineered --
    'flag_is_active_12m', 'flag_recent_buyer', 'flag_high_points', 'flag_digital_active',
    'ratio_burn_rate', 'ratio_spend_velocity',
    'log_LAST_12MTH_SPEND', 'log_POINTS_BALANCE',
    
    # -- Raw (Selected based on Images) --
    'AGE_RANGE_ENC', 'INCOME_RANGE_ENC', 
    'MES_ENG_SCORE', 'LTM_TRAN_ACTIVE',
    'D_MONTHONBOOK', 'D_YEAR_ON_YEAR_TREND_FOR_SPEND'
]

# Note: Add any other 'D_' columns created by your data team if they are numeric.

# 2. Run the Pipeline
# Assuming 'df' is your loaded dataframe
df_eng = engineer_marketing_features(df)
final_features = select_optimal_features(df_eng, candidate_pool)

print("\nFINAL SELECTED FEATURES FOR CLUSTERING:")
print(final_features)

# 3. Create Final Dataset for Clustering
X_clustering = df_eng[final_features].fillna(0)

# 4. Scaling (Crucial for Clustering)
# Use RobustScaler to handle the outliers visible in your 'Spend' histograms
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clustering)

print("Data scaled and ready for UMAP/HDBSCAN.")


