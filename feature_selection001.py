import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# ======================================================
# STEP 1: DEFINE CANDIDATES BASED ON SEGMENTATION TABLE
# ======================================================
# We prioritize Engineered features (Log/Ratio) over Raw to reduce correlation.

candidate_pool = [
    # --- 1. DEMOGRAPHIC (Who they are) ---
    'AGE_RANGE_ENC', 
    'INCOME_RANGE_ENC',
    'HOME_MARKET_VALUE_RANGE_ENC',
    
    # --- 2. GEOGRAPHIC (Where they live) ---
    # Using the bucketized version to avoid 50+ dummy columns for states
    # Ensure this column is label-encoded (numeric) in your df
    'PRIMARY_STATE_CD_BUCKETIZED', 
    
    # --- 3. PSYCHOGRAPHIC (What they like/value) ---
    # Engagement Score is a strong proxy for affinity/lifestyle
    'MES_ENG_SCORE',
    'MES_RANK', 
    
    # --- 4. BEHAVIORAL (What they do) ---
    # A. SPEND (Magnitude & Trend)
    'log_LAST_12MTH_SPEND',       # Better than raw spend for clustering
    'ratio_spend_velocity',       # (3M vs 12M) - Identifying acceleration
    'D_YEAR_ON_YEAR_TREND_FOR_SPEND', # Provided trend feature
    
    # B. EARN (Loyalty Accumulation)
    'log_LTM_DLR_EARNED',         # Recent earning power
    
    # C. REDEEM (Usage/Burn)
    'ratio_burn_rate',            # (Redeem / Earn) - The "Burner" vs "Hoarder" behavior
    'flag_redeemer_life',         # Binary: Has ever redeemed?
    
    # D. BALANCE (Liability/Hoarding)
    'log_POINTS_BALANCE',         # Current holding magnitude
    'flag_high_points',           # The "Whale" flag (>10k points)
    
    # E. ACTIVE (Channel Usage)
    'flag_digital_active',        # Online/Email engaged
    'LTM_TRAN_ACTIVE',            # Transaction count bucket
    'D_MONTHONBOOK'               # Tenure (Lifecycle)
]

# ======================================================
# STEP 2: EXECUTION PIPELINE
# ======================================================

def run_feature_selection_pipeline(df, candidates):
    # 1. Engineer Features (Create the logs/ratios/flags defined above)
    print("Engineering features...")
    df_eng = engineer_marketing_features(df)
    
    # 2. Sanitize (Fix Infinity/NaN from logs)
    print("Sanitizing data...")
    df_clean = sanitize_features(df_eng)
    
    # 3. Filter Candidates (Ensure they exist in dataframe)
    # This prevents errors if a column name is slightly misspelled
    valid_candidates = [c for c in candidates if c in df_clean.columns]
    missing = set(candidates) - set(valid_candidates)
    if missing:
        print(f"Warning: These candidates were not found in data: {missing}")
    
    # 4. Run Selection (Correlation Drop + Proxy Selection)
    print(f"Running Selection on {len(valid_candidates)} features...")
    best_features = select_optimal_features(df_clean, valid_candidates)
    
    return df_clean, best_features

# ======================================================
# STEP 3: RUN IT
# ======================================================
# df_final, selected_features = run_feature_selection_pipeline(df, candidate_pool)

# print("\nFinal Selected Features for Clustering:")
# print(selected_features)


# CELL 2: DOMAIN-DRIVEN FEATURE ENGINEERING (ROBUST VERSION)
import numpy as np
import pandas as pd

def engineer_marketing_features(df):
    """
    Transforms raw marketing data into clustering-ready features.
    Includes safeguards against Infinity/NaN values.
    """
    data = df.copy()
    
    # --- A. THE "HURDLE" FLAGS (Binary) ---
    data['flag_is_active_12m'] = (data['LAST_12MTH_SPEND'] > 0).astype(int)
    data['flag_recent_buyer'] = (data['LAST_3MTH_SPEND'] > 0).astype(int)
    data['flag_high_points'] = (data['POINTS_BALANCE'] > 10000).astype(int)
    
    # Handle mixed types in LTM_ONLINE_ACTIVE
    if data['LTM_ONLINE_ACTIVE'].dtype == 'object':
         data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE'].apply(lambda x: 1 if x == 'Y' else 0)
    else:
         data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE']

    # --- SAFETY PRE-PROCESSING ---
    # Create temporary columns where negative values (refunds/-1) are clipped to 0
    # This ensures we never divide by zero or take log of a negative number
    safe_earn = data['TOTAL_EARNED_SINCE_2020'].clip(lower=0)
    safe_spend_12m = data['LAST_12MTH_SPEND'].clip(lower=0)
    safe_spend_3m = data['LAST_3MTH_SPEND'].clip(lower=0)

    # --- B. BEHAVIORAL RATIOS (Continuous) ---
    
    # 1. Burn Rate
    # Using 'safe_earn' guarantees denominator is at least 1 (0 + 1)
    data['ratio_burn_rate'] = data['TOTAL_RDM_SINCE_2020'] / (safe_earn + 1)
    
    # 2. Spend Velocity
    # Using 'safe_spend' guarantees denominator is at least 1
    data['ratio_spend_velocity'] = (safe_spend_3m * 4) / (safe_spend_12m + 1)

    # --- C. LOG TRANSFORMATIONS (Magnitude) ---
    cols_to_log = ['LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 'POINTS_BALANCE', 'LTM_DLR_EARNED']
    
    for col in cols_to_log:
        # clip(lower=0) prevents np.log1p(-1) which causes -inf
        data[f'log_{col}'] = np.log1p(data[col].clip(lower=0))
    
    # --- D. FINAL SAFETY NET ---
    # Just in case any other edge case produced infinity, replace it with 0
    data = data.replace([np.inf, -np.inf], 0)
    
    # Fill NaNs with 0 (assuming NaN implies no activity for these features)
    data = data.fillna(0)
        
    return data
