from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def validate_clusters_on_raw_data(df, labels, features_to_check):
    """
    Validates clusters by calculating the Calinski-Harabasz score 
    on the ORIGINAL (Raw) features after applying fresh scaling.
    """
    print("--- Validation on Raw Business Features ---")
    
    # 1. Select the specific raw features you want to validate against
    # Use .copy() to ensure we don't modify the original dataframe
    X_val = df[features_to_check].copy()
    
    # 2. Safety Cleanup (Raw data often has NaNs that break scoring)
    # Fill NaN with 0 (assuming null = no activity)
    X_val = X_val.fillna(0)
    
    # 3. Scaling
    # We must scale raw metrics (e.g., Spend $5000 vs Score 10) so they contribute equally
    scaler = MinMaxScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    
    # 4. Calculate Score
    ch_score = calinski_harabasz_score(X_val_scaled, labels)
    
    print(f"Features used: {len(features_to_check)}")
    print(f"Calinski-Harabasz Score: {ch_score:,.2f} (Higher is better)")
    
    # 5. Cluster Size Distribution (Always good to see)
    print("\nCluster Size Distribution:")
    counts = pd.Series(labels).value_counts().sort_index()
    print(counts)
    
    return ch_score

# ==========================================
# EXECUTION
# ==========================================

# Your list of raw features (from your image)
cols_check = [
    'LAST_3MTH_SPEND', 'LAST_12MTH_SPEND', 'LAST_36MTH_SPEND', 
    'LTM_RDM', 'LTM_DLR_EARNED', 'POINTS_BALANCE',
    'MES_ENG_SCORE', 'LTM_ONLINE_ACTIVE', 'L24M_EMAIL_ACTIVE', 
    'TOTAL_RDM_SINCE_2020', 'AGE_RANGE_ENC',
    'CENSUS_INCOME_RANGE_ENC', 'L6M_REDEEMER', 'LTM_TRAN_ACTIVE', 
    'INCOME_RANGE_ENC', 'RFM_SEGMENT_ENC'
]

# Run the validation
# df_train: Your original training data
# labels: The cluster labels you generated from Nyström/Spectral
score = validate_clusters_on_raw_data(df_train, labels, cols_check)
