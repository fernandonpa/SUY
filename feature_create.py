import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data (Replace with your actual file path)
# df = pd.read_csv('your_dataset.csv')
df_fe = df.copy()

# 2. STRICTLY DROP MES FEATURES
mes_cols = [col for col in df_fe.columns if 'MES' in col]
print(f"Dropping MES Features: {mes_cols}")
df_fe.drop(columns=mes_cols, inplace=True, errors='ignore')

# 3. Define Key Feature Groups (Available in your list)
spend_cols = ['LAST_3MTH_SPEND', 'LAST_12MTH_SPEND', 'LAST_36MTH_SPEND']
email_cols = ['LTM_EMAIL_ACTIVE', 'L18M_EMAIL_ACTIVE', 'L24M_EMAIL_ACTIVE']
points_cols = ['POINTS_BALANCE', 'TOTAL_EARNED_SINCE_2020', 'LTM_DLR_EARNED']
active_cols = ['LTM_ONLINE_ACTIVE', 'LTM_TRAN_ACTIVE', 'LTM_EMAIL_ACTIVE']

print("Setup Complete. MES features removed.")

#######################################################################################

# Create flags for missing critical financial info
# (Approved customers usually have this info populated)

# 1. Critical Missing Flags
df_fe['NULL_HOME_VALUE'] = df_fe['HOME_MARKET_VALUE'].isnull().astype(int)
df_fe['NULL_INCOME'] = df_fe['INCOME_RANGE'].isnull().astype(int)
df_fe['NULL_AGE'] = df_fe['AGE_RANGE'].isnull().astype(int)

# 2. "Ghost" Profile (Missing both Home and Income info)
df_fe['IS_FINANCIAL_GHOST'] = (df_fe['NULL_HOME_VALUE'] & df_fe['NULL_INCOME']).astype(int)

# 3. "Complete" Profile Score (Inverse of nulls)
# Higher score = More data available = Likely higher creditworthiness
df_fe['PROFILE_COMPLETENESS'] = 3 - (df_fe['NULL_HOME_VALUE'] + df_fe['NULL_INCOME'] + df_fe['NULL_AGE'])

print("Null Pattern Features Created.")

#######################################################################################


def get_outlier_score(series):
    """Returns 1 if value is > 95th percentile (High Value Outlier)"""
    threshold = series.quantile(0.95)
    return (series > threshold).astype(int)

# 1. Spending Whales
df_fe['IS_SPEND_WHALE'] = get_outlier_score(df_fe['LAST_12MTH_SPEND'])

# 2. Points Hoarders (High balance often indicates loyalty/value)
df_fe['IS_POINTS_WHALE'] = get_outlier_score(df_fe['POINTS_BALANCE'])

# 3. Earning Whales
df_fe['IS_EARNING_WHALE'] = get_outlier_score(df_fe['TOTAL_EARNED_SINCE_2020'])

# 4. Composite "High Value" Flag
# If they are an outlier in at least 2 categories, they are very likely Minority class
df_fe['HIGH_VALUE_OUTLIER_SCORE'] = (
    df_fe['IS_SPEND_WHALE'] + 
    df_fe['IS_POINTS_WHALE'] + 
    df_fe['IS_EARNING_WHALE']
)

print("Outlier Features Created.")



#######################################################################################


# 1. Spending Acceleration (Recent vs Annual)
# Formula: (Last 3 Months * 4) / Last 12 Months
# > 1.0 means they are spending MORE recently (Acceleration)
df_fe['SPEND_ACCELERATION'] = np.where(
    df_fe['LAST_12MTH_SPEND'] > 0,
    (df_fe['LAST_3MTH_SPEND'] * 4) / df_fe['LAST_12MTH_SPEND'],
    0
)

# 2. Long-term Spending Trend (Last 12m vs Last 36m)
# Formula: Last 12m / (Last 36m / 3)
df_fe['SPEND_LONG_TERM_TREND'] = np.where(
    df_fe['LAST_36MTH_SPEND'] > 0,
    df_fe['LAST_12MTH_SPEND'] / (df_fe['LAST_36MTH_SPEND'] / 3),
    0
)

# 3. Email Engagement Stickiness
# Are they still active recently compared to 2 years ago?
# 1 = Active now and 2 years ago (Loyal)
# 0 = Dropped off
df_fe['EMAIL_STICKINESS'] = (df_fe['LTM_EMAIL_ACTIVE'] == 1) & (df_fe['L24M_EMAIL_ACTIVE'] == 1)
df_fe['EMAIL_STICKINESS'] = df_fe['EMAIL_STICKINESS'].astype(int)

# 4. Activity Momentum Flag
df_fe['HAS_POSITIVE_MOMENTUM'] = (
    (df_fe['SPEND_ACCELERATION'] > 1.0) | 
    (df_fe['EMAIL_STICKINESS'] == 1)
).astype(int)

print("Velocity & Momentum Features Created.")


#######################################################################################

# 1. Spend to Earning Ratio (Burn Rate)
# How much of their points earning are they "spending" (redeeming)?
# High redemption can indicate engagement OR cashing out.
df_fe['POINTS_BURN_RATE'] = np.where(
    df_fe['TOTAL_EARNED_SINCE_2020'] > 0,
    df_fe['TOTAL_RDM_SINCE_2020'] / df_fe['TOTAL_EARNED_SINCE_2020'],
    0
)

# 2. "House Rich" Flag
# High Home Value but Low Income (Encoded) might indicate asset wealth vs cash poor
# Assuming Encoded variables: Higher number = Higher Value
df_fe['ASSET_TO_INCOME_GAP'] = df_fe['HOME_MARKET_VALUE_RANGE_ENC'] - df_fe['INCOME_RANGE_ENC']

# 3. Spend Consistency
# If 12m spend is roughly 1/3 of 36m spend, they are consistent (Low Risk)
# Calculate deviation from perfect consistency (1.0)
df_fe['SPEND_STABILITY_INDEX'] = np.abs(1.0 - df_fe['SPEND_LONG_TERM_TREND'])
# Flag for highly stable spenders (deviation < 0.2)
df_fe['IS_STABLE_SPENDER'] = (df_fe['SPEND_STABILITY_INDEX'] < 0.2).astype(int)

print("Financial Capacity Features Created.")


#######################################################################################

# 1. Omnichannel Score
# Sum of active flags (Online + Transaction + Email)
# Range 0 to 3
df_fe['OMNICHANNEL_SCORE'] = (
    df_fe['LTM_ONLINE_ACTIVE'] + 
    df_fe['LTM_TRAN_ACTIVE'] + 
    df_fe['LTM_EMAIL_ACTIVE']
)

# 2. "Digital Only" User
# Active Online but NOT in Transactions (Browsing but not buying?)
df_fe['IS_DIGITAL_WINDOW_SHOPPER'] = (
    (df_fe['LTM_ONLINE_ACTIVE'] == 1) & 
    (df_fe['LTM_TRAN_ACTIVE'] == 0)
).astype(int)

# 3. "Dark" User (No Email, No Online, but Active Transactions)
# Hard to reach, but spends money
df_fe['IS_DARK_SPENDER'] = (
    (df_fe['LTM_EMAIL_ACTIVE'] == 0) & 
    (df_fe['LTM_ONLINE_ACTIVE'] == 0) & 
    (df_fe['LTM_TRAN_ACTIVE'] == 1)
).astype(int)

print("Channel Engagement Features Created.")

#######################################################################################

# Create a composite score without MES
# Weights derived from typical credit risk intuition

df_fe['TARGET_QUALITY_SCORE'] = (
    # Financial Stability (40 pts)
    (df_fe['IS_STABLE_SPENDER'] * 20) +
    (df_fe['PROFILE_COMPLETENESS'] * 5) +  # Max 15 pts (3 * 5)
    (df_fe['IS_EARNING_WHALE'] * 5) +
    
    # Engagement/Activity (30 pts)
    (df_fe['OMNICHANNEL_SCORE'] * 5) +     # Max 15 pts (3 * 5)
    (df_fe['HAS_POSITIVE_MOMENTUM'] * 15) +
    
    # Value/Assets (30 pts)
    (df_fe['IS_SPEND_WHALE'] * 15) +
    (df_fe['IS_POINTS_WHALE'] * 15)
)

# Normalize to 0-1
df_fe['TARGET_QUALITY_SCORE'] = df_fe['TARGET_QUALITY_SCORE'] / 100.0

print("Master Composite Score Created.")


#######################################################################################

# Drop intermediate columns if desired, or keep them for the tree to decide
# Ensure no infinite values from division
df_fe.replace([np.inf, -np.inf], 0, inplace=True)
df_fe.fillna(0, inplace=True)

# Select Numeric features only for modeling (drop original object cols)
numeric_df = df_fe.select_dtypes(include=[np.number])

print(f"Final Feature Set Shape: {numeric_df.shape}")
print(f"New Features Added: {numeric_df.shape[1] - len(df.select_dtypes(include=[np.number]).columns)}")
# numeric_df.to_csv('features_no_mes.csv', index=False)


#######################################################################################

# XGBOOST CONFIGURATION FOR EXTREME IMBALANCE
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Calculate Ratio for Scale Pos Weight
# Formula: sum(negative instances) / sum(positive instances)
num_neg = (df_fe['APPLICATION_STATUS'] == 0).sum()
num_pos = (df_fe['APPLICATION_STATUS'] == 1).sum()
scale_weight = num_neg / num_pos

print(f"Imbalance Ratio: {scale_weight:.2f}")

# 2. XGBoost Parameters
# 'scale_pos_weight': Tells the model to pay 170x more attention to Class 1 errors
# 'max_delta_step': Helps convergence in extreme imbalance (set to 1-10)
xgb_params = {
    'scale_pos_weight': scale_weight,
    'max_delta_step': 1,              # Critical for extreme imbalance
    'learning_rate': 0.02,            # Go slow
    'max_depth': 4,                   # Keep trees shallow to prevent overfitting minority
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'aucpr',           # Optimize for Precision-Recall AUC, not ROC
    'objective': 'binary:logistic',
    'n_jobs': -1
}

print("Model Parameters Configured.")
# clf = xgb.XGBClassifier(**xgb_params)
# clf.fit(X_train, y_train)

#######################################################################################
