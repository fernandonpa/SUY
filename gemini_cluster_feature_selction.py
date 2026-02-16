import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

# 1. Define Target Variable
# Target = 1 if APPLICATION_STATUS is NOT Null (Meaning they applied), else 0
# We use the original df_train to get the status, assuming indices match df_eng_t
y = df_train['APPLICATION_STATUS'].notnull().astype(int)

# 2. Define Feature Matrix (X)
# We use the 'candidate_pool' features you defined in the previous steps
# Ensure we strictly use the engineered dataframe
X = df_eng_t[candidate_pool].copy()

# 3. Quick Cleanup
# Replace infinity and fill NaNs to ensure models don't crash
X = X.replace([np.inf, -np.inf], 0).fillna(0)

print(f"Feature Selection Data Prepared.")
print(f"Features: {X.shape[1]}")
print(f"Target Distribution: {y.value_counts(normalize=True).to_dict()}")


# ==========================================
# METHOD 1: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================
print("\n--- 1. Running Random Forest Feature Importance ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Get Importance Scores
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_10_rf = importances.head(10)['Feature'].tolist()
print("Top 10 (RF Importance):", top_10_rf)

# ==========================================
# METHOD 2: RECURSIVE FEATURE ELIMINATION (RFE)
# ==========================================
print("\n--- 2. Running Recursive Feature Elimination (RFE) ---")
# We use a lighter model for RFE to keep it fast, or the same RF
rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), 
                   n_features_to_select=10, 
                   step=1) # Remove 1 feature at a time
rfe_selector.fit(X, y)

top_10_rfe = X.columns[rfe_selector.support_].tolist()
print("Top 10 (RFE):", top_10_rfe)

# ==========================================
# METHOD 3: CORRELATION WITH TARGET
# ==========================================
print("\n--- 3. Running Correlation with Target ---")
# We calculate correlation of every feature with 'y'
correlations = X.apply(lambda x: x.corr(y))

# We take absolute value because strong negative correlation is also a good predictor
top_10_corr = correlations.abs().sort_values(ascending=False).head(10).index.tolist()

# Display with actual signs (positive/negative) for context
print("Top 10 (Target Correlation):")
print(correlations.loc[top_10_corr])


# ==========================================
# 4. CONSENSUS & FINAL SELECTION
# ==========================================
# Union of all top features (Unique set)
selected_features_union = list(set(top_10_rf + top_10_rfe + top_10_corr))

print(f"\nTotal Unique Features Selected: {len(selected_features_union)}")
print(selected_features_union)

# ==========================================
# 5. CROSS-CORRELATION CHECK (Prevent Multicollinearity)
# ==========================================
print("\n--- Checking Correlation BETWEEN Selected Features ---")

# Calculate correlation matrix of just the winners
corr_matrix_selected = X[selected_features_union].corr()

# Visualize
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix_selected, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    center=0,
    linewidths=0.5
)
plt.title('Correlation Matrix of Optimal Features')
plt.show()

# ==========================================
# AUTOMATED FILTERING (Optional)
# ==========================================
# If you want to automatically drop highly correlated pairs (e.g., > 0.8)
def drop_collinear_features(df_features, threshold=0.85):
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [c for c in df_features.columns if c not in to_drop]

final_optimal_features = drop_collinear_features(X[selected_features_union])

print(f"\nFinal Optimized Feature List (Redundancy Removed): {len(final_optimal_features)}")
print(final_optimal_features)


