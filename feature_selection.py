import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score

# ---------------------------------------------------------
# 1. DEFINE TARGETS
# ---------------------------------------------------------
# TARGET 1: Did they Apply? (1 = Approved OR Denied, 0 = No Response)
df['target_applied'] = df['APPLICATION_STATUS'].notna().astype(int)

# TARGET 2: Were they Approved? (1 = Approved, 0 = Denied)
# Note: This is only valid where target_applied == 1
df['target_approved'] = np.where(df['APPLICATION_STATUS'] == 'approved', 1, 0)

# ---------------------------------------------------------
# 2. CREATE DATASETS
# ---------------------------------------------------------
# Select your optimized feature list (Top 24 from RFE)
features = [
    'LAST_12MTH_SPEND', 'TOTAL_EARNED_SINCE_2020', 'POINTS_BALANCE',
    'IS_SPEND_WHALE', 'IS_EARNING_WHALE', 'SPEND_ACCELERATION',
    'SPEND_LONG_TERM_TREND', 'HAS_POSITIVE_MOMENTUM', 'LAST_3MTH_SPEND',
    'OMNICHANNEL_SCORE', 'LTM_EMAIL_ACTIVE', 'IS_DIGITAL_WINDOW_SHOPPER',
    'TARGET_QUALITY_SCORE', 'PROFILE_COMPLETENESS', 'IS_STABLE_SPENDER',
    'ASSET_TO_INCOME_GAP', 'NULL_HOME_VALUE', 'D_MONTHONBOOK',
    'HOME_MARKET_VALUE'
]
# Add remaining features from your top 24 list...

X = df[features].fillna(0) # XGBoost handles NaNs, but 0 is safer for engineered flags

# --- DATASET 1: FOR INTENT MODEL (All Customers) ---
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X, df['target_applied'], test_size=0.2, stratify=df['target_applied'], random_state=42
)

# --- DATASET 2: FOR RISK MODEL (Only Applicants) ---
# We only train on people who actually applied (we know their outcome)
mask_applicants = df['target_applied'] == 1
X_applicants = X[mask_applicants]
y_applicants = df[mask_applicants]['target_approved']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_applicants, y_applicants, test_size=0.2, stratify=y_applicants, random_state=42
)

# ---------------------------------------------------------
# MODEL 1: WHO WANTS TO APPLY?
# ---------------------------------------------------------

# Calculate Imbalance Ratio for Weighting
ratio_1 = float(np.sum(y_train_1 == 0)) / np.sum(y_train_1 == 1)
print(f"Model 1 Imbalance Ratio: {ratio_1:.2f}")

clf_intent = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,       # Slow learning for better generalization
    max_depth=4,              # Shallow trees to prevent overfitting noise
    min_child_weight=5,       # Require more samples per leaf
    scale_pos_weight=ratio_1, # CRITICAL: Force model to care about Applicants
    subsample=0.8,
    colsample_bytree=0.7,
    objective='binary:logistic',
    eval_metric='aucpr',      # Optimize for Precision-Recall Area
    n_jobs=-1,
    random_state=42
)

print("Training Model 1 (Intent)...")
clf_intent.fit(X_train_1, y_train_1)

# EVALUATION
probs_intent = clf_intent.predict_proba(X_test_1)[:, 1]
print(f"Intent Model AUC: {roc_auc_score(y_test_1, probs_intent):.4f}")

# ---------------------------------------------------------
# MODEL 2: WHO WILL BE APPROVED?
# ---------------------------------------------------------

clf_risk = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,              # Slightly deeper to capture credit rules
    # scale_pos_weight=1,     # Usually balanced, remove if 50/50
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='auc',
    n_jobs=-1,
    random_state=42
)

print("Training Model 2 (Risk)...")
clf_risk.fit(X_train_2, y_train_2)

# EVALUATION
probs_risk = clf_risk.predict_proba(X_test_2)[:, 1]
print(f"Risk Model AUC: {roc_auc_score(y_test_2, probs_risk):.4f}")

# ---------------------------------------------------------
# 3. COMBINE LOGIC (ON FULL TEST POPULATION)
# ---------------------------------------------------------

# Step A: Get Predictions for EVERYONE using Both Models
# Even if they didn't apply, we want to know: "IF they applied, would they be approved?"
all_intent_scores = clf_intent.predict_proba(X_test_1)[:, 1]
all_risk_scores = clf_risk.predict_proba(X_test_1)[:, 1]

# Step B: Determine Optimal Thresholds
# For Intent: Maximize F1 (Balance) or Recall (Coverage)
p, r, t = precision_recall_curve(y_test_1, all_intent_scores)
f1_scores = 2 * (p * r) / (p + r)
thresh_intent = t[np.argmax(f1_scores)]
print(f"Optimal Intent Threshold: {thresh_intent:.4f}")

# For Risk: Fixed conservative threshold (e.g., 0.5 or 0.6)
thresh_risk = 0.5 

# Step C: The Waterfall Filter
# Logic: A "Good Lead" is someone who WANTS it AND can GET it.
final_predictions = (
    (all_intent_scores >= thresh_intent) &  # They are interested
    (all_risk_scores >= thresh_risk)        # They are credit-worthy
).astype(int)

# ---------------------------------------------------------
# FINAL EVALUATION
# ---------------------------------------------------------
print("\n--- Final Pipeline Results ---")
# Compare against the Ground Truth "Applied" (y_test_1)
# Note: This is strict. It counts "Applied but Denied" as a failure here 
# because we filtered them out with the Risk model.
print(classification_report(y_test_1, final_predictions))

# ALTERNATIVE BUSINESS METRIC:
# "How many of the people we predicted as 'Good Leads' actually Applied?"
predicted_leads = np.sum(final_predictions == 1)
actual_applicants_captured = np.sum((final_predictions == 1) & (y_test_1 == 1))

print(f"Total Recommended Leads: {predicted_leads}")
print(f"Actual Applicants Captured: {actual_applicants_captured}")
print(f"Pipeline Precision: {actual_applicants_captured / predicted_leads:.2%}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

class HybridFeatureSelector:
    def __init__(self, n_features_to_select=24, n_jobs=-1, random_state=42):
        self.n_features = n_features_to_select
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selected_features_ = []
        self.ranking_df_ = None

    def fit(self, X, y, task_name="Model"):
        """
        Runs both RFE and RF Importance, then combines them via Rank Sum.
        """
        print(f"--- Starting Hybrid Selection for {task_name} ---")
        
        # 1. SETUP ESTIMATORS
        # We use Random Forest for both to maintain consistency in feature evaluation logic.
        # For the 'Applied' model, class_weight='balanced' is CRITICAL.
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,            # Restrict depth to prevent overfitting during selection
            class_weight='balanced', # Critical for Imbalance
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        # 2. RUN RANDOM FOREST IMPORTANCE (Embedded)
        print(f"[{task_name}] Running Random Forest Importance...")
        rf_model.fit(X, y)
        importances = rf_model.feature_importances_
        
        # Create DataFrame and Rank (1 = Best)
        df_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        df_imp['Rank_Emb'] = df_imp['Importance'].rank(ascending=False)

        # 3. RUN RECURSIVE FEATURE ELIMINATION (Wrapper)
        # RFE removes the weakest features one by one.
        print(f"[{task_name}] Running RFE (Wrapper)... This may take time.")
        rfe = RFE(
            estimator=rf_model,
            n_features_to_select=self.n_features,
            step=0.05  # Remove 5% of features at each step for speed
        )
        rfe.fit(X, y)
        
        # Get RFE Ranking (1 = Selected, >1 = Eliminated)
        # We handle ties by using the raw ranking from RFE
        rfe_ranks = rfe.ranking_
        df_rfe = pd.DataFrame({'Feature': X.columns, 'Rank_RFE': rfe_ranks})

        # 4. COMBINE RESULTS (Rank Sum / Borda Count)
        self.ranking_df_ = pd.merge(df_imp, df_rfe, on='Feature')
        
        # Calculate Final Score: Lower is Better
        # We weight RFE slightly higher (multiply by 1.0) vs Importance (multiply by 1.0)
        # You can adjust weights if you trust one method more.
        self.ranking_df_['Final_Score'] = self.ranking_df_['Rank_Emb'] + self.ranking_df_['Rank_RFE']
        
        # Sort by Best Score
        self.ranking_df_ = self.ranking_df_.sort_values('Final_Score')
        
        # Select Top N
        self.selected_features_ = self.ranking_df_.head(self.n_features)['Feature'].tolist()
        
        print(f"[{task_name}] Selection Complete. Top {len(self.selected_features_)} features selected.")
        return self

# =========================================================
# EXECUTION BLOCK
# =========================================================

# Initialize the Selector
# We want top 24 features as per your request
selector = HybridFeatureSelector(n_features_to_select=24)

# ---------------------------------------------------------
# TASK 1: FEATURE SELECTION FOR APPLIED MODEL (Imbalanced)
# ---------------------------------------------------------
# Uses X_train_1, y_train_1 (The Dataset with All Customers)
# Ensure X_train_1 contains ALL your candidate features (100+)
selector.fit(X_train_1, y_train_1, task_name="Applied_Model")

top_features_applied = selector.selected_features_
ranking_applied = selector.ranking_df_

print("\nTop 24 Features for APPLIED Model:")
print(top_features_applied)

# ---------------------------------------------------------
# TASK 2: FEATURE SELECTION FOR APPROVED MODEL (Balanced)
# ---------------------------------------------------------
# Uses X_train_2, y_train_2 (The Dataset with Only Applicants)
selector.fit(X_train_2, y_train_2, task_name="Approved_Model")

top_features_approved = selector.selected_features_
ranking_approved = selector.ranking_df_

print("\nTop 24 Features for APPROVED Model:")
print(top_features_approved)

