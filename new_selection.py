import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.semi_supervised import SelfTrainingClassifier
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# 1. PREPARE DATA
# Assume 'df' is your main dataframe
# Target: 1 = Applied, 0 = Not Applied (Treating as Unlabeled for Self-Training)
target_col = 'target_applied'
if target_col not in df.columns:
    df[target_col] = df['APPLICATION_STATUS'].notna().astype(int)

# Use your Full Feature Candidate List (100+ features)
# X_full = df[list_of_all_features].fillna(0)
# y_full = df[target_col]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
)

# ---------------------------------------------------------
# HELPER: SELF-TRAINING MODEL WRAPPER
# ---------------------------------------------------------
def train_self_supervised_xgboost(X_train_sel, y_train_sel, X_test_sel, y_test_sel, method_name):
    """
    Implements Self-Training:
    1. Treat 95% of '0' labels as Unlabeled (-1) to let model find hidden positives.
    2. Iteratively predict on Unlabeled data.
    3. If confidence > threshold, move Unlabeled -> Labeled.
    """
    print(f"\n--- Training Self-Supervised Model ({method_name}) ---")
    
    # 1. PREPARE Y FOR SELF-TRAINING
    # Sklearn SelfTrainingClassifier expects -1 for Unlabeled data.
    # We keep Class 1 (Applied) as 1.
    # We keep a small set of Class 0 (Not Applied) as 0 (True Negatives - e.g. Opt-Outs).
    # We mark the rest of Class 0 as -1 (Unlabeled - Potential Applicants).
    
    y_train_ss = y_train_sel.copy()
    
    # Logic: Mark 90% of the 0s as "Unlabeled" (-1) so the model can re-classify them
    # This helps if your problem is that "Non-Applicants" are actually just "Not Yet Applicants"
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(y_train_ss)) < 0.90
    
    # Only mask the 0s, leave the 1s alone (we know they applied)
    mask = (y_train_ss == 0) & random_unlabeled_points
    y_train_ss[mask] = -1
    
    print(f"Original 1s: {sum(y_train_sel==1)}")
    print(f"Original 0s: {sum(y_train_sel==0)}")
    print(f"Self-Training Setup -> Labeled 1s: {sum(y_train_ss==1)}, Labeled 0s: {sum(y_train_ss==0)}, Unlabeled (-1): {sum(y_train_ss==-1)}")

    # 2. DEFINE BASE MODEL (XGBoost)
    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )

    # 3. CONFIGURE SELF-TRAINING
    # criterion='threshold': Only label if prob > threshold
    # threshold=0.85: Be strict. Only call it a '1' if extremely confident.
    self_training_model = SelfTrainingClassifier(
        base_model,
        criterion='threshold',
        threshold=0.85, 
        verbose=True
    )

    # 4. FIT
    self_training_model.fit(X_train_sel, y_train_ss)

    # 5. EVALUATE
    print(f"--- Results for {method_name} ---")
    y_pred = self_training_model.predict(X_test_sel)
    print(classification_report(y_test_sel, y_pred))
    
    # Check ROC
    y_prob = self_training_model.predict_proba(X_test_sel)[:, 1]
    print(f"AUC Score: {roc_auc_score(y_test_sel, y_prob):.4f}")
    
    return self_training_model   


# ---------------------------------------------------------
# METHOD 1: BORUTA FEATURE SELECTION
# ---------------------------------------------------------
print("\n========== RUNNING EXPERIMENT 1: BORUTA ==========")

# 1. SETUP BORUTA
# Boruta needs a Random Forest (XGBoost works too but RF is standard for Boruta)
# We use class_weight='balanced' because Boruta struggles with imbalance otherwise
rf_boruta = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    max_depth=5,
    n_jobs=-1
)

feat_selector_boruta = BorutaPy(
    rf_boruta, 
    n_estimators='auto', 
    verbose=2, 
    random_state=42, 
    max_iter=50 # Number of trials
)

# 2. RUN SELECTION (Accepts numpy arrays only)
# This might take time
feat_selector_boruta.fit(X_train.values, y_train.values)

# 3. GET SELECTED FEATURES
boruta_support = feat_selector_boruta.support_
selected_cols_boruta = X_train.columns[boruta_support].tolist()

print(f"\nBoruta Selected {len(selected_cols_boruta)} Features:")
print(selected_cols_boruta)

# 4. TRAIN SELF-SUPERVISED MODEL WITH BORUTA FEATURES
if len(selected_cols_boruta) > 0:
    train_self_supervised_xgboost(
        X_train[selected_cols_boruta], y_train,
        X_test[selected_cols_boruta], y_test,
        method_name="Boruta + Self-Training"
    )
else:
    print("Boruta selected 0 features. Relax constraints or check data.")

# ---------------------------------------------------------
# METHOD 2: CHI-SQUARED FEATURE SELECTION
# ---------------------------------------------------------
print("\n========== RUNNING EXPERIMENT 2: CHI-SQUARED ==========")

# 1. PRE-PROCESS (MinMax Scaling)
# Chi2 requires non-negative values. Standard Scaling (Z-score) creates negatives.
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 2. RUN SELECTION
# Select Top 30 features (Chi2 is a ranking method, you must pick K)
k_features = 30
chi2_selector = SelectKBest(chi2, k=k_features)
chi2_selector.fit(X_train_scaled, y_train)

# 3. GET SELECTED FEATURES
cols_idxs = chi2_selector.get_support(indices=True)
selected_cols_chi2 = X_train.columns[cols_idxs].tolist()

print(f"\nChi-Squared Selected Top {k_features} Features:")
print(selected_cols_chi2)

# 4. TRAIN SELF-SUPERVISED MODEL WITH CHI2 FEATURES
# Note: passing the *Original* X_train values (not scaled) to XGBoost is fine,
# but passing the Scaled ones is also fine. XGBoost is invariant to scaling.
train_self_supervised_xgboost(
    X_train[selected_cols_chi2], y_train,
    X_test[selected_cols_chi2], y_test,
    method_name="Chi-Squared + Self-Training"
)


######################################################################################################

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.neighbors import NearestNeighbors

class ClusterOverSampler:
    """
    Clustering-Based Oversampling:
    1. Clusters the Minority Class into K groups.
    2. Upsamples EACH cluster independently to ensure diversity.
    3. Handles small clusters by switching from SMOTE to Random Oversampling.
    """
    def __init__(self, n_clusters=5, sampling_strategy=0.5, random_state=42):
        self.n_clusters = n_clusters
        self.sampling_strategy = sampling_strategy # Ratio of Minority to Majority (e.g., 0.5 = 1:2)
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        # Ensure input is DataFrame/Series for easier handling
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        print(f"--- Starting Cluster-Based Oversampling (K={self.n_clusters}) ---")
        
        # 1. Separate Classes
        X_maj = X[y == 0]
        y_maj = y[y == 0]
        
        X_min = X[y == 1]
        y_min = y[y == 1]
        
        print(f"Original Counts -> Majority: {len(X_maj)}, Minority: {len(X_min)}")
        
        # 2. Cluster the Minority Class
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_min)
        
        # 3. Calculate Target Size
        # Total target minority count
        target_total_minority = int(len(X_maj) * self.sampling_strategy)
        # Target per cluster (Uniform distribution to boost rare sub-groups)
        target_per_cluster = int(target_total_minority / self.n_clusters)
        
        print(f"Targeting ~{target_per_cluster} samples per cluster (Total Min: {target_total_minority})")
        
        X_resampled_list = [X_maj]
        y_resampled_list = [y_maj]
        
        # 4. Iterate and Upsample Each Cluster
        for i in range(self.n_clusters):
            # Get data for this cluster
            cluster_mask = (cluster_labels == i)
            X_cluster = X_min[cluster_mask]
            y_cluster = y_min[cluster_mask]
            
            n_samples = len(X_cluster)
            
            if n_samples == 0:
                continue
                
            # Define Sampler
            # If cluster is too small for SMOTE (needs neighbors), use Random Duplication
            if n_samples < 6:
                sampler = RandomOverSampler(
                    sampling_strategy={1: target_per_cluster},
                    random_state=self.random_state
                )
                method = "Random"
            else:
                # Use SMOTE but be careful with k_neighbors
                k_neighbors = min(n_samples - 1, 5)
                sampler = SMOTE(
                    sampling_strategy={1: target_per_cluster},
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                )
                method = "SMOTE"
            
            # Upsample
            # We must create a dummy majority class to satisfy imblearn API, then discard it
            # Or simpler: just use fit_resample on the cluster itself by faking a target
            # Hack: Append 1 fake majority row to make SMOTE work, then remove it
            try:
                X_cls_res, y_cls_res = sampler.fit_resample(X_cluster, y_cluster)
                
                # Append to list
                X_resampled_list.append(X_cls_res)
                y_resampled_list.append(y_cls_res)
                # print(f"  Cluster {i}: {n_samples} -> {len(X_cls_res)} ({method})")
            except Exception as e:
                print(f"  Cluster {i} failed to resample: {e}. Keeping original.")
                X_resampled_list.append(X_cluster)
                y_resampled_list.append(y_cluster)

        # 5. Combine All
        X_final = pd.concat(X_resampled_list, axis=0)
        y_final = pd.concat(y_resampled_list, axis=0)
        
        # Shuffle
        X_final = X_final.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        y_final = y_final.loc[X_final.index].reset_index(drop=True)
        
        print(f"Final Counts -> Majority: {sum(y_final==0)}, Minority: {sum(y_final==1)}")
        return X_final, y_final



# ---------------------------------------------------------
# EXECUTION: CLUSTER-BASED OVERSAMPLING
# ---------------------------------------------------------

# 1. Initialize Sampler
# n_clusters=5: Good balance for finding distinct applicant types
# sampling_strategy=0.5: Minority will be 50% of Majority size (1:2 Ratio)
cluster_sampler = ClusterOverSampler(n_clusters=5, sampling_strategy=0.5)

# 2. Resample the Training Data (NEVER RESAMPLE TEST DATA)
# Ensure X_train and y_train are from your previous split
X_train_clus, y_train_clus = cluster_sampler.fit_resample(X_train, y_train)

# 3. Train XGBoost on New Data
print("\n--- Training XGBoost on Clustered Data ---")

# Note: We reduce scale_pos_weight because we have manually balanced the data
# If ratio is 1:2 (0.5), scale_pos_weight should be ~2.0
clf_cluster = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    scale_pos_weight=2.0,  # Adjusted for new 1:2 balance
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='aucpr',
    n_jobs=-1,
    random_state=42
)

clf_cluster.fit(X_train_clus, y_train_clus)

# 4. Evaluate on Original Test Set (X_test is untouched)
probs_cluster = clf_cluster.predict_proba(X_test)[:, 1]
from sklearn.metrics import classification_report, roc_auc_score

print(f"\nROC-AUC Score: {roc_auc_score(y_test, probs_cluster):.4f}")

# Check Precision/Recall at decision threshold
preds_cluster = (probs_cluster >= 0.5).astype(int)
print(classification_report(y_test, preds_cluster))


