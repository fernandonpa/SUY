pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost imbalanced-learn scipy joblib

# Core data science libraries
pip install pandas numpy scipy

# Visualization
pip install matplotlib seaborn

# Machine Learning - scikit-learn (includes most ML models and metrics)
pip install scikit-learn

# Gradient Boosting libraries
pip install xgboost
pip install lightgbm
pip install catboost

# Imbalanced learning
pip install imbalanced-learn

# Model persistence (if not already included)
pip install joblib


#################################################
# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models (NO Deep Learning - Explainability required)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Imbalance Handling
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Feature Selection
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, mutual_info_classif, f_classif
)

# Preprocessing & Metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, 
    average_precision_score, f1_score,
    precision_score, recall_score, roc_curve
)

# Calibration
from sklearn.calibration import CalibratedClassifierCV

# Cost-sensitive learning
from sklearn.utils.class_weight import compute_class_weight

# Set random seed
np.random.seed(42)
pd.set_option('display.max_columns', None)

print("Libraries loaded successfully")


#####################################

# Load data
df = pd.read_parquet('your_data_path.parquet')

# Create binary target (Applied = 1, Not Applied = 0)
df['target_applied'] = df['APPLICATION_STATUS'].notna().astype(int)

# Filter to winback population only (those who received email)
df = df[df['WINBACK_POPULATION'] == 1].copy()

print(f"Dataset shape: {df.shape}")
print(f"\nClass Distribution:")
print(df['target_applied'].value_counts())
print(f"\nImbalance Ratio: 1:{(df['target_applied']==0).sum() / (df['target_applied']==1).sum():.1f}")
print(f"Applied Rate: {df['target_applied'].mean():.4f}")

##############################################3

def engineer_supervised_features(df):
    """
    Feature engineering for supervised learning
    EXCLUDES: Age and Location features
    FOCUS: Behavioral patterns that predict application
    """
    data = df.copy()
    
    # --- SAFETY PREPROCESSING ---
    safe_earn = data['TOTAL_EARNED_SINCE_2020'].clip(lower=0)
    safe_spend_12m = data['LAST_12MTH_SPEND'].clip(lower=0)
    safe_spend_3m = data['LAST_3MTH_SPEND'].clip(lower=0)
    safe_spend_36m = data['LAST_36MTH_SPEND'].clip(lower=0)
    safe_rdm = data['TOTAL_RDM_SINCE_2020'].clip(lower=0)
    safe_ltm_rdm = data['LTM_RDM'].clip(lower=0)
    
    # ========== 1. RECENCY FEATURES (CRITICAL for Response) ==========
    
    # Activity flags
    data['flag_active_3m'] = (safe_spend_3m > 0).astype(int)
    data['flag_active_12m'] = (safe_spend_12m > 0).astype(int)
    data['flag_active_36m'] = (safe_spend_36m > 0).astype(int)
    
    # Dormancy detection
    data['flag_dormant'] = (
        (safe_spend_12m == 0) & (safe_spend_36m > 0)
    ).astype(int)
    
    # Reactivation potential (was active, stopped, came back)
    data['flag_reactivated'] = (
        (safe_spend_3m > 0) & 
        (data['flag_dormant'] == 1)
    ).astype(int)
    
    # ========== 2. MONETARY FEATURES (Log + Binned) ==========
    
    # Log transformations (winsorized)
    from scipy.stats import mstats
    
    for col in ['LAST_12MTH_SPEND', 'LAST_3MTH_SPEND', 
                'POINTS_BALANCE', 'LTM_DLR_EARNED']:
        if col in data.columns:
            winsorized = mstats.winsorize(data[col], limits=[0, 0.05])
            data[f'log_{col}'] = np.log1p(np.clip(winsorized, 0, None))
    
    # Spend percentiles (binning for non-linearity)
    data['spend_12m_percentile'] = pd.qcut(
        safe_spend_12m, q=10, labels=False, duplicates='drop'
    )
    
    # ========== 3. FREQUENCY/ENGAGEMENT FEATURES ==========
    
    # Transaction intensity
    data['txn_intensity'] = data['LTM_TRAN_ACTIVE'] / (data['D_MONTHONBOOK'] + 1)
    
    # Email engagement composite
    data['email_engagement_score'] = (
        data['LTM_EMAIL_ACTIVE'] * 0.5 + 
        data['L18M_EMAIL_ACTIVE'] * 0.3 + 
        data['L24M_EMAIL_ACTIVE'] * 0.2
    )
    
    # Digital activity flag
    if data['LTM_ONLINE_ACTIVE'].dtype == 'object':
        data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE'].apply(
            lambda x: 1 if x in ['Y', 1, True] else 0
        )
    else:
        data['flag_digital_active'] = data['LTM_ONLINE_ACTIVE']
    
    # Multi-channel engagement
    data['flag_omnichannel'] = (
        (data['flag_digital_active'] == 1) & 
        (data['LTM_TRAN_ACTIVE'] > 0)
    ).astype(int)
    
    # ========== 4. BEHAVIORAL RATIOS ==========
    
    # Burn rate (redemption behavior)
    data['ratio_burn_rate'] = safe_rdm / (safe_earn + 1)
    
    # Active burner (earns AND redeems)
    data['flag_active_burner'] = (
        (data['LTM_DLR_EARNED'] > 0) & 
        (safe_ltm_rdm > 0)
    ).astype(int)
    
    # Redemption recency
    data['flag_recent_redeemer'] = (safe_ltm_rdm > 0).astype(int)
    
    # Redemption to spend ratio
    data['ratio_rdm_to_spend'] = safe_ltm_rdm / (safe_spend_12m + 1)
    
    # ========== 5. TREND/VELOCITY FEATURES ==========
    
    # Spend acceleration
    data['ratio_spend_velocity'] = (safe_spend_3m * 4) / (safe_spend_12m + 1)
    data['flag_spend_increasing'] = (
        (safe_spend_3m * 4) > safe_spend_12m
    ).astype(int)
    
    # Year-over-year comparison
    data['ratio_12m_to_36m_spend'] = safe_spend_12m / ((safe_spend_36m / 3) + 1)
    
    # ========== 6. ENGAGEMENT × CAPACITY INTERACTIONS ==========
    
    # High engagement + high capacity
    data['engage_x_income'] = (
        data['MES_ENG_SCORE'] * data['INCOME_RANGE_ENC']
    )
    
    # Spend to income ratio (spending power utilization)
    data['spend_to_income_ratio'] = (
        safe_spend_12m / (data['INCOME_RANGE_ENC'] + 1)
    )
    
    # Engagement × spending
    data['engage_x_spend'] = (
        data['MES_ENG_SCORE'] * np.log1p(safe_spend_12m)
    )
    
    # ========== 7. LOYALTY MATURITY FEATURES ==========
    
    # Tenure segments
    data['tenure_segment'] = pd.cut(
        data['D_MONTHONBOOK'],
        bins=[0, 6, 12, 24, 36, np.inf],
        labels=['New', 'Growing', 'Established', 'Mature', 'Veteran']
    )
    data['tenure_segment_encoded'] = data['tenure_segment'].cat.codes
    
    # Points accumulation rate
    data['points_per_month'] = data['POINTS_BALANCE'] / (data['D_MONTHONBOOK'] + 1)
    
    # High points holder
    data['flag_high_points'] = (data['POINTS_BALANCE'] > 10000).astype(int)
    
    # ========== 8. RFM-BASED FEATURES ==========
    
    # RFM interaction with email engagement
    data['rfm_x_email'] = (
        data['RFM_SEGMENT_ENC'] * data['email_engagement_score']
    )
    
    # ========== 9. CAPACITY FEATURES (Income-related, NOT age) ==========
    
    # Income adequacy (comparing income to census income)
    data['income_vs_census'] = (
        data['INCOME_RANGE_ENC'] - data['CENSUS_INCOME_RANGE_ENC']
    )
    
    # ========== CLEANUP ==========
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    return data

# Apply feature engineering
df_features = engineer_supervised_features(df)
print("Feature engineering completed")
print(f"New features created: {df_features.shape[1] - df.shape[1]}")

########################################333


# SUPERVISED LEARNING CANDIDATE FEATURES (NO Age/Location)
candidate_features_supervised = [
    # ===== RECENCY (Highest Priority) =====
    'flag_active_3m',
    'flag_active_12m',
    'flag_dormant',
    'flag_reactivated',
    
    # ===== MONETARY =====
    'log_LAST_12MTH_SPEND',
    'log_LAST_3MTH_SPEND',
    'LAST_12MTH_SPEND',
    'LAST_3MTH_SPEND',
    'spend_12m_percentile',
    
    # ===== FREQUENCY/ENGAGEMENT =====
    'MES_ENG_SCORE',
    'email_engagement_score',
    'flag_digital_active',
    'flag_omnichannel',
    'txn_intensity',
    'LTM_TRAN_ACTIVE',
    'LTM_EMAIL_ACTIVE',
    'L24M_EMAIL_ACTIVE',
    
    # ===== REDEMPTION BEHAVIOR =====
    'ratio_burn_rate',
    'flag_active_burner',
    'flag_recent_redeemer',
    'ratio_rdm_to_spend',
    'LTM_RDM',
    'TOTAL_RDM_SINCE_2020',
    'log_POINTS_BALANCE',
    'POINTS_BALANCE',
    'flag_high_points',
    'points_per_month',
    
    # ===== TREND/VELOCITY =====
    'ratio_spend_velocity',
    'flag_spend_increasing',
    'ratio_12m_to_36m_spend',
    'D_YEAR_ON_YEAR_TREND_FOR_SPEND',
    
    # ===== CAPACITY (Income, NOT Age) =====
    'INCOME_RANGE_ENC',
    'CENSUS_INCOME_RANGE_ENC',
    'income_vs_census',
    'spend_to_income_ratio',
    
    # ===== INTERACTIONS =====
    'engage_x_income',
    'engage_x_spend',
    'rfm_x_email',
    
    # ===== LOYALTY MATURITY =====
    'D_MONTHONBOOK',
    'tenure_segment_encoded',
    'RFM_SEGMENT_ENC',
    
    # ===== EARNING =====
    'LTM_DLR_EARNED',
    'log_LTM_DLR_EARNED',
    'TOTAL_EARNED_SINCE_2020'
]

# Verify no age/location features
age_location_keywords = ['AGE', 'STATE', 'CITY', 'ZIP', 'GEOGRAPHIC', 'PRIMARY_STATE']
violations = [f for f in candidate_features_supervised if any(kw in f.upper() for kw in age_location_keywords)]

if violations:
    print(f"⚠️ WARNING: Age/Location features detected: {violations}")
else:
    print("✓ No age/location features - VERIFIED")

print(f"\nCandidate features: {len(candidate_features_supervised)}")

# Create feature matrix
X_full = df_features[candidate_features_supervised].copy()
y = df_features['target_applied'].copy()

print(f"Feature matrix shape: {X_full.shape}")
print(f"Target distribution:\n{y.value_counts()}")


###########################################

from sklearn.feature_selection import mutual_info_classif

def advanced_feature_selection(X, y, n_features=25):
    """
    Uses 4 methods and combines their rankings
    """
    print("Running Advanced Feature Selection...")
    print("="*80)
    
    feature_scores = pd.DataFrame(index=X.columns)
    
    # ===== METHOD 1: XGBoost Feature Importance =====
    print("\n1. XGBoost Feature Importance...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X, y)
    feature_scores['xgb_importance'] = xgb_model.feature_importances_
    feature_scores['xgb_rank'] = feature_scores['xgb_importance'].rank(ascending=False)
    
    # ===== METHOD 2: Mutual Information =====
    print("2. Mutual Information...")
    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    feature_scores['mi_score'] = mi_scores
    feature_scores['mi_rank'] = feature_scores['mi_score'].rank(ascending=False)
    
    # ===== METHOD 3: Random Forest (with class weights) =====
    print("3. Random Forest with Class Weights...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    feature_scores['rf_importance'] = rf_model.feature_importances_
    feature_scores['rf_rank'] = feature_scores['rf_importance'].rank(ascending=False)
    
    # ===== METHOD 4: SelectKBest (f_classif) =====
    print("4. ANOVA F-Score...")
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    feature_scores['f_score'] = selector.scores_
    feature_scores['f_rank'] = feature_scores['f_score'].rank(ascending=False)
    
    # ===== ENSEMBLE RANKING =====
    feature_scores['avg_rank'] = feature_scores[[
        'xgb_rank', 'mi_rank', 'rf_rank', 'f_rank'
    ]].mean(axis=1)
    
    feature_scores = feature_scores.sort_values('avg_rank')
    
    # Select top N features
    top_features = feature_scores.head(n_features).index.tolist()
    
    print(f"\n{'='*80}")
    print(f"Top {n_features} Features Selected")
    print("="*80)
    print(feature_scores.head(n_features)[['avg_rank', 'xgb_importance', 'mi_score', 'rf_importance']])
    
    return top_features, feature_scores

# Select features
top_features, feature_importance_df = advanced_feature_selection(X_full, y, n_features=25)

# Create optimized feature set
X_optimized = X_full[top_features].copy()

print(f"\nOptimized feature set: {X_optimized.shape}")


###############################################

from sklearn.model_selection import train_test_split

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train-Test Split:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"\nTrain distribution:\n{y_train.value_counts()}")
print(f"Test distribution:\n{y_test.value_counts()}")
print(f"\nTrain imbalance ratio: 1:{(y_train==0).sum()/(y_train==1).sum():.1f}")
print(f"Test imbalance ratio: 1:{(y_test==0).sum()/(y_test==1).sum():.1f}")


####################################################


def create_balanced_datasets(X_train, y_train):
    """
    Creates multiple balanced datasets using different strategies
    """
    strategies = {}
    
    print("Creating Balanced Datasets...")
    print("="*80)
    
    # ===== STRATEGY 1: SMOTE + Tomek Links (Remove borderline) =====
    print("\n1. SMOTE + Tomek Links...")
    smt = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5),
        random_state=42
    )
    X_smt, y_smt = smt.fit_resample(X_train, y_train)
    strategies['SMOTE_Tomek'] = (X_smt, y_smt)
    print(f"   Result: {y_smt.value_counts().to_dict()}")
    
    # ===== STRATEGY 2: ADASYN (Adaptive Synthetic) =====
    print("\n2. ADASYN...")
    adasyn = ADASYN(sampling_strategy=0.1, random_state=42, n_neighbors=5)
    X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
    strategies['ADASYN'] = (X_ada, y_ada)
    print(f"   Result: {y_ada.value_counts().to_dict()}")
    
    # ===== STRATEGY 3: BorderlineSMOTE =====
    print("\n3. Borderline SMOTE...")
    bsmote = BorderlineSMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_bs, y_bs = bsmote.fit_resample(X_train, y_train)
    strategies['BorderlineSMOTE'] = (X_bs, y_bs)
    print(f"   Result: {y_bs.value_counts().to_dict()}")
    
    # ===== STRATEGY 4: SMOTE + ENN (More aggressive cleaning) =====
    print("\n4. SMOTE + ENN...")
    smenn = SMOTEENN(
        smote=SMOTE(sampling_strategy=0.1, random_state=42),
        random_state=42
    )
    X_smenn, y_smenn = smenn.fit_resample(X_train, y_train)
    strategies['SMOTE_ENN'] = (X_smenn, y_smenn)
    print(f"   Result: {y_smenn.value_counts().to_dict()}")
    
    # ===== STRATEGY 5: Conservative Undersampling + SMOTE =====
    print("\n5. Undersample (1:20) + SMOTE...")
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('undersample', RandomUnderSampler(sampling_strategy=0.05, random_state=42)),
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42))
    ])
    X_pipe, y_pipe = pipeline.fit_resample(X_train, y_train)
    strategies['Under_SMOTE'] = (X_pipe, y_pipe)
    print(f"   Result: {y_pipe.value_counts().to_dict()}")
    
    # ===== STRATEGY 6: Original (for ensemble methods) =====
    strategies['Original'] = (X_train, y_train)
    
    print("\n" + "="*80)
    print(f"Created {len(strategies)} balanced datasets")
    
    return strategies

# Create all balanced datasets
balanced_datasets = create_balanced_datasets(X_train, y_train)


################################################3

def train_models_all_strategies(balanced_datasets, X_test, y_test):
    """
    Trains multiple models on each balanced dataset
    Focus on Precision AND Recall
    """
    results = []
    
    # Scale positive weight for original data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Define models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=False
        ),
        'BalancedRF': BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1
        ),
        'EasyEnsemble': EasyEnsembleClassifier(
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    print("Training Models on All Strategies...")
    print("="*80)
    
    for strategy_name, (X_bal, y_bal) in balanced_datasets.items():
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy_name}")
        print(f"Training Set Size: {X_bal.shape}")
        print(f"Class Distribution: {dict(pd.Series(y_bal).value_counts())}")
        print("="*80)
        
        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            
            try:
                # Train
                model.fit(X_bal, y_bal)
                
                # Predict probabilities
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Find optimal threshold using PR curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                
                # Predict with optimal threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Metrics
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc_pr = average_precision_score(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results.append({
                    'Strategy': strategy_name,
                    'Model': model_name,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'AUC-PR': auc_pr,
                    'ROC-AUC': roc_auc,
                    'Threshold': optimal_threshold,
                    'model_object': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                })
                
                print(f"    Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['model_object', 'predictions', 'probabilities']}
        for r in results
    ])
    
    return results_df, results

# Train all models
results_df, all_results = train_models_all_strategies(balanced_datasets, X_test, y_test)


#####################################

# Sort by F1 score (balances precision and recall)
results_df_sorted = results_df.sort_values('F1', ascending=False)

print("\n" + "="*100)
print("TOP 10 CONFIGURATIONS (Sorted by F1 Score)")
print("="*100)
print(results_df_sorted.head(10).to_string(index=False))

# Best model
best_config = results_df_sorted.iloc[0]
print(f"\n{'='*100}")
print("BEST CONFIGURATION")
print("="*100)
print(f"Strategy: {best_config['Strategy']}")
print(f"Model: {best_config['Model']}")
print(f"Precision: {best_config['Precision']:.4f}")
print(f"Recall: {best_config['Recall']:.4f}")
print(f"F1 Score: {best_config['F1']:.4f}")
print(f"AUC-PR: {best_config['AUC-PR']:.4f}")
print(f"Optimal Threshold: {best_config['Threshold']:.4f}")
print("="*100)

# Get best model object
best_result = [r for r in all_results if r['Strategy']==best_config['Strategy'] and r['Model']==best_config['Model']][0]
best_model = best_result['model_object']
best_predictions = best_result['predictions']
best_probabilities = best_result['probabilities']



###########################################3



from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "="*100)
print("DETAILED EVALUATION - BEST MODEL")
print("="*100)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, best_predictions, target_names=['Not Applied', 'Applied']))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
print("\nConfusion Matrix:")
print(f"                 Predicted Not Applied  Predicted Applied")
print(f"Actually Not Applied:    {cm[0][0]:>10}          {cm[0][1]:>10}")
print(f"Actually Applied:        {cm[1][0]:>10}          {cm[1][1]:>10}")

# Business Metrics
tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
print(f"\nBusiness Metrics:")
print(f"True Positives (Correctly identified applicants): {tp}")
print(f"False Positives (Wasted targeting): {fp}")
print(f"False Negatives (Missed opportunities): {fn}")
print(f"True Negatives: {tn}")
print(f"\nCapture Rate (of all applicants): {tp/(tp+fn)*100:.2f}%")
print(f"Precision (of predicted applicants): {tp/(tp+fp)*100:.2f}%")

####################################

def calibrate_and_evaluate(best_model, X_bal, y_bal, X_test, y_test, strategy_name):
    """
    Calibrates probabilities for better threshold tuning
    """
    print(f"\nCalibrating {strategy_name}...")
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(
        best_model, 
        method='isotonic', 
        cv=3
    )
    calibrated_model.fit(X_bal, y_bal)
    
    # Predict
    y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_cal)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred_cal = (y_pred_proba_cal >= optimal_threshold).astype(int)
    
    # Metrics
    prec = precision_score(y_test, y_pred_cal)
    rec = recall_score(y_test, y_pred_cal)
    f1 = f1_score(y_test, y_pred_cal)
    
    print(f"Calibrated Metrics:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    
    return calibrated_model, y_pred_proba_cal, y_pred_cal, optimal_threshold

# Get balanced data for best strategy
X_best_bal, y_best_bal = balanced_datasets[best_config['Strategy']]

# Calibrate
calibrated_model, y_pred_proba_cal, y_pred_cal, cal_threshold = calibrate_and_evaluate(
    best_model, X_best_bal, y_best_bal, X_test, y_test, best_config['Strategy']
)


#####################################

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots feature importance from best model
    """
    # Get importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("Model doesn't have feature_importances_")
        return
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance_df['Importance'].head(top_n))
    plt.yticks(range(top_n), importance_df['Feature'].head(top_n))
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances - {best_config["Model"]}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTop {top_n} Features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df

# Plot importance
importance_df = plot_feature_importance(best_model, top_features, top_n=20)


##############################

def plot_pr_curve(y_test, y_pred_proba, model_name):
    """
    Plots Precision-Recall curve
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.4f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"AUC-PR Score: {auc_pr:.4f}")

# Plot PR curve
plot_pr_curve(y_test, best_probabilities, best_config['Model'])

########################################



import joblib

# Save best model
model_path = '/mnt/user-data/outputs/best_model_applied_prediction.pkl'
joblib.dump(best_model, model_path)
print(f"Model saved: {model_path}")

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': best_predictions,
    'probability': best_probabilities,
    'predicted_calibrated': y_pred_cal,
    'probability_calibrated': y_pred_proba_cal
})

# Add to test data
test_indices = y_test.index
df_predictions = df_features.loc[test_indices].copy()
df_predictions['predicted_applied'] = best_predictions
df_predictions['probability_applied'] = best_probabilities

# Export
predictions_path = '/mnt/user-data/outputs/predictions_applied_model.parquet'
df_predictions.to_parquet(predictions_path, index=False)

# Export summary
summary = {
    'Best_Strategy': best_config['Strategy'],
    'Best_Model': best_config['Model'],
    'Precision': best_config['Precision'],
    'Recall': best_config['Recall'],
    'F1_Score': best_config['F1'],
    'AUC_PR': best_config['AUC-PR'],
    'Optimal_Threshold': best_config['Threshold'],
    'Features_Used': len(top_features),
    'Training_Size': X_best_bal.shape[0],
    'Test_Size': X_test.shape[0]
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('/mnt/user-data/outputs/model_summary.csv', index=False)

print("\nFiles exported:")
print(f"  - Model: {model_path}")
print(f"  - Predictions: {predictions_path}")
print(f"  - Summary: /mnt/user-data/outputs/model_summary.csv")
print(f"  - Feature Importance: /mnt/user-data/outputs/feature_importance.png")
print(f"  - PR Curve: /mnt/user-data/outputs/pr_curve.png")

print(f"\n{'='*100}")
print("MODELING COMPLETE")
print("="*100)
print(f"Best Configuration: {best_config['Strategy']} + {best_config['Model']}")
print(f"Final Precision: {best_config['Precision']:.4f}")
print(f"Final Recall: {best_config['Recall']:.4f}")
print(f"Final F1: {best_config['F1']:.4f}")
