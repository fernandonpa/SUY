import numpy as np
import xgboost as xgb
from scipy.special import expit  # This is the stable Sigmoid function

# ---------------------------------------------------------
# 1. DEFINE FOCAL LOSS OBJECTIVE
# ---------------------------------------------------------
def focal_binary(y_pred, dtrain):
    """
    Custom Focal Loss for XGBoost.
    Gamma: Focusing parameter (default 2.0). Higher = focuses more on hard examples.
    Alpha: Balancing parameter. equivalent to scale_pos_weight logic.
    """
    y_true = dtrain.get_label()
    
    # Tunable parameters
    gamma = 2.0   # Focus on hard samples (2.0 is standard)
    alpha = 0.95  # Handling Imbalance (0.90 to 0.99 for extreme imbalance)
    
    # Sigmoid transformation (since XGBoost outputs raw logits)
    sigmoid_p = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Calculate gradients and hessians
    pt = y_true * sigmoid_p + (1 - y_true) * (1 - sigmoid_p)
    class_mask = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    # Gradient (First Derivative)
    grad = (sigmoid_p - y_true) * class_mask * (1 - pt)**gamma
    
    # Add Focal Term correction to gradient
    grad += class_mask * gamma * (1 - pt)**(gamma - 1) * \
            (2 * pt - 1) * (sigmoid_p * (1 - sigmoid_p) * np.log(np.maximum(pt, 1e-15)))

    # Hessian (Second Derivative)
    # Approximation often works well enough for trees
    hess = class_mask * sigmoid_p * (1 - sigmoid_p) * (1 - pt)**gamma
    
    # To stabilize Hessian (prevent division by zero in tree split)
    hess = np.maximum(hess, 1e-6) 
    
    return grad, hess

# ---------------------------------------------------------
# 2. CONFIGURE MODEL
# ---------------------------------------------------------
# Note: We REMOVE 'scale_pos_weight' because we handle balance via 'alpha' above
xgb_params_focal = {
    'max_delta_step': 1,        # Critical for imbalance
    'learning_rate': 0.02,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'aucpr',     # Metric for validation monitoring
    'n_jobs': -1,
    'disable_default_eval_metric': 1 # Prevent default metric conflicts
}

print("Training XGBoost with Custom Focal Loss...")

# 3. TRAIN USING LOW-LEVEL API (Recommended for Custom Objectives)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model_focal = xgb.train(
    params=xgb_params_focal,
    dtrain=dtrain,
    num_boost_round=500,
    obj=focal_binary,           # <--- Insert Custom Loss Here
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=50
)

# ---------------------------------------------------------
# 4. PREDICT (CRITICAL STEP)
# ---------------------------------------------------------
# Custom objectives return 'logits' (raw scores), NOT probabilities.
# You MUST apply Sigmoid manually.

# Raw Logits
y_pred_logits = model_focal.predict(dtest)

# Convert to Probability (0 to 1)
y_pred_proba = expit(y_pred_logits)

print("\nFocal Loss Training Complete.")
print(f"Prediction Range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
