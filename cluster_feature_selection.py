# ============================================================================
# FEATURE SELECTION ANALYSIS
# Methods: RFE, Random Forest Feature Importance, Correlation Analysis
# ============================================================================

# %% [markdown]
# # Feature Selection Analysis
# This notebook performs feature selection using three methods:
# 1. Recursive Feature Elimination (RFE)
# 2. Random Forest Feature Importance
# 3. Correlation with Target
# 
# Target: Applied (1 if APPLICATION_STATUS is not null, 0 otherwise)

# %% CELL 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# %% CELL 2: Create Target Variable
"""
Create binary target variable 'applied':
- 1 if APPLICATION_STATUS is not null
- 0 if APPLICATION_STATUS is null
"""

# Create target variable
df['applied'] = df['APPLICATION_STATUS'].notna().astype(int)

# Check target distribution
print("Target Variable Distribution:")
print(df['applied'].value_counts())
print(f"\nTarget Balance: {df['applied'].mean():.2%} applied")

# %% CELL 3: Prepare Features for Analysis
"""
Prepare the dataset by:
- Selecting only engineered features (df_eng_t_selected)
- Handling missing values
- Separating features and target
"""

# Use the engineered features
X = df_eng_t_selected.copy()
y = df['applied'].copy()

# Check for missing values
print("Missing values in features:")
print(X.isnull().sum().sum())

# Fill missing values with median (for numerical features)
X_filled = X.fillna(X.median())

# Get feature names
feature_names = X_filled.columns.tolist()

print(f"\nTotal features: {len(feature_names)}")
print(f"Total samples: {len(X_filled)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# %% CELL 4: Scale Features
"""
Standardize features for better performance in RFE and correlation analysis
"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=X_filled.index)

print("Features scaled successfully!")
print(f"Scaled features shape: {X_scaled_df.shape}")

# %% CELL 5: Method 1 - Recursive Feature Elimination (RFE)
"""
Use RFE with Logistic Regression to select top 10 features
"""

print("="*70)
print("METHOD 1: RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*70)

# Initialize RFE with Logistic Regression
estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)

# Fit RFE
rfe.fit(X_scaled_df, y)

# Get selected features
rfe_features = X_scaled_df.columns[rfe.support_].tolist()
rfe_ranking = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print("\nTop 10 Features selected by RFE:")
print("-" * 70)
for i, feature in enumerate(rfe_features, 1):
    print(f"{i:2d}. {feature}")

print("\nFull RFE Rankings (Top 20):")
print(rfe_ranking.head(20))

# %% CELL 6: Method 2 - Random Forest Feature Importance
"""
Use Random Forest to rank features by importance and select top 10
"""

print("\n" + "="*70)
print("METHOD 2: RANDOM FOREST FEATURE IMPORTANCE")
print("="*70)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_filled, y)

# Get feature importances
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Get top 10 features
rf_features = rf_importance.head(10)['Feature'].tolist()

print("\nTop 10 Features by Random Forest Importance:")
print("-" * 70)
for i, row in rf_importance.head(10).iterrows():
    print(f"{list(rf_importance.head(10).index).index(i)+1:2d}. {row['Feature']:50s} | Importance: {row['Importance']:.4f}")

# Visualize top 20 features
plt.figure(figsize=(12, 8))
top_20 = rf_importance.head(20)
plt.barh(range(len(top_20)), top_20['Importance'])
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features - Random Forest Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nRandom Forest importance plot saved!")

# %% CELL 7: Method 3 - Correlation with Target
"""
Calculate correlation between each feature and target, select top 10
"""

print("\n" + "="*70)
print("METHOD 3: CORRELATION WITH TARGET")
print("="*70)

# Calculate correlation with target
correlations = []
for col in X_filled.columns:
    corr = X_filled[col].corr(y)
    correlations.append({
        'Feature': col,
        'Correlation': corr,
        'Abs_Correlation': abs(corr)
    })

correlation_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)

# Get top 10 features by absolute correlation
corr_features = correlation_df.head(10)['Feature'].tolist()

print("\nTop 10 Features by Correlation with Target:")
print("-" * 70)
for i, row in correlation_df.head(10).iterrows():
    print(f"{list(correlation_df.head(10).index).index(i)+1:2d}. {row['Feature']:50s} | Corr: {row['Correlation']:7.4f}")

# Visualize top 20 correlations
plt.figure(figsize=(12, 8))
top_20_corr = correlation_df.head(20)
colors = ['green' if x > 0 else 'red' for x in top_20_corr['Correlation']]
plt.barh(range(len(top_20_corr)), top_20_corr['Correlation'], color=colors)
plt.yticks(range(len(top_20_corr)), top_20_corr['Feature'])
plt.xlabel('Correlation with Target')
plt.title('Top 20 Features - Correlation with Target')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/correlation_with_target.png', dpi=300, bbox_inches='tight')
print("\nCorrelation plot saved!")

# %% CELL 8: Combine Results from All Methods
"""
Combine results from all three methods and analyze overlaps
"""

print("\n" + "="*70)
print("COMBINED FEATURE SELECTION RESULTS")
print("="*70)

# Create summary dataframe
all_features = list(set(rfe_features + rf_features + corr_features))
summary = pd.DataFrame({'Feature': all_features})

# Add selection flags
summary['Selected_by_RFE'] = summary['Feature'].isin(rfe_features).astype(int)
summary['Selected_by_RF'] = summary['Feature'].isin(rf_features).astype(int)
summary['Selected_by_Corr'] = summary['Feature'].isin(corr_features).astype(int)
summary['Selection_Count'] = summary[['Selected_by_RFE', 'Selected_by_RF', 'Selected_by_Corr']].sum(axis=1)

# Add rankings and scores
summary = summary.merge(
    rfe_ranking[['Feature', 'Ranking']].rename(columns={'Ranking': 'RFE_Rank'}),
    on='Feature', how='left'
)
summary = summary.merge(
    rf_importance[['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'}),
    on='Feature', how='left'
)
summary = summary.merge(
    correlation_df[['Feature', 'Correlation', 'Abs_Correlation']],
    on='Feature', how='left'
)

# Sort by selection count and importance
summary = summary.sort_values(['Selection_Count', 'RF_Importance'], ascending=[False, False])

print("\nFeature Selection Summary:")
print(summary.to_string(index=False))

# Features selected by all three methods
unanimous = summary[summary['Selection_Count'] == 3]['Feature'].tolist()
print(f"\n{'='*70}")
print(f"Features Selected by ALL THREE methods ({len(unanimous)}):")
print(f"{'='*70}")
for i, feat in enumerate(unanimous, 1):
    print(f"{i}. {feat}")

# Features selected by at least two methods
majority = summary[summary['Selection_Count'] >= 2]['Feature'].tolist()
print(f"\n{'='*70}")
print(f"Features Selected by AT LEAST TWO methods ({len(majority)}):")
print(f"{'='*70}")
for i, feat in enumerate(majority, 1):
    count = summary[summary['Feature'] == feat]['Selection_Count'].values[0]
    methods = []
    if summary[summary['Feature'] == feat]['Selected_by_RFE'].values[0]:
        methods.append('RFE')
    if summary[summary['Feature'] == feat]['Selected_by_RF'].values[0]:
        methods.append('RF')
    if summary[summary['Feature'] == feat]['Selected_by_Corr'].values[0]:
        methods.append('Corr')
    print(f"{i:2d}. {feat:50s} | Methods: {', '.join(methods)}")

# Save summary
summary.to_csv('/mnt/user-data/outputs/feature_selection_summary.csv', index=False)
print("\nSummary saved to 'feature_selection_summary.csv'")

# %% CELL 9: Visualize Method Overlaps (Venn-style)
"""
Visualize the overlap between the three feature selection methods
"""

print("\n" + "="*70)
print("METHOD OVERLAP ANALYSIS")
print("="*70)

# Count overlaps
only_rfe = len(set(rfe_features) - set(rf_features) - set(corr_features))
only_rf = len(set(rf_features) - set(rfe_features) - set(corr_features))
only_corr = len(set(corr_features) - set(rfe_features) - set(rf_features))

rfe_rf = len(set(rfe_features) & set(rf_features) - set(corr_features))
rfe_corr = len(set(rfe_features) & set(corr_features) - set(rf_features))
rf_corr = len(set(rf_features) & set(corr_features) - set(rfe_features))

all_three = len(set(rfe_features) & set(rf_features) & set(corr_features))

print(f"\nOverlap Statistics:")
print(f"  Only RFE: {only_rfe}")
print(f"  Only RF: {only_rf}")
print(f"  Only Correlation: {only_corr}")
print(f"  RFE + RF: {rfe_rf}")
print(f"  RFE + Correlation: {rfe_corr}")
print(f"  RF + Correlation: {rf_corr}")
print(f"  All Three: {all_three}")

# Create overlap visualization
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['RFE', 'Random Forest', 'Correlation']
counts = [len(rfe_features), len(rf_features), len(corr_features)]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = ax.bar(methods, counts, color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Features Selected', fontsize=12)
ax.set_title('Feature Count by Selection Method (Top 10 Each)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 12)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/method_comparison.png', dpi=300, bbox_inches='tight')
print("\nMethod comparison plot saved!")

# %% CELL 10: Correlation Matrix of Selected Features
"""
Analyze correlation between features selected by at least 2 methods
"""

print("\n" + "="*70)
print("CORRELATION MATRIX OF SELECTED FEATURES")
print("="*70)

# Get features selected by at least 2 methods
selected_features = majority

print(f"\nAnalyzing correlation for {len(selected_features)} features selected by ≥2 methods")

# Create correlation matrix
correlation_matrix = X_filled[selected_features].corr()

# Find highly correlated pairs (|correlation| > 0.7)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
            high_corr_pairs.append({
                'Feature_1': correlation_matrix.columns[i],
                'Feature_2': correlation_matrix.columns[j],
                'Correlation': corr_value
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                          key=lambda x: abs(x), 
                                                          ascending=False)

print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
print("-" * 70)
if len(high_corr_df) > 0:
    print(high_corr_df.to_string(index=False))
else:
    print("No highly correlated pairs found!")

# Visualize correlation matrix
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Features Selected by ≥2 Methods', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/correlation_matrix_selected_features.png', 
            dpi=300, bbox_inches='tight')
print("\nCorrelation matrix heatmap saved!")

# %% CELL 11: Alternative - Correlation Matrix for ALL Selected Features
"""
Show correlation matrix for all unique features from any method
"""

print("\n" + "="*70)
print("CORRELATION MATRIX - ALL SELECTED FEATURES (ANY METHOD)")
print("="*70)

all_selected = list(set(rfe_features + rf_features + corr_features))
print(f"\nTotal unique features from all methods: {len(all_selected)}")

# Create correlation matrix for all selected features
correlation_matrix_all = X_filled[all_selected].corr()

# Find highly correlated pairs
high_corr_pairs_all = []
for i in range(len(correlation_matrix_all.columns)):
    for j in range(i+1, len(correlation_matrix_all.columns)):
        corr_value = correlation_matrix_all.iloc[i, j]
        if abs(corr_value) > 0.7:
            high_corr_pairs_all.append({
                'Feature_1': correlation_matrix_all.columns[i],
                'Feature_2': correlation_matrix_all.columns[j],
                'Correlation': corr_value
            })

high_corr_df_all = pd.DataFrame(high_corr_pairs_all).sort_values('Correlation',
                                                                   key=lambda x: abs(x),
                                                                   ascending=False)

print("\nHighly Correlated Pairs among ALL selected features (|corr| > 0.7):")
print("-" * 70)
if len(high_corr_df_all) > 0:
    print(high_corr_df_all.to_string(index=False))
else:
    print("No highly correlated pairs found!")

# Visualize
plt.figure(figsize=(20, 18))
mask = np.triu(np.ones_like(correlation_matrix_all, dtype=bool))
sns.heatmap(correlation_matrix_all,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7})
plt.title('Correlation Matrix - All Features from Any Method',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/correlation_matrix_all_selected.png',
            dpi=300, bbox_inches='tight')
print("\nAll-features correlation matrix saved!")

# %% CELL 12: Final Recommendations
"""
Provide final feature selection recommendations
"""

print("\n" + "="*70)
print("FINAL FEATURE SELECTION RECOMMENDATIONS")
print("="*70)

# Recommended features (selected by at least 2 methods)
recommended_features = majority.copy()

# Remove highly correlated features to avoid multicollinearity
if len(high_corr_df) > 0:
    print("\nHandling Multicollinearity:")
    print("-" * 70)
    features_to_remove = set()
    
    for _, row in high_corr_df.iterrows():
        feat1, feat2 = row['Feature_1'], row['Feature_2']
        
        # Keep the feature with higher RF importance
        imp1 = summary[summary['Feature'] == feat1]['RF_Importance'].values[0]
        imp2 = summary[summary['Feature'] == feat2]['RF_Importance'].values[0]
        
        if imp1 < imp2:
            features_to_remove.add(feat1)
            print(f"  Removing '{feat1}' (corr={row['Correlation']:.3f} with '{feat2}')")
        else:
            features_to_remove.add(feat2)
            print(f"  Removing '{feat2}' (corr={row['Correlation']:.3f} with '{feat1}')")
    
    recommended_features = [f for f in recommended_features if f not in features_to_remove]

print(f"\n{'='*70}")
print(f"FINAL RECOMMENDED FEATURES: {len(recommended_features)}")
print(f"{'='*70}")

final_recommendations = summary[summary['Feature'].isin(recommended_features)].copy()
final_recommendations = final_recommendations.sort_values(['Selection_Count', 'RF_Importance'], 
                                                          ascending=[False, False])

print("\nRanked by Selection Count and RF Importance:")
print(final_recommendations[['Feature', 'Selection_Count', 'RFE_Rank', 
                            'RF_Importance', 'Correlation']].to_string(index=False))

# Save final recommendations
final_recommendations.to_csv('/mnt/user-data/outputs/final_recommended_features.csv', index=False)
print("\nFinal recommendations saved to 'final_recommended_features.csv'")

# Save feature list
with open('/mnt/user-data/outputs/recommended_features_list.txt', 'w') as f:
    f.write("FINAL RECOMMENDED FEATURES\n")
    f.write("="*70 + "\n\n")
    for i, feat in enumerate(final_recommendations['Feature'], 1):
        f.write(f"{i}. {feat}\n")

print("Feature list saved to 'recommended_features_list.txt'")

# %% CELL 13: Summary Statistics
"""
Print comprehensive summary statistics
"""

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nOriginal number of features: {len(feature_names)}")
print(f"\nFeature Selection Results:")
print(f"  - RFE selected: {len(rfe_features)} features")
print(f"  - Random Forest selected: {len(rf_features)} features")
print(f"  - Correlation selected: {len(corr_features)} features")
print(f"  - Unique features (any method): {len(all_selected)} features")
print(f"  - Features selected by ≥2 methods: {len(majority)} features")
print(f"  - Features selected by all 3 methods: {len(unanimous)} features")

print(f"\nFinal Recommendations:")
print(f"  - After removing highly correlated: {len(recommended_features)} features")
print(f"  - Feature reduction: {len(feature_names)} → {len(recommended_features)} "
      f"({(1 - len(recommended_features)/len(feature_names))*100:.1f}% reduction)")

print(f"\nTarget Variable:")
print(f"  - Total samples: {len(y)}")
print(f"  - Positive class (applied=1): {y.sum()} ({y.mean()*100:.2f}%)")
print(f"  - Negative class (applied=0): {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.2f}%)")

print("\n" + "="*70)
print("FEATURE SELECTION ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. feature_selection_summary.csv")
print("  2. final_recommended_features.csv")
print("  3. recommended_features_list.txt")
print("  4. rf_feature_importance.png")
print("  5. correlation_with_target.png")
print("  6. method_comparison.png")
print("  7. correlation_matrix_selected_features.png")
print("  8. correlation_matrix_all_selected.png")

# ============================================================================
# END OF FEATURE SELECTION ANALYSIS
# ============================================================================
