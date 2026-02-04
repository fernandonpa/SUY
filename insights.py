import pandas as pd
import numpy as np

def get_cluster_deep_insights(df_original, cluster_labels):
    """
    Calculates detailed business metrics for each cluster using the ORIGINAL data.
    """
    # 1. Setup: Merge Clusters back to Raw Data
    df = df_original.copy()
    df['Cluster'] = cluster_labels
    
    # 2. Define Helper for Categorical Mode (Most common value)
    def get_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else "Unknown"
    
    # --- PRE-CALCULATIONS FOR CUSTOM METRICS ---
    
    # A. Activity Flags (Convert Y/N or Counts to 0/1 for percentages)
    # Adjust logic if your data uses different codes (e.g., '1' or 'Yes')
    df['is_online'] = df['LTM_ONLINE_ACTIVE'].apply(lambda x: 1 if x in ['Y', 1] else 0)
    df['is_email'] = df['LTM_EMAIL_ACTIVE'].apply(lambda x: 1 if x in ['Y', 1] else 0)
    df['is_txn_active'] = (df['LTM_TRAN_ACTIVE'] > 0).astype(int)
    
    # B. Application Status Flags
    # "Applied" = Status is Approved OR Denied (based on your definition)
    df['is_approved'] = (df['APPLICATION_STATUS'] == 'Approved').astype(int)
    df['is_denied'] = (df['APPLICATION_STATUS'] == 'Denied').astype(int)
    df['is_applied'] = df['is_approved'] + df['is_denied']
    
    # 3. AGGREGATION
    # We group by Cluster and calculate Mean (for amts) or Sum (for counts)
    summary = df.groupby('Cluster').agg({
        # 1. Redeems
        'LTM_RDM': 'mean',
        'TOTAL_RDM_SINCE_2020': 'mean',
        
        # 2. Spend
        'LAST_3MTH_SPEND': 'mean',
        'LAST_12MTH_SPEND': 'mean',
        
        # 3. Earns
        'LTM_DLR_EARNED': 'mean',
        'TOTAL_EARNED_SINCE_2020': 'mean',
        
        # 4. Balance
        'POINTS_BALANCE': 'mean',
        
        # 5/6. Demographics (Distributions - Getting the Mode)
        'AGE_RANGE': get_mode,
        'CENSUS_INCOME_RANGE': get_mode,
        
        # 7. Activity Percentages (Mean of 0/1 gives %)
        'is_online': 'mean',
        'is_txn_active': 'mean',
        'is_email': 'mean',
        
        # 8. Application Base Counts (for derived calc)
        'is_applied': 'sum',
        'is_approved': 'sum',
        'LYL_ID_NO': 'count' # Total members in cluster
    }).rename(columns={'LYL_ID_NO': 'Cluster_Size'})
    
    # 4. DERIVED METRICS (The complex percentages)
    
    # A. Applied % (Total Applied / Total Cluster Members)
    summary['% Applied (of Cluster)'] = (summary['is_applied'] / summary['Cluster_Size']) * 100
    
    # B. Approved % (Approved / Total Applied) - "Approval Rate"
    # Fillna(0) handles clusters where nobody applied (div by zero)
    summary['% Approved (of Applicants)'] = (summary['is_approved'] / summary['is_applied']).fillna(0) * 100
    
    # C. Approved % (Approved / Total Cluster Members) - "Conversion Rate"
    summary['% Approved (of Cluster)'] = (summary['is_approved'] / summary['Cluster_Size']) * 100
    
    # Format Percentages for Readability
    activity_cols = ['is_online', 'is_txn_active', 'is_email']
    for col in activity_cols:
        summary[col] = summary[col] * 100  # Convert 0.5 to 50.0
        
    # Cleanup / Reordering
    final_view = summary[[
        'Cluster_Size',
        'AGE_RANGE', 'CENSUS_INCOME_RANGE',
        'LAST_12MTH_SPEND', 'POINTS_BALANCE',
        'is_online', 'is_email',
        '% Applied (of Cluster)', '% Approved (of Applicants)'
    ]].round(2)
    
    return final_view, summary

# Usage:
# insights_table, raw_stats = get_cluster_deep_insights(df_train, labels)
# display(insights_table)



import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_demographics(df, labels, cluster_id):
    """
    Plots Age & Income distribution for a SINGLE cluster vs Population Average.
    """
    data = df.copy()
    data['Cluster'] = labels
    
    # Filter Data
    cluster_data = data[data['Cluster'] == cluster_id]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Age Distribution
    # Calculate % of each age group in Population vs Cluster
    pop_age = data['AGE_RANGE'].value_counts(normalize=True).sort_index()
    clus_age = cluster_data['AGE_RANGE'].value_counts(normalize=True).sort_index()
    
    df_plot = pd.DataFrame({'Population': pop_age, f'Cluster {cluster_id}': clus_age})
    df_plot.plot(kind='bar', ax=axes[0], color=['gray', 'blue'], alpha=0.8)
    axes[0].set_title(f'Age Distribution: Cluster {cluster_id} vs Pop')
    axes[0].set_ylabel('Percentage')
    
    # 2. Income Distribution
    pop_inc = data['CENSUS_INCOME_RANGE'].value_counts(normalize=True).sort_index()
    clus_inc = cluster_data['CENSUS_INCOME_RANGE'].value_counts(normalize=True).sort_index()
    
    df_plot2 = pd.DataFrame({'Population': pop_inc, f'Cluster {cluster_id}': clus_inc})
    df_plot2.plot(kind='bar', ax=axes[1], color=['gray', 'green'], alpha=0.8)
    axes[1].set_title(f'Income Distribution: Cluster {cluster_id} vs Pop')
    
    plt.tight_layout()
    plt.show()

# Usage:
# plot_cluster_demographics(df_train, labels, cluster_id=2)
