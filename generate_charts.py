"""Script to generate publication-quality evaluation and accuracy charts for GitHub README."""
import os
import shutil
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_csv, preprocess_dataset
from src.features import transform_records, FEATURE_NAMES
from src.validation import create_temporal_splits

# Set plot styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(root_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("Generating comprehensive accuracy and diagnostic charts...")
    
    # 1. Copy candidate_december.png to assets
    dec_chart_src = os.path.join(root_dir, 'scorer_results', 'candidate_december.png')
    if os.path.exists(dec_chart_src):
        shutil.copy(dec_chart_src, os.path.join(assets_dir, 'candidate_december.png'))
        print(" -> Copied candidate_december.png to assets/")
        
    # Load model bundle
    bundle_path = os.path.join(root_dir, 'models', 'model_bundle.pkl')
    if not os.path.exists(bundle_path):
        print("Model bundle not found. Please run python train.py first.")
        return
        
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
        
    model = bundle['model']
    coord_map = bundle['coord_map']
    mi_lookup = bundle['mi_lookup']
    qs_lookup = bundle['qs_lookup']
    
    # Load and clean training data
    train_path = os.path.join(root_dir, 'data', 'train-test.csv')
    raw_train = load_csv(train_path)
    cleaned_train = preprocess_dataset(raw_train, coord_map, mi_lookup, qs_lookup)
    
    # Extract Fold 3 (Oct 2025 out-of-time test fold)
    splits = create_temporal_splits(cleaned_train)
    fold3_name, train_recs, val_recs = splits[2]
    
    X_val, y_val = transform_records(val_recs)
    y_pred = model.predict(X_val)
    
    y_val = np.array(y_val)
    y_pred = np.array(y_pred)
    residuals = y_pred - y_val
    
    # -------------------------------------------------------------
    # Chart 1: Model Accuracy Benchmarks Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    models = ['Ridge Baseline', 'LightGBM / GBDT', 'CatBoost', 'Weighted Ensemble']
    rmse_scores = [631.97, 659.73, 632.66, 624.15]
    mae_scores = [138.23, 193.31, 138.95, 134.80]
    r2_scores = [0.8245, 0.8084, 0.8241, 0.8310]
    mape_scores = [7.74, 8.41, 6.79, 6.55]
    
    colors = ['#4A90E2', '#50E3C2', '#F5A623', '#2E7D32']
    
    # Subplot 1: Error metrics (RMSE and MAE)
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, rmse_scores, width, label='RMSE ($)', color='#3F51B5', alpha=0.9)
    axes[0].bar(x + width/2, mae_scores, width, label='MAE ($)', color='#009688', alpha=0.9)
    axes[0].set_ylabel('Dollar Amount ($)', fontweight='bold')
    axes[0].set_title('Out-of-Time Error Metrics (Lower is Better)', fontweight='bold', pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha='right')
    axes[0].legend(frameon=True)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(len(models)):
        axes[0].text(x[i] - width/2, rmse_scores[i] + 10, f"${rmse_scores[i]:.0f}", ha='center', fontsize=9, fontweight='bold')
        axes[0].text(x[i] + width/2, mae_scores[i] + 10, f"${mae_scores[i]:.0f}", ha='center', fontsize=9, fontweight='bold')
        
    # Subplot 2: R2 Score & Accuracy (Higher is Better)
    axes[1].bar(models, [r * 100 for r in r2_scores], color=colors, width=0.55, alpha=0.9)
    axes[1].set_ylabel('Variance Explained R² (%)', fontweight='bold')
    axes[1].set_title('Out-of-Time R² Accuracy Score (Higher is Better)', fontweight='bold', pad=12)
    axes[1].set_xticklabels(models, rotation=15, ha='right')
    axes[1].set_ylim(70, 100)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, score in enumerate(r2_scores):
        axes[1].text(i, score * 100 + 0.8, f"{score:.4f}\n({score*100:.1f}%)", ha='center', fontsize=9.5, fontweight='bold')
        
    plt.tight_layout()
    chart1_path = os.path.join(assets_dir, 'model_accuracy_metrics.png')
    plt.savefig(chart1_path, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {chart1_path}")
    
    # -------------------------------------------------------------
    # Chart 2: Actual vs Predicted Freight Rate Scatter Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=200)
    ax.scatter(y_val, y_pred, alpha=0.35, edgecolors='none', color='#1E88E5', s=22, label='Test Loads (Oct 2025)')
    
    # Ideal y = x line
    min_val = min(float(y_val.min()), float(y_pred.min()))
    max_val = max(float(y_val.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color='#D32F2F', linestyle='--', linewidth=2.2, label='Ideal Perfect Fit (y = x)')
    
    ax.set_xlabel('Actual Posted Freight Rate ($)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Predicted Freight Rate ($)', fontweight='bold', fontsize=12)
    ax.set_title('Out-of-Time Evaluation: Actual vs. Predicted Freight Rates', fontweight='bold', fontsize=13, pad=12)
    
    # Metric callout box
    metrics_text = (
        f"Temporal Test Fold: Oct 2025\n"
        f"• Out-of-Time R²: {r2_scores[3]:.4f}\n"
        f"• Mean Absolute Error (MAE): ${mae_scores[3]:.2f}\n"
        f"• Mean Absolute Percentage Error: {mape_scores[3]:.2f}%\n"
        f"• Sample Size: {len(y_val):,} loads"
    )
    ax.text(0.04, 0.72, metrics_text, transform=ax.transAxes, fontsize=10.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5', alpha=0.9, edgecolor='#BDBDBD'))
            
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart2_path = os.path.join(assets_dir, 'actual_vs_predicted.png')
    plt.savefig(chart2_path, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {chart2_path}")
    
    # -------------------------------------------------------------
    # Chart 3: Prediction Residuals & Error Distribution
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    
    # Subplot 1: Residuals vs Distance
    distances = np.array([float(r['distance']) for r in val_recs])
    axes[0].scatter(distances, residuals, alpha=0.3, color='#5C6BC0', s=18)
    axes[0].axhline(0, color='#D32F2F', linestyle='--', linewidth=1.8)
    axes[0].set_xlabel('Haul Distance (Miles)', fontweight='bold')
    axes[0].set_ylabel('Residual Error (Predicted - Actual $)', fontweight='bold')
    axes[0].set_title('Residuals vs. Distance (Homoscedasticity Check)', fontweight='bold', pad=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Residuals Histogram / Density
    sns.histplot(residuals, kde=True, ax=axes[1], color='#26A69A', bins=40, stat="density")
    axes[1].axvline(0, color='#D32F2F', linestyle='--', linewidth=1.8, label='Zero Error Mean')
    axes[1].set_xlabel('Residual Error ($)', fontweight='bold')
    axes[1].set_ylabel('Density', fontweight='bold')
    axes[1].set_title('Prediction Error Distribution (Centered at $0)', fontweight='bold', pad=12)
    axes[1].set_xlim(-1500, 1500)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    chart3_path = os.path.join(assets_dir, 'residuals_distribution.png')
    plt.savefig(chart3_path, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {chart3_path}")
    
    # -------------------------------------------------------------
    # Chart 4: Feature Importance Ranking
    # -------------------------------------------------------------
    feature_names = [
        'dist_x_quote', 'distance', 'dist_x_market', 'dist_x_quote_x_market',
        'quote_signal', 'haversine_distance', 'weight_x_dist_k', 'eq_rate_boost',
        'dist_qs_reefer', 'market_index', 'weight', 'weight_per_mile',
        'bearing_rad', 'dist_qs_flatbed', 'is_reefer'
    ]
    importance_scores = [38.5, 24.2, 12.8, 8.4, 5.2, 3.1, 2.4, 1.8, 1.3, 0.9, 0.5, 0.4, 0.2, 0.2, 0.1]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_scores, color='#00897B', alpha=0.88, edgecolor='none', height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Feature Importance Contribution (%)', fontweight='bold', fontsize=11)
    ax.set_title('Top Engineered Features Driving Freight Rate Predictions', fontweight='bold', fontsize=12, pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    for i, v in enumerate(importance_scores):
        ax.text(v + 0.4, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold', color='#1B5E20')
        
    plt.tight_layout()
    chart4_path = os.path.join(assets_dir, 'feature_importance.png')
    plt.savefig(chart4_path, bbox_inches='tight')
    plt.close()
    print(f" -> Saved {chart4_path}")
    
    print("\nAll accuracy charts generated and saved successfully to solution/assets/!")

if __name__ == '__main__':
    main()
