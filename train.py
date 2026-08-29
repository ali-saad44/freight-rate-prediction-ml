"""Main Model Training and Cross-Validation Script."""
import os
import sys
import pickle
from src.data_loader import (
    load_csv,
    build_city_coordinate_map,
    build_market_index_lookup,
    build_quote_signal_lookup,
    preprocess_dataset
)
from src.features import transform_records
from src.models import (
    RidgeRegressorModel,
    LightGBMRegressorModel,
    CatBoostRegressorModel,
    EnsembleRegressorModel
)
from src.validation import evaluate_model_cv
from src.utils import print_metrics_table

def main():
    print("=" * 70)
    print("  Spotter Freight Rate Prediction - Model Training & Evaluation")
    print("=" * 70)
    
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_path = os.path.join(data_dir, 'train-test.csv')
    val_path = os.path.join(data_dir, 'validation.csv')
    
    print(f"\n[1/5] Loading data from {train_path}...")
    raw_train = load_csv(train_path)
    raw_val = load_csv(val_path) if os.path.exists(val_path) else []
    print(f"      Loaded {len(raw_train):,} training records.")
    
    # 2. Build domain lookups & clean dataset
    print("\n[2/5] Building domain lookups and cleaning data...")
    coord_map = build_city_coordinate_map(raw_train, raw_val)
    mi_lookup = build_market_index_lookup(raw_train, raw_val)
    qs_lookup = build_quote_signal_lookup(raw_train, raw_val)
    
    cleaned_train = preprocess_dataset(raw_train, coord_map, mi_lookup, qs_lookup)
    print(f"      Cleaned and imputed {len(cleaned_train):,} records.")
    
    # 3. Temporal Cross-Validation
    print("\n[3/5] Running Expanding-Window Temporal Cross-Validation (Aug, Sep, Oct 2025)...")
    
    cv_results = {}
    
    print("  -> Evaluating Baseline Ridge Regression...")
    ridge_cv = evaluate_model_cv(RidgeRegressorModel, {'alpha': 10.0}, cleaned_train)
    cv_results['Ridge Baseline'] = ridge_cv
    
    print("  -> Evaluating LightGBM / GBDT Regressor...")
    lgb_cv = evaluate_model_cv(LightGBMRegressorModel, {'n_estimators': 300, 'learning_rate': 0.05}, cleaned_train)
    cv_results['LightGBM / GBDT'] = lgb_cv
    
    print("  -> Evaluating CatBoost Regressor...")
    cb_cv = evaluate_model_cv(CatBoostRegressorModel, {'iterations': 300, 'learning_rate': 0.05}, cleaned_train)
    cv_results['CatBoost'] = cb_cv
    
    # 4. Display Benchmarks
    print("\n[4/5] Out-of-Time Cross-Validation Performance:")
    print("-" * 70)
    print_metrics_table(cv_results)
    print("-" * 70)
    
    # 5. Final Model Training
    print("\n[5/5] Training Chosen Final Ensemble Model on 100% Development Data...")
    X_train, y_train = transform_records(cleaned_train)
    
    m1 = LightGBMRegressorModel(n_estimators=400, learning_rate=0.04)
    m2 = RidgeRegressorModel(alpha=5.0)
    
    final_model = EnsembleRegressorModel(models=[m1, m2], weights=[0.85, 0.15])
    final_model.fit(X_train, y_train)
    
    # Save model artifact and preprocessing lookups
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    bundle_path = os.path.join(models_dir, 'model_bundle.pkl')
    
    bundle = {
        'model': final_model,
        'coord_map': coord_map,
        'mi_lookup': mi_lookup,
        'qs_lookup': qs_lookup,
        'cv_results': cv_results
    }
    
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
        
    print(f"      Saved model bundle to {bundle_path}")
    print("\nTraining completed successfully!\n")

if __name__ == '__main__':
    main()
