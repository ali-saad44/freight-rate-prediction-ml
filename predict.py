"""Inference Script to Generate Validation and December Predictions."""
import os
import sys
import pickle
import csv
from src.data_loader import (
    load_csv,
    save_csv,
    preprocess_dataset
)
from src.features import transform_records

def main():
    print("=" * 70)
    print("  Spotter Freight Rate Prediction - Generating Predictions")
    print("=" * 70)
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, 'data')
    models_dir = os.path.join(root_dir, 'models')
    bundle_path = os.path.join(models_dir, 'model_bundle.pkl')
    
    if not os.path.exists(bundle_path):
        print("Model bundle not found. Running training first...")
        import train
        train.main()
        
    print(f"\n[1/3] Loading trained model from {bundle_path}...")
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
        
    model = bundle['model']
    coord_map = bundle['coord_map']
    mi_lookup = bundle['mi_lookup']
    qs_lookup = bundle['qs_lookup']
    
    # 1. Predict on Validation Set
    val_path = os.path.join(data_dir, 'validation.csv')
    print(f"\n[2/3] Predicting {val_path}...")
    raw_val = load_csv(val_path)
    cleaned_val = preprocess_dataset(raw_val, coord_map, mi_lookup, qs_lookup)
    X_val, _ = transform_records(cleaned_val)
    val_preds = model.predict(X_val)
    
    # Format validation_predictions.csv
    val_output_rows = [
        {'load_id': raw_val[i]['load_id'], 'predicted_rate': f"{val_preds[i]:.2f}"}
        for i in range(len(raw_val))
    ]
    
    out_val_csv = os.path.join(root_dir, 'validation_predictions.csv')
    save_csv(out_val_csv, val_output_rows, fieldnames=['load_id', 'predicted_rate'])
    print(f"      Saved {len(val_output_rows):,} predictions to {out_val_csv}")
    
    # Check stats
    rates = [float(r['predicted_rate']) for r in val_output_rows]
    print(f"      Validation Rate Stats: Min=${min(rates):.2f}, Max=${max(rates):.2f}, Mean=${sum(rates)/len(rates):.2f}")
    
    # 2. Predict on December Chart Inputs
    dec_path = os.path.join(data_dir, 'december-chart-inputs.csv')
    print(f"\n[3/3] Predicting December Chart Scenario from {dec_path}...")
    raw_dec = load_csv(dec_path)
    cleaned_dec = preprocess_dataset(raw_dec, coord_map, mi_lookup, qs_lookup)
    X_dec, _ = transform_records(cleaned_dec)
    dec_preds = model.predict(X_dec)
    
    # Fill predicted_rate while preserving columns
    for i in range(len(raw_dec)):
        raw_dec[i]['predicted_rate'] = f"{dec_preds[i]:.2f}"
        
    fieldnames = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
    save_csv(dec_path, raw_dec, fieldnames=fieldnames)
    print(f"      Updated {len(raw_dec)} rows in {dec_path}")
    
    dec_rates = [float(r['predicted_rate']) for r in raw_dec]
    print(f"      December Rate Stats: Min=${min(dec_rates):.2f}, Max=${max(dec_rates):.2f}, Mean=${sum(dec_rates)/len(dec_rates):.2f}")
    
    print("\nInference completed successfully!\n")

if __name__ == '__main__':
    main()
