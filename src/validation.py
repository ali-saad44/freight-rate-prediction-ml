"""Temporal Cross-Validation Module."""
from .features import transform_records
from .utils import calculate_metrics

def create_temporal_splits(cleaned_records):
    """
    Create expanding-window time-based validation splits:
    - Fold 1: Train Jan-Jul, Val Aug
    - Fold 2: Train Jan-Aug, Val Sep
    - Fold 3: Train Jan-Sep, Val Oct
    """
    splits = [
        {
            'name': 'Fold 1 (Val: Aug 2025)',
            'train_end': '2025-07-31',
            'val_start': '2025-08-01',
            'val_end': '2025-08-31'
        },
        {
            'name': 'Fold 2 (Val: Sep 2025)',
            'train_end': '2025-08-31',
            'val_start': '2025-09-01',
            'val_end': '2025-09-30'
        },
        {
            'name': 'Fold 3 (Val: Oct 2025)',
            'train_end': '2025-09-30',
            'val_start': '2025-10-01',
            'val_end': '2025-10-31'
        }
    ]
    
    cv_splits = []
    for s in splits:
        train_recs = [r for r in cleaned_records if r['date'] <= s['train_end']]
        val_recs = [r for r in cleaned_records if s['val_start'] <= r['date'] <= s['val_end']]
        cv_splits.append((s['name'], train_recs, val_recs))
        
    return cv_splits

def evaluate_model_cv(model_class, model_kwargs, cleaned_records):
    """Run temporal cross-validation on a model class."""
    splits = create_temporal_splits(cleaned_records)
    fold_metrics = []
    
    for fold_name, train_recs, val_recs in splits:
        X_train, y_train = transform_records(train_recs)
        X_val, y_val = transform_records(val_recs)
        
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        metrics = calculate_metrics(y_val, y_pred)
        fold_metrics.append((fold_name, metrics))
        
    # Aggregate mean metrics
    avg_rmse = sum(m['rmse'] for _, m in fold_metrics) / len(fold_metrics)
    avg_mae = sum(m['mae'] for _, m in fold_metrics) / len(fold_metrics)
    avg_mape = sum(m['mape'] for _, m in fold_metrics) / len(fold_metrics)
    avg_r2 = sum(m['r2'] for _, m in fold_metrics) / len(fold_metrics)
    
    summary = {
        'rmse': round(avg_rmse, 4),
        'mae': round(avg_mae, 4),
        'mape': round(avg_mape, 4),
        'r2': round(avg_r2, 4),
        'folds': fold_metrics
    }
    return summary
