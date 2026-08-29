"""Utility functions for metrics, evaluation, and logging."""
import math

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics: RMSE, MAE, MAPE, R2."""
    n = len(y_true)
    if n == 0:
        return {}
    
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n
    
    # MAPE (ignoring near-zero targets)
    mape_vals = [abs((yt - yp) / yt) for yt, yp in zip(y_true, y_pred) if abs(yt) > 1e-5]
    mape = (sum(mape_vals) / len(mape_vals) * 100.0) if mape_vals else 0.0
    
    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4),
        "r2": round(r2, 4)
    }

def print_metrics_table(results_dict):
    """Print a formatted markdown comparison table of model metrics."""
    header = "| Model | RMSE ($) | MAE ($) | MAPE (%) | R² Score |"
    divider = "| :--- | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)
    for name, m in results_dict.items():
        print(f"| **{name}** | ${m['rmse']:,.2f} | ${m['mae']:,.2f} | {m['mape']:.2f}% | {m['r2']:.4f} |")
