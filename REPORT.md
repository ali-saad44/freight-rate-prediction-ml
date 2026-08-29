# Machine Learning Technical Assessment Report: Freight Rate Prediction

**Author**: Machine Learning Engineer Candidate  
**Assessment**: Spotter Freight Rate Prediction Challenge  
**Date**: August 2026  

---

## 1. Executive Summary

This report presents an end-to-end Machine Learning solution developed to predict truckload freight rates (`posted_rate`) in the spot market. Using historical freight transaction data from January 1, 2025 to October 31, 2025 (48,000 loads), our objective was to generate accurate, robust predictions for 12,000 out-of-time spot loads in November and December 2025 (`validation.csv`) and evaluate price trajectories for a fixed December lane scenario (`december-chart-inputs.csv`).

Key Highlights:
- **Validation Methodology**: Implemented an expanding-window temporal validation strategy across 3 sequential folds (August, September, October 2025) to prevent data leakage and simulate real production forecasting.
- **Feature Engineering**: Engineered 36 domain-specific features encompassing line-haul pricing interactions, spherical Haversine geometry, equipment rate multipliers, and cyclical temporal encodings.
- **Model Selection**: Benchmarked Ridge Regularized Regression, LightGBM, CatBoost, and a weighted Ensemble. The final LightGBM-dominated ensemble achieved an out-of-time **$R^2$ of 0.9910** and an average **RMSE of $176.90** (a **35.5% error reduction** over baseline linear models).

---

## 2. Exploratory Data Analysis & Key Findings

### 2.1 Pricing Dynamics & Feature Relationships
1. **Distance & Quote Signal Interaction**:
   - Distance alone has a high correlation with rate ($r = 0.9085$), but the rate-per-mile varies considerably based on the quote signal ($r = 0.8987$ for $	ext{distance} 	imes 	ext{quote\_signal}$).
   - The primary driver of freight pricing is the compound interaction between distance, equipment type, and market tightness.
2. **Equipment Type Variations**:
   - **Reefer (Refrigerated)** loads command the highest average rate multiplier (~1.19x of base rate-per-mile) due to fuel consumption for refrigeration units and equipment scarcity.
   - **Flatbed** loads average ~1.13x multiplier due to specialized securing and loading requirements.
   - **Dry Van** represents standard freight (~1.05x multiplier).
3. **Distribution & Outliers**:
   - Rates range from \$57.22 to \$25,533.00 with a mean of \$2,373.98.
   - Rate-per-mile distributions exhibit higher fixed minimum charges for short hauls (< 100 miles) and linear marginal rates for long-haul routes (> 500 miles).

---

## 3. Data Quality Issues & Remediation Strategy

| Issue Identified | Affected Columns | Impact | Remediation Strategy |
| :--- | :--- | :--- | :--- |
| **Missing Weights** | `weight` (300 in train, 165 in val) | Weight affects load density and rate. | Imputed using equipment-specific medians (`Dry Van: 30,000 lbs`, `Reefer: 32,000 lbs`, `Flatbed: 34,000 lbs`) + binary missingness indicator. |
| **Missing Market Index** | `market_index` (374 in train, 249 in val) | Loss of daily market tightness signal. | Imputed using temporal day-specific mean lookup + rolling historical interpolation + binary missingness flag. |
| **Unseen Locations** | `pickup`, `delivery` (8 unseen cities in val) | Target encoding on city names fails. | Extracted geographical coordinates (`lat`, `lon`), computed Haversine distance and directional bearing to ensure continuous spatial generalization. |
| **Missing Inputs in December File** | `december-chart-inputs.csv` | Missing coordinates and market signals. | Populated coordinates via city lookup mapping (`Lexington` $	o 36.99^\circ	ext{N}, -85.00^\circ	ext{W}$, `Fort Wayne` $	o 41.32^\circ	ext{N}, -85.36^\circ	ext{W}$) and inferred daily market index from validation temporal lookups. |

---

## 4. Feature Engineering

We designed 36 structured features categorized into 4 core domain areas:

1. **Pricing & Market Interactions**:
   - $	ext{dist\_x\_quote} = 	ext{distance} 	imes 	ext{quote\_signal}$
   - $	ext{dist\_x\_market} = 	ext{distance} 	imes 	ext{market\_index}$
   - $	ext{dist\_x\_quote\_x\_market} = 	ext{distance} 	imes 	ext{quote\_signal} 	imes 	ext{market\_index}$
   - $	ext{quote\_div\_market} = 	ext{quote\_signal} / (	ext{market\_index} + 10^{-4})$

2. **Geographical & Spatial Geometry**:
   - Great-circle **Haversine distance** calculated from origin to destination.
   - $	ext{distance\_ratio} = 	ext{reported\_distance} / (	ext{haversine\_distance} + 1.0)$ (identifies circuitous routing).
   - Coordinate deltas ($\Delta 	ext{lat}, \Delta 	ext{lon}$) and directional bearing angle in radians.

3. **Equipment & Load Density**:
   - Categorical indicators: `is_dry_van`, `is_reefer`, `is_flatbed`.
   - Interaction terms: $	ext{dist\_qs\_reefer}$, $	ext{dist\_qs\_flatbed}$.
   - Density: $	ext{weight\_per\_mile} = 	ext{weight} / (	ext{distance} + 1.0)$.
   - Heavy load flag: $\mathbb{I}(	ext{weight} > 38,000	ext{ lbs})$.

4. **Temporal & Cyclical Seasonality**:
   - Calendar features: `month`, `day_of_month`, `day_of_week`, `day_of_year`, `is_weekend`, `is_month_end`.
   - Cyclical transformations: $\sin(2\pi \cdot 	ext{day} / 365.25)$, $\cos(2\pi \cdot 	ext{day} / 365.25)$, $\sin(2\pi \cdot 	ext{dow} / 7)$, $\cos(2\pi \cdot 	ext{dow} / 7)$.
   - Holiday / Q4 surge flag for November and December.

---

## 5. Validation & Data Split Strategy

### Why Standard Random K-Fold Cross-Validation is Inappropriate:
In freight rate modeling, random K-Fold CV introduces severe **temporal data leakage**. Future market shocks, seasonal indices, and fuel pricing would leak into the training sets, generating artificially inflated performance scores that degrade on the unseen November–December test set.

### Expanding-Window Time-Series Split:
To faithfully mimic the actual evaluation environment, we partitioned the 10-month training data into an expanding window:

```
Fold 1: [--- Train: Jan 01 - Jul 31 ---] -> [Val: Aug 01 - Aug 31]
Fold 2: [------ Train: Jan 01 - Aug 31 ------] -> [Val: Sep 01 - Sep 30]
Fold 3: [--------- Train: Jan 01 - Sep 30 ---------] -> [Val: Oct 01 - Oct 31]
```

All hyperparameter tuning and model selection decisions were based solely on the mean out-of-time performance across these three folds.

---

## 6. Model Exploration, Benchmarking & Selection

We evaluated four model configurations under identical temporal CV splits:

| Model Architecture | Out-of-Time RMSE ($) | Out-of-Time MAE ($) | Out-of-Time MAPE (%) | Out-of-Time $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regularized Linear Model** | \$274.50 | \$182.10 | 8.45% | 0.9782 |
| **CatBoost Regressor** | \$198.30 | \$124.60 | 5.82% | 0.9886 |
| **LightGBM Regressor** | \$182.40 | \$112.50 | 5.20% | 0.9904 |
| 🏆 **Final Weighted Ensemble** | **\$176.90** | **\$108.20** | **4.98%** | **0.9910** |

### Reasoning for Chosen Model:
1. **LightGBM** provides superior performance on tabular regression with non-linear interactions. Its leaf-wise tree growth efficiently captures step-function pricing regimes (such as short-haul minimum rates).
2. **Ensemble Blending**: Combining LightGBM with Ridge Regularization and CatBoost provides variance reduction, smoothing out extreme edge-case predictions and improving out-of-time generalization.

---

## 7. December 2025 Fixed Lane Analysis

For the 31 daily scenario inputs in `december-chart-inputs.csv` (Lexington $	o$ Fort Wayne, 360 miles, Dry Van, 32,000 lbs):
- Rates maintain stability around the lane baseline (\$750 - \$850).
- Daily fluctuations reflect the market quote signals and weekend/weekday carrier capacity availability.
- Late December reflects holiday volume tightening as carrier capacity decreases before Christmas and New Year.

The validated plot has been generated and saved to:
`scorer_results/candidate_december.png`.

---

## 8. Codebase Walkthrough

The repository is modular and follows production software engineering practices:

- [`src/data_loader.py`](src/data_loader.py): Handles schema parsing, coordinate mapping, and missing value imputation.
- [`src/features.py`](src/features.py): Transforms raw records into 36 engineered pricing, geographic, and temporal features.
- [`src/models.py`](src/models.py): Implements model wrappers (LightGBM, CatBoost, Ridge, Standalone GBDT, and Ensemble).
- [`src/validation.py`](src/validation.py): Implements temporal expanding-window cross-validation.
- [`train.py`](train.py): Trains candidate models, logs benchmark tables, and serializes the final model bundle.
- [`predict.py`](predict.py): Executes inference to produce `validation_predictions.csv` and updates `december-chart-inputs.csv`.
- [`run_pipeline.py`](run_pipeline.py): Orchestrates the complete workflow end-to-end.
- [`score.py`](score.py): Validates predictions and generates the official December chart.

---

## 9. Conclusion
The developed solution meets all assessment criteria:
1. Validated and formatted predictions for all 12,000 loads in `validation_predictions.csv`.
2. Fully populated `data/december-chart-inputs.csv` and generated `scorer_results/candidate_december.png`.
3. Fully documented codebase with clear reproduction instructions in `README.md`.
