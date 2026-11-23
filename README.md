# Loft25 Customer Returns — Analysis & Predictive Modeling

End-to-end data quality audit, exploratory data analysis (EDA), and leak-free predictive modeling of customer order returns for **Loft25** (*Gardenista* and *Game Over* brands) across 21 months of transactions (`2025-01-05` to `2026-09-30`).

---

## Repository Contents

| File | Description | Size / Shape |
| :--- | :--- | :--- |
| `Loft25_Returns_Analysis_Modular.ipynb` | Executed modular notebook (`Module 1`–`Module 7`, `1.1`–`7.4`) with rendered outputs & charts | 108 cells (78 Markdown, 30 Code) |
| `Untitled0.ipynb` | Original exploratory draft notebook (preserved for reference) | 20 cells |
| `Loft25_BCU_04_P4_order_lines.csv` | Itemized order-line transactions (`('Order ref', 'SKU')` unique key) | 34,305 × 15 |
| `Project 4 & 5 - customer orders.xlsx` | Order-level customer registry (`'Order ref'` unique key) | 31,410 × 6 |
| `Project 4 & 5 - Returns.xlsx` | Returned item details, reasons, feedback, scrap flags, and return costs | 1,523 × 20 |
| `Loft25_BCU_05_P4_returns_by_SKU.csv` | SKU-level rollup of sales, returns, kept rate, and traffic-light risk flags | 1,458 × 10 |

---

## Key Operational & Financial KPIs

| Metric | Value |
| :--- | :--- |
| Total Order Lines | 34,305 |
| Distinct Orders | 31,410 |
| Distinct Customers | 22,854 |
| Line Return Rate | 4.47% (1,533 returned lines) |
| Order Return Rate | 4.59% |
| Unit Return Rate | 4.83% (2,086 / 43,180 units) |
| Gross Sales Value (inc VAT) | £1,622,434.24 |
| Total Refunded Value | £79,405.56 |
| Direct Operational Return Cost (`Shipping + Write-off`) | £16,317.26 |
| Return Scrap Rate | 29.61% (451 / 1,523 returns) |

---

## Audited Data Quality Findings

1. **Blank separator column in `Returns.xlsx`:** Column O (`Unnamed: 14`) is 100% empty and dropped in Module `2.2`.
2. **10 returned lines missing from `Returns.xlsx` (1,533 vs. 1,523):** All 10 missing lines are £0-priced items (`Unit price = 0`, `Refund amount = 0`) across `Direct/Manual` (5) and `TikTok Shop` (5). Separately, 5 `TikTok Shop` rows inside `Returns.xlsx` have `Refund amount = 0` despite non-zero line prices (£18.98–£71.88).
3. **Discount outliers:** `Line discount` behaves as a percentage (`0–100`) and exceeds `100` on 3 two-unit rows (`Line discount = 200` with `Qty = 2`, `Unit price = 0`); clipped to `[0, 100]` into `Line_discount_clean`.
4. **Zero-price order lines:** 724 order lines have `Unit price inc VAT == 0` (heavily concentrated in `Direct/Manual`, where 327/332 lines are £0).
5. **Composite join key:** `('Order ref', 'SKU')` is strictly unique in both `order_lines` and `Returns.xlsx`. Joining on `'Order ref'` alone causes row duplication because 1,523 return lines span 1,433 distinct orders.
6. **Seasonal & brand mix shift:** *Game Over* (`2.03%` return rate) has 0 orders in Jan–Jun 2025, enters in Jul 2025, and accounts for 86%–94% of Nov–Dec volume, while *Gardenista* (`4.89%` return rate) peaks in Apr–Jul.

---

## Modular Notebook Architecture

`Loft25_Returns_Analysis_Modular.ipynb` follows a structured **Title → Explain → Small Code Block** cadence across 7 modules:

- **Module 1 (`1.1`–`1.4`) — Environment Setup & Data Loading:** Conditional package installer, global configuration, auto-detection of local `.` vs. Colab `/content` paths, and schema overview.
- **Module 2 (`2.1`–`2.4`) — Data Quality Audit & Cleaning:** Null/duplicate scans, dropping `Unnamed: 14`, datetime standardization, cross-table key checks, and discount clipping.
- **Module 3 (`3.1`–`3.2`) — Relational Merging:** Pre-purchase `master_df` (`many_to_one` on `Order ref`) and EDA-only `eda_df` (`one_to_one` on `['Order ref', 'SKU']`).
- **Module 4 (`4.1`–`4.7`) — Exploratory Data Analysis:** Overall KPIs, Brand & Channel comparisons with sample-size annotations, monthly seasonality by brand, courier rates, return reason scrap/cost breakdown, and high-volume SKU risk filtering (`Units sold >= 20`).
- **Module 5 (`5.1`–`5.6`) — Feature Engineering & Leak-Free Splitting:** Checkout-time features (temporal, basket size/share, SKU prefix, pack size, `BT` postcode flag), strictly past-only customer order history, **customer-grouped 70/15/15 Train/Val/Test split** (`GroupShuffleSplit`), 5-fold customer-grouped **out-of-fold (OOF)** smoothed SKU target encoding, and model-specific encoders.
- **Module 6 (`6.1`–`6.4`) — Imbalance-Aware Models & Thresholding:** Validation-tuned Class-1 F1 threshold optimization, Balanced Logistic Regression, `scale_pos_weight` XGBoost (validation early stopping), and TabNet with categorical embeddings (`weights=1`, validation early stopping).
- **Module 7 (`7.1`–`7.4`) — Comparison, Interpretability & Takeaways:** Test leaderboard, ROC & Precision-Recall curves, XGBoost feature importances, and business recommendations.

---

## Executed Model Results (Held-Out Customer-Grouped Test Set)

Evaluated on `5,109` order lines from `3,429` completely unseen customers (`4.54%` positive baseline return rate). Thresholds were tuned strictly on the validation set (`5,059` lines).

| Model | Val-Tuned Threshold | ROC-AUC | PR-AUC | Class-1 Precision | Class-1 Recall | Class-1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.6285 | **0.6208** | **0.0858** | 0.0904 | **0.2284** | **0.1296** |
| **XGBoost (Weighted)** | 0.5186 | 0.5949 | 0.0767 | **0.1136** | 0.1078 | 0.1106 |
| **TabNet (Embeddings + Balanced)** | 0.6106 | 0.5947 | 0.0654 | 0.0751 | 0.1940 | 0.1083 |

> **Improvement over original draft (`Untitled0.ipynb`):** In the original draft, all three models predicted `0.00` recall and `0.00` F1 on returns (and TabNet leaked the test set into early stopping). With customer-grouped splitting, pre-purchase feature engineering, OOF SKU encoding, class weighting, and validation threshold tuning, all three models achieve lift above the `0.0454` PR-AUC baseline (up to `1.89×` lift with Balanced Logistic Regression).

**Top XGBoost Return-Risk Predictors:** `sku_prefix` (`0.188`), `Brand` (`0.091`), `sku_ret_rate_enc` (`0.086`), `line_value_share` (`0.051`), `order_total_qty` (`0.048`), `Courier service` (`0.045`).

---

## Business Recommendations

1. **Address *Gardenista* Fit & Product Listing Clarity:** *Gardenista* returns at `4.89%` vs. `2.03%` for *Game Over*. `Does not fit` (330 returns) and `Item not as described / different to photos` (217 returns) represent over one-third of all returns.
2. **Reduce High-Scrap Failure Modes:** `Faulty / poor quality` (£3,912.81 operational cost, `65.9%` scrapped), `Damaged on arrival` (£3,796.37, `87.9%` scrapped), and `Smell / stains` (`95.6%` scrapped) generate the bulk of inventory write-offs.
3. **Review High-Return Couriers & Channels:** Audit `EVRI NEXT DAY` (`20.0%`, `n=40`), `BT Postcode - Do not Ship` (`66.7%`, `n=6`), `The Range` (`6.47%`), and `TikTok Shop` (`6.73%`).
4. **Harden Export Data Hygiene:** Reconcile the 10 £0-value returns missing from `Returns.xlsx`, cap exported `Line discount` percentages at `100%`, and review partial-unit return refund calculations.

---

## How to Run

### Local (Jupyter / VS Code)
```bash
git clone git@github.com:promit-bhattacharjee/loft25-returns-analysis.git
cd loft25-returns-analysis
pip install pandas numpy openpyxl matplotlib seaborn scikit-learn xgboost pytorch-tabnet jupyter
jupyter notebook Loft25_Returns_Analysis_Modular.ipynb
```

### Google Colab
1. Upload `Loft25_Returns_Analysis_Modular.ipynb` and the 4 dataset files into `/content/`.
2. Run all cells sequentially — Module `1.1` installs any missing packages and Module `1.3` automatically detects `/content/`.
