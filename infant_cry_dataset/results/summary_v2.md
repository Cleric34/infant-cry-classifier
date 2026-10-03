# Infant Cry Classification: Pipeline v2 Evaluation Report

## 🏆 Headline Model Performance (Repeat 0, 5-Fold Grouped CV with 95% CIs)

| Model | Accuracy (95% CI) | Balanced Accuracy (95% CI) | Macro F1 (95% CI) | Burping Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (LGBM v2cv P1)** | 84.09% [82.82%, 85.31%] | 70.17% [68.26%, 72.06%] | 0.7388 [0.7182, 0.7579] | 38.7% |
| **Final Candidate (stack_offsets_oof)** | 87.66% [86.57%, 88.75%] | 77.72% [75.76%, 79.53%] | 0.7993 [0.7812, 0.8162] | 49.0% |

### 📊 Paired Bootstrap Difference (Final - Baseline)
* **Accuracy Gain**: +3.57% (95% CI: [+2.65%, +4.48%])
* **Macro F1 Gain**: +0.0607 (95% CI: [+0.0459, +0.0752])

## 📜 Historical Baseline Comparison (Old Split 'test' Subset, n=489)
* **Random Forest Baseline**: 68.10% Accuracy, 0.6029 Macro F1
* **KNN Baseline**: 59.30% Accuracy, 0.5450 Macro F1
* **Majority Class Baseline (`needs`)**: 63.60% Accuracy, 0.1943 Macro F1
* **Final Model (Old Test View)**: 87.32% Accuracy, 0.7913 Macro F1

## 🔒 Honesty & Audit Section
* **Total Candidates Evaluated**: ~12 candidate model/feature/policy combinations across the pipeline.
* **Tuning Guardrails**: All hyperparameter selection, scaler fitting, offset search, and stacker training were strictly performed inside inner training fold loops without seeing outer held-out validation folds.
