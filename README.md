# Predicting Electric Vehicle Purchase

## Objective

Build a binary classifier that estimates the probability that a customer will purchase an electric vehicle (EV). The target is `Will_Buy_EV`, where `Yes` is encoded as `1` and `No` as `0`.

## Data

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| Training set | 668,665 | 15 | Model development; includes `Will_Buy_EV` |
| Test set | 286,571 | 14 | Final probability predictions |

The test set contained no missing values. The `id` field was removed before modeling and retained only for the submission file.

### Features

- Numeric: age, annual income, daily commute distance, number of cars owned, charging stations near home and work, and environmental-concern level.
- Categorical: gender, city type, current car type, home-charging availability, subsidy availability, and range-anxiety level.

### Training-set profile

| Measure | Value |
|---|---:|
| Mean age | 47.0 years |
| Mean annual income | $84,769 |
| Mean daily commute | 32.2 km |
| EV-purchase rate | 17.46% |
| Positive examples | 116,779 |
| Negative examples | 551,886 |

The target is imbalanced: approximately 82.54% of records are non-purchasers. The train/validation split was therefore stratified.

## Preparation

1. Dropped `id` from the modeling data.
2. Converted `Will_Buy_EV` from `Yes`/`No` to `1`/`0`.
3. One-hot encoded categorical variables with the first category dropped. This produced 18 model features.
4. Split the data into 80% training (534,932 rows) and 20% validation (133,733 rows), using `random_state=42` and stratification.
5. Standardized features for logistic regression only.

An IQR check found 41 potential outliers in `Daily_Commute_km`; the notebook did not remove them.

## Models evaluated

| Model | Validation ROC-AUC | Notes |
|---|---:|---|
| Logistic Regression | 0.93797 | Baseline model; trained on standardized features |
| HistGradientBoostingClassifier | **0.94116** | Selected for final predictions |
| CatBoostClassifier | 0.94116 | Used native categorical columns; 500 iterations |

The gradient-boosting model matched CatBoost to the reported precision and slightly exceeded the logistic-regression baseline. It was retrained on all 668,665 training examples before inference on the test data.

### Logistic-regression classification results

At its default decision threshold, the logistic-regression baseline achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.8949 |
| Precision | 0.7166 |
| Recall | 0.6588 |
| F1 score | 0.6865 |

Confusion matrix (rows: actual; columns: predicted):

| | Predicted no | Predicted yes |
|---|---:|---:|
| Actual no | 104,292 | 6,085 |
| Actual yes | 7,968 | 15,388 |

## Main drivers

Permutation importance from the HistGradientBoosting model ranked the leading variables as follows:

| Rank | Feature | Mean ROC-AUC importance |
|---:|---|---:|
| 1 | Environmental concern level | 0.19175 |
| 2 | Subsidy available: yes | 0.16476 |
| 3 | Annual income | 0.02884 |
| 4 | Range anxiety: low | 0.01427 |
| 5 | Daily commute distance | 0.00142 |

Environmental concern and subsidy availability dominate the model signal. Importance measures contribution to predictive performance, not a causal effect.

## Final prediction output

The final HistGradientBoosting model produced 286,571 test-set purchase probabilities. The generated submission has two columns:

```text
id,Will_Buy_EV
668665,0.008739
668666,0.020414
...
```

Validation checks on the submission confirmed:

- Shape: 286,571 rows × 2 columns
- Missing values: none
- Minimum predicted probability: 0.000126
- Maximum predicted probability: 0.983175
- Unique predicted probabilities: 285,205

## Reproducibility notes

- Use the same one-hot-encoded column set and order in both training and test data.
- Keep `random_state=42` for the train/validation split and gradient-boosting model.
- ROC-AUC is the principal comparison metric because it evaluates ranking quality independently of a single classification threshold.
- If a yes/no action is required, tune the probability threshold against the business cost of false positives and false negatives instead of automatically using 0.50.
