# ML Financial Health Analytics

Two end-to-end machine learning pipelines in Python, one supervised and one unsupervised, each taken from raw CSV through EDA, pre-processing, modelling, tuning and evaluation. Both notebooks are written up step by step so a reader can follow the reasoning, not just the code.

| Track | Notebook | Problem | Data | Models |
|---|---|---|---|---|
| **Supervised** | [`Supervised_Stroke_prediction.ipynb`](Supervised_Stroke_prediction.ipynb) | Predict whether a patient will have a stroke (binary classification, heavily imbalanced) | 5,110 patients, 11 features (`data/healthcare-dataset-stroke-data.csv`) | Logistic Regression vs Random Forest |
| **Unsupervised** | [`Unsupervised_CC.ipynb`](Unsupervised_CC.ipynb) | Segment credit-card customers by spending behaviour | 8,950 customers, 17 behavioural features (`data/CC GENERAL.csv`) | Hierarchical clustering → K-Means (k = 4) → PCA |

Both notebooks were built in Google Colab and open there directly from the badge at the top of each file.

## Supervised: stroke prediction

**The challenge is class imbalance.** Only ~5% of patients had a stroke, so a model that always says "no stroke" scores ~95% accuracy while being useless. Recall on the positive class is the metric that matters.

Pipeline:

1. **EDA** — descriptive stats, histogram (age), box plot (BMI outliers), count plot (target imbalance), pair plot and scatter (age vs glucose). Age emerged as the dominant predictor.
2. **Pre-processing** — clip BMI at 60, median-impute 201 missing BMI values, one-hot encode five categoricals, `StandardScaler` on numerics; all wrapped in a `ColumnTransformer` so train and test get identical treatment.
3. **Models** — `LogisticRegression` and `RandomForestClassifier`, both with `class_weight='balanced'`, stratified 80/20 split.
4. **Tuning** — `GridSearchCV` scored on recall, then 5-fold cross-validation to check the result wasn't luck.
5. **Evaluation** — confusion matrices, ROC curves and AUC.

Results:

| Model | Accuracy | Recall (stroke) | CV recall after tuning | AUC (tuned) |
|---|---|---|---|---|
| Logistic Regression | 74.4% | **78.0%** | 0.83 ± 0.09 | 0.84 |
| Random Forest (untuned) | 94.8% | 0.0% | — | — |
| Random Forest (tuned: depth 5, 200 trees) | — | — | 0.81 ± 0.06 | 0.83 |

The untuned random forest is the textbook accuracy trap: it never predicted a single stroke. After constraining depth and leaf size it caught up with logistic regression, but the simpler model (best `C = 0.01`) generalised as well or better, which says the signal is in a few obvious features rather than complex interactions.

## Unsupervised: credit-card customer segmentation

1. **EDA** — 313 missing `MINIMUM_PAYMENTS` and 1 missing `CREDIT_LIMIT`; balance and purchases are heavily right-skewed by a small number of big spenders; correlation heatmap to spot redundant features.
2. **Pre-processing** — median imputation, winsorising every feature at the 95th percentile so outliers don't drag the centroids, `StandardScaler`.
3. **Choosing k** — Ward-linkage dendrogram for a first look, then the elbow method over k = 1…10, which flattens at **k = 4**.
4. **Clustering** — K-Means (`k-means++`, 10 inits), projected to 2-D with PCA for visual validation; clusters separate cleanly.
5. **Evaluation** — silhouette score plus a per-cluster profile on `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, so each segment can be described in plain business terms rather than as a cluster number.

## Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SciPy (hierarchical clustering) · Google Colab

## Running the notebooks

Open either notebook in Colab via the badge, upload the matching CSV from `data/` when prompted (the notebooks read from `/content/`), and run all cells. Locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
jupyter notebook
```

## What I learned

- **Pick the metric before the model.** Optimising for accuracy on imbalanced data produces confident, useless models.
- **Put pre-processing inside the pipeline.** A `ColumnTransformer` + `make_pipeline` guarantees the test set is transformed exactly like the training set and makes grid search honest.
- **Clustering is only as good as the scaling.** Without clipping and standardising, K-Means on raw financial data just finds "big spenders vs everyone else".
