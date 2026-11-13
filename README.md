# Stacking and Ensemble Methods on the Iris Dataset

A hands-on notebook that builds a **stacking classifier** and compares it with other popular ensemble methods (Random Forest, AdaBoost, Gradient Boosting, XGBoost) on the classic Iris dataset.

## Overview

The goal of this project is to understand how different ensemble techniques work and how they compare on the same data split. The notebook covers:

- Loading and preparing the Iris dataset
- Building a **stacking ensemble** with three different base learners and a KNN meta-learner
- Training four other ensemble models for comparison
- Evaluating every model with accuracy on the same held-out test set

## What is stacking?

Stacking combines several different models into one:

1. **Base learners** (here: Decision Tree, Logistic Regression, SVM) each make predictions.
2. A **meta-learner** (here: K-Nearest Neighbors) learns how to best combine those predictions into the final answer.
3. The base learners' predictions used to train the meta-learner are generated with **5-fold cross-validation** (`cv=5`), which prevents the meta-learner from seeing predictions made on data the base models were trained on.

This differs from bagging (Random Forest), which averages many copies of the same model type, and from boosting (AdaBoost, Gradient Boosting, XGBoost), which trains models one after another to fix earlier mistakes.

## Dataset

**Iris** (loaded with `seaborn.load_dataset('iris')`): 150 flowers, 4 numeric features, 3 classes.

| Feature | Description |
|---|---|
| `sepal_length`, `sepal_width` | Sepal measurements (cm) |
| `petal_length`, `petal_width` | Petal measurements (cm) |
| `species` (target) | setosa, versicolor, virginica |

## Method

- **Encoding:** the species labels are converted to numbers with `LabelEncoder`.
- **Split:** 80% train / 20% test (`test_size=0.2`, `random_state=42`), **stratified** so each class appears in the same proportion in both sets. The test set contains 30 samples.
- **Stacking model:**
  - Base learners: `DecisionTreeClassifier`, `LogisticRegression(max_iter=200)`, `SVC(probability=True)`
  - Meta-learner: `KNeighborsClassifier(n_neighbors=5)`
  - `cv=5`
- **Comparison models:**
  - `RandomForestClassifier(n_estimators=100)`
  - `AdaBoostClassifier(n_estimators=100)`
  - `GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)`
  - `XGBClassifier(n_estimators=100)`

All models use `random_state=42`.

## Results

Accuracy on the 30-sample test set:

| Model | Accuracy |
|---|---|
| Stacking (Decision Tree + Logistic Regression + SVM, KNN meta-learner) | 0.967 |
| Gradient Boosting | 0.967 |
| AdaBoost | 0.933 |
| XGBoost | 0.933 |
| Random Forest | 0.900 |

### How to read these results

The test set has only 30 samples, so **one sample equals 3.3 percentage points**. The gap between the best and worst model is just 2 samples. These numbers come from a single train/test split, so the ranking could change with a different `random_state`. A fairer comparison would use cross-validation (see "Possible improvements").

## Getting started

### Requirements

- Python 3.9 or newer
- scikit-learn
- xgboost
- seaborn (installs pandas and matplotlib)

```bash
pip install scikit-learn xgboost seaborn pandas matplotlib jupyter
```

### Run the notebook

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook Stacking.ipynb
```

Note: `sns.load_dataset('iris')` downloads the data on first use, so an internet connection is needed the first time. As an offline alternative, use `from sklearn.datasets import load_iris`.

## Repository structure

```
.
├── Stacking.ipynb   # Full code: data prep, stacking model, ensemble comparison
└── README.md
```

## Possible improvements

- Compare models with **k-fold cross-validation** (`cross_val_score`) instead of a single split, and report mean and standard deviation.
- Add a **classification report and confusion matrix** to see which classes get confused (versicolor and virginica usually overlap).
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`.
- Try a different meta-learner, such as Logistic Regression, which is the scikit-learn default for stacking.
- Repeat the experiment on a larger, more difficult dataset, where differences between ensemble methods show up more clearly.

## Tools used

Python, scikit-learn, XGBoost, seaborn, Jupyter Notebook
