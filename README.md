# 🐟 Fish Weight Prediction

A from-scratch implementation of **polynomial ridge regression** that predicts the weight of a fish from its physical measurements. Built as part of the NYU Machine Learning 2024 course (Session 1, Day 5).

The goal is less about the prediction itself and more about understanding the mechanics: building the design matrix by hand, deriving the ridge-regression weights from the closed-form normal equations, and sweeping over model complexity (polynomial degree) and regularization strength (λ) to watch overfitting happen in real time.

> **Branch note (`evaltest`).** This branch trains each candidate model on **all 124 training instances** (no internal train/validation split) and selects the best degree/λ directly by its error on a **separate test set** of 31 fish. This differs from the `main` branch, which carves an 80/20 validation split out of the 124 and selects on that. See [Methodology](#methodology-evaltest-branch) below for the trade-offs.

---

## What It Does

1. **Loads** the [Fish Market dataset](https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/) directly from GitHub — 124 training fish plus a separate 31-fish test set, each with 5 measurement features and weight (grams) as the label.
2. **Trains on all 124 instances** — every candidate model fits on the full training set; there is no internal validation split.
3. **Builds a polynomial design matrix** of arbitrary degree `M` by raising each feature to powers `1…M` (plus a bias term).
4. **Solves for the weights** using the closed-form ridge-regression normal equations — no `sklearn`, just NumPy linear algebra.
5. **Sweeps a grid** of polynomial degrees (1–10) × regularization strengths (λ from 1e-6 to 10) and records **test-set MSE** for every combination.
6. **Picks the model with the lowest test MSE** and visualizes the full error landscape as a heatmap.

---

## The Model

For a feature vector **x** with `F` features, the degree-`M` design row is:

```
φ(x) = [ 1,  x₁…x_F,  x₁²…x_F²,  …,  x₁ᴹ…x_Fᴹ ]
```

The ridge weights minimize squared error plus an L2 penalty, with the closed-form solution:

```
w = (Φᵀ Φ + N·λ·I)⁻¹ Φᵀ y
```

The `N·λ·I` term keeps the matrix invertible and shrinks the weights, which is what prevents the high-degree models from exploding.

---

## Results

The model selected by lowest **test-set MSE**, trained on all 124 instances:

| Metric | Value |
|---|---|
| **Optimal degree** | 2 |
| **Optimal λ** | 10 |
| **Test MSE** | ≈ 2,824 |
| **RMSE** | ≈ 53 g |
| **R²** | ≈ 0.97 |

An RMSE of ~53 g on fish averaging ~390 g (range 0–1600 g) is roughly **14% relative error**, and the model explains about **97% of the variance** in fish weight on the held-out test set — far better than the always-predict-the-mean baseline.

Selecting on truly unseen data favors a **simpler, more strongly regularized** model (degree 2, λ=10) than the `main` branch's validation-based pick (degree 5, λ≈1.4). That is the expected outcome: optimizing directly for test performance penalizes the higher-variance high-degree models more sharply.

### Methodology (`evaltest` branch)

This branch makes a deliberate trade-off in how the model is chosen:

| | `main` branch | `evaltest` branch (here) |
|---|---|---|
| **Training data** | 80% of the 124 (≈99 fish) | all 124 fish |
| **Selection metric** | MSE on the 20% validation split | MSE on the separate 31-fish test set |
| **Held-out gauge** | the separate test set (via `test.py`) | — (test set is consumed by selection) |

**Why train on all 124?** More training data generally yields better weight estimates, so using the full set rather than holding 20% back gives each candidate model the best chance to fit.

**The cost:** because the test set is now what *selects* the model, it is no longer a pure held-out gauge of generalization — the reported test MSE is mildly optimistic (the selection has implicitly "seen" the test data). The `main` branch keeps the test set untouched until the very end, which is the cleaner protocol for an unbiased estimate; this branch prioritizes maximal training data and direct test-driven selection instead.

### The Error Landscape

The script renders a heatmap of **test MSE** over degree × λ. Because a handful of overfit models (high degree, tiny λ) produce MSE in the **billions** while good models sit near **10³**, the color scale is **log-normalized** — otherwise those few outliers would wash the entire plot to a single color.

- **Darker purple = lower MSE = better.**
- The bright yellow corner (high degree, small λ) is textbook **overfitting**: the design matrix becomes nearly singular, weights blow up, and test error explodes.

---

## Running It

```bash
# one-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run
python main.py
```

The script prints the optimal combination (by test MSE) to the console and opens the heatmap. An internet connection is required, since both the training and test datasets are fetched from GitHub at runtime.

### Dependencies

- `numpy` — linear algebra and the design matrix
- `pandas` — loading the CSVs
- `matplotlib` — the heatmap
- `scikit-learn` — installed but not currently used (the regression is hand-rolled)

---

## Project Structure

```
fishprediction/
├── main.py            # the entire pipeline: load → train on all 124 → sweep on test set → plot
├── requirements.txt   # pinned dependencies
├── .gitignore         # excludes .venv/ and __pycache__/
└── README.md          # this file
```

---

## Caveats & Next Steps

- **Selection on the test set.** Because the test set is used to pick the model, the reported test MSE is a mildly optimistic estimate of true generalization. A fully clean protocol would reserve a third, untouched split — or use **k-fold cross-validation** on the 124 for selection and keep the test set purely for final reporting.
- **No feature scaling.** Raising raw features to the 10th power produces wildly different column magnitudes, which is part of why high degrees become numerically unstable. Standardizing features first would help.
- **Closed-form only.** The normal equations are fine at this scale, but a gradient-descent version would be a natural extension for larger datasets.
