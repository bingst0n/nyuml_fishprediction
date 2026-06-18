# 🐟 Fish Weight Prediction

A from-scratch implementation of **polynomial ridge regression** that predicts the weight of a fish from its physical measurements. Built as part of the NYU Machine Learning 2024 course (Session 1, Day 5).

The goal is less about the prediction itself and more about understanding the mechanics: building the design matrix by hand, deriving the ridge-regression weights from the closed-form normal equations, and sweeping over model complexity (polynomial degree) and regularization strength (λ) to watch overfitting happen in real time.

---

## What It Does

1. **Loads** the [Fish Market dataset](https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/) directly from GitHub — 124 fish, 5 measurement features, and weight (grams) as the label.
2. **Splits** the data 80/20 into training and validation sets.
3. **Builds a polynomial design matrix** of arbitrary degree `M` by raising each feature to powers `1…M` (plus a bias term).
4. **Solves for the weights** using the closed-form ridge-regression normal equations — no `sklearn`, just NumPy linear algebra.
5. **Sweeps a grid** of polynomial degrees (1–10) × regularization strengths (λ from 1e-6 to 10) and records validation MSE for every combination.
6. **Reports the optimal** degree/λ pair and visualizes the full error landscape as a heatmap.

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

The best model on the validation set:

| Metric | Value |
|---|---|
| **Optimal degree** | 5 |
| **Optimal λ** | ≈ 1.39 |
| **Validation MSE** | ≈ 1,943 |
| **RMSE** | ≈ 44 g |
| **R²** | ≈ 0.98 |

An RMSE of ~44 g on fish averaging ~390 g (range 0–1600 g) is roughly **11% relative error**, and the model explains about **98% of the variance** in fish weight — far better than the always-predict-the-mean baseline (MSE ≈ 111,600).

### Held-Out Test Set

The grid sweep above uses a validation split to *choose* the model, so its MSE is slightly optimistic. `test.py` gives the honest estimate by retraining the chosen model (degree 5, λ ≈ 1.39) on the training data and scoring it on a completely separate test set ([`fish_market_test_*`](https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/), 31 fish it never saw):

| Metric | Validation | Test |
|---|---|---|
| **MSE** | ≈ 1,943 | ≈ 3,270 |
| **RMSE** | ≈ 44 g | ≈ 57 g |
| **R²** | ≈ 0.98 | ≈ 0.97 |

The test error being a bit higher than validation is expected — the validation set was used to pick the model, so true unseen data is always the fairer judge. An RMSE of ~57 g and R² ≈ 0.97 on data the model never touched confirms it generalizes well rather than just memorizing the training fish.

Run it with:

```bash
python test.py
```

`test.py` imports the model functions directly from `main.py` (which is guarded with `if __name__ == '__main__':` so importing it doesn't trigger the grid sweep or plot).

### The Error Landscape

The script renders a heatmap of validation MSE over degree × λ. Because a handful of overfit models (high degree, tiny λ) produce MSE in the **billions** while good models sit near **10³**, the color scale is **log-normalized** — otherwise those few outliers would wash the entire plot to a single color.

- **Darker purple = lower MSE = better.**
- The bright yellow corner (high degree, small λ) is textbook **overfitting**: the design matrix becomes nearly singular, weights blow up, and validation error explodes.

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

The script prints the optimal combination to the console and opens the MSE heatmap. An internet connection is required, since the dataset is fetched from GitHub at runtime.

### Dependencies

- `numpy` — linear algebra and the design matrix
- `pandas` — loading the CSVs
- `matplotlib` — the heatmap
- `scikit-learn` — installed but not currently used (the regression is hand-rolled)

---

## Project Structure

```
fishprediction/
├── main.py            # the entire pipeline: load → fit → sweep → plot
├── test.py            # evaluates the chosen model on a held-out test set
├── requirements.txt   # pinned dependencies
├── .gitignore         # excludes .venv/ and __pycache__/
└── README.md          # this file
```

---

## Caveats & Next Steps

- **Single split, small data.** Results come from one 80/20 split of just 124 rows (25 validation points), so the exact MSE is a bit noisy. **k-fold cross-validation** would give a more trustworthy estimate.
- **No feature scaling.** Raising raw features to the 10th power produces wildly different column magnitudes, which is part of why high degrees become numerically unstable. Standardizing features first would help.
- **Closed-form only.** The normal equations are fine at this scale, but a gradient-descent version would be a natural extension for larger datasets.
