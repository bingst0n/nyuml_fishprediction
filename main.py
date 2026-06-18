import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors

feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_label.csv')

test_feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_feature.csv')
test_label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_label.csv')

# Train on all 124 instances; evaluate on the separate test set.
X = feature.values
Y = label.values

X_test = test_feature.values
Y_test = test_label.values

F = X.shape[1]
N = X.shape[0]

def mega_design_matrix(xvals, M: int):
  rows = xvals.shape[0]
  design_matrix = np.zeros((rows, F*M+1))

  for i in range(0, rows):
    design_matrix_row = [1]

    for j in range(1,M+1):
      for k in range(0, F):
        design_matrix_row.append(xvals[i][k]**j)

    design_matrix[i] = design_matrix_row

  return design_matrix

def compute_weights(design, degree, hyper):
  phi = design
  phiT = phi.transpose()

  D = F * degree + 1

  weights = np.matmul(
    np.linalg.inv(
      np.matmul(phiT, phi) + N * hyper * np.eye(D)
    ),
    np.matmul(
      phiT, Y
    )
  )

  return weights

M_range = range(1, 11)
lam_range = np.logspace(-6, 1, 50)

def evaluate_model(degree, hyper):
    design_train = mega_design_matrix(X, degree)
    design_test = mega_design_matrix(X_test, degree)

    weights = compute_weights(design_train, degree, hyper)

    preds = np.matmul(design_test, weights)
    mse = np.mean((Y_test - preds) ** 2)

    return mse

def bulk_test(degs, lams):
    degree_lambdas_results = np.zeros((len(degs), len(lams)))
    for di, deg in enumerate(degs):
       for li, lam in enumerate(lams):
          print(f"Evaluating degree {deg}, lambda {lam}")
          degree_lambdas_results[di][li] = evaluate_model(deg, lam)
    return degree_lambdas_results

if __name__ == '__main__':
    finalresults = bulk_test(M_range, lam_range)

    best = np.unravel_index(np.argmin(finalresults), finalresults.shape)
    print(f"Optimal (by test MSE): degree {M_range[best[0]]}, lambda {lam_range[best[1]]:.4g}, test MSE {finalresults[best]:.2f}, mean test MSE {finalresults.mean():.2f}")

    plt.figure(figsize=(10, 6))
    plt.imshow(finalresults, aspect='auto', origin='lower', cmap='viridis',
               norm=matplotlib.colors.LogNorm())
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Degree')
    plt.colorbar(label='Test MSE')
    plt.title('Test MSE vs Degree and Lambda')
    plt.show()