import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.colors

feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_label.csv')

train_size = int(0.8 * len(feature)) # 80-20 split

X = feature.values[:train_size]
Y = label.values[:train_size]

X_valid = feature.values[train_size:]
Y_valid = label.values[train_size:]

F = X.shape[1]
N = X.shape[0]
N_valid = X_valid.shape[0]

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
    design_valid = mega_design_matrix(X_valid, degree)

    weights = compute_weights(design_train, degree, hyper)

    results = Y_valid - np.matmul(design_valid, weights)
    total1 = 0
    for result in results:
       total1 += result**2
    total1 /= N_valid

    # Lambda omitted here; regularization not in output

    return (total1)[0]

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
    print(f"Optimal: degree {M_range[best[0]]}, lambda {lam_range[best[1]]:.4g}, MSE {finalresults[best]:.2f}, mean MSE {finalresults.mean():.2f}")

    plt.figure(figsize=(10, 6))
    plt.imshow(finalresults, aspect='auto', origin='lower', cmap='viridis',
               norm=matplotlib.colors.LogNorm())
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Degree')
    plt.colorbar(label='MSE')
    plt.title('MSE vs Degree and Lambda')
    plt.show()