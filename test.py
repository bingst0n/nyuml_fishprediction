from main import np, pd, mega_design_matrix, compute_weights, X as X_train

feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_test_label.csv')

X = feature.values
Y = label.values

M = 5
lam = 1.389

print('feature shape:', X.shape)
print('label shape:', Y.shape)


def test_metrics(M, lam):
    weights = compute_weights(mega_design_matrix(X_train, M), M, lam)
    preds = mega_design_matrix(X, M) @ weights
    mse = np.mean((Y - preds) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((Y - preds) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return mse, rmse, r2


mse, rmse, r2 = test_metrics(M, lam)
print(f'Test (degree {M}, lambda {lam}): MSE {mse}, RMSE {rmse}, R^2 {r2}')
