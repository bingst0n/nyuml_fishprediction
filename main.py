import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

feature = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/rugvedmhatre/NYU-ML-2024-Session-1/main/day5/fish_market_label.csv')
X = feature.values
y = label.values

print(X.shape, y.shape)

# Hint:
# Do you need to split the dataset?
# How to prediction? Features?
# Polynomial? Order? Would it be overfitting?
# Loss function?
# Plot?

# Go over the past slides and labs!