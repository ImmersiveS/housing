from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

boston = load_boston()
boston.data = scale(boston.data)

kf = KFold(506, n_folds=5, shuffle=True, random_state=42)

p = 0
score = -100
for i in np.linspace(1, 10, 200):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=i)
    current_score = (cross_val_score(neigh, boston.data, boston.target, scoring='mean_squared_error', cv=kf)).mean()

    if max(current_score, score) == current_score:
        score = current_score
        p = i
print('The best parameter p for Minkowski distance metric is ' + str(p))

