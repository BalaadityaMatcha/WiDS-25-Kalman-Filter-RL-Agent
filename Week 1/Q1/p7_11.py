import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

data = pd.read_csv("linear_regression_dataset.csv")
X = data.iloc[:,0:12]

X = np.hstack([np.ones((X.shape[0], 1)), X.to_numpy()])

y = data.iloc[:,12]
y = y.to_numpy()

pinvX = np.linalg.pinv(X)

# Part 7
beta_np = pinvX @ y

# sklearn method
X_sklearn = X[:, 1:]
model = LinearRegression(fit_intercept=True)
model.fit(X_sklearn, y)

beta_sklearn = np.concatenate(([model.intercept_], model.coef_))


print("numpy beta:     ", beta_np)
print("sklearn beta:   ", beta_sklearn)
print("Difference:    ", beta_np - beta_sklearn)




# Part 8
y_hat = X @ beta_np
res = y - y_hat

plt.scatter(y_hat, res)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# Part 9
stats.probplot(res, dist="norm", plot=plt)
plt.title("Q–Q Plot of Residuals")
plt.show()

# Part 11
H = X @ pinvX

# Leverage values
leverage = np.diag(H)

n, p_1 = X.shape
leverage_threshold = 2 * p_1 / n

high_leverage = np.where(leverage > leverage_threshold)[0]
print("High leverage points:", high_leverage)

MSE = np.mean(res**2)

cooks_d = (res**2/(p_1 * MSE)) * (leverage/(1 - leverage)**2)

cooks_threshold = 4/n
influential_points = np.where(cooks_d > cooks_threshold)[0]

print("Influential points (Cook's distance):", influential_points)


