import numpy as np

np.random.seed(42)


# Sample data generation
n = 500

# generating predictors
x1 = np.random.randn(n)
z = np.random.randn(n)
x2 = x1 + 0.9 * z

# Design matrix
X = np.column_stack((np.ones(n), x1, x2))
# Regression coefficients matrix
beta_true = np.array([3.0, -4.0, 7.0])
# noise
eps = np.random.randn(n)

# response
y = X @ beta_true + eps


# Condition number
XtX = X.T @ X
cond_number = np.linalg.cond(XtX)

print("Condition number of X^T X:", cond_number)


# Simulating relation btw estimated beta and correlation
alphas = [0.0, 0.4, 0.7, 0.9]

for a in alphas:
    x2_corr = x1 + a * np.random.randn(n)
    X_corr = np.column_stack((np.ones(n), x1, x2_corr))
    XtX_corr = X_corr.T @ X_corr
    var_beta = np.linalg.inv(XtX_corr)
    print(f"a = {a}, Var(beta1) = {var_beta[1,1]:.6f}, Var(beta2) = {var_beta[2,2]:.6f}")


