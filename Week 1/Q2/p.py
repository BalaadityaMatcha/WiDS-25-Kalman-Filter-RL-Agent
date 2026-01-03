import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error , mean_absolute_error, r2_score
from scipy.stats import ttest_ind

import statsmodels.api as sm


# Part 1
df = pd.read_csv("salary_dataset.csv")

# Univariate Statistics
# Basic info
print(df.info())
print("------------------------------------")
print(df.describe())
print("------------------------------------")

# Salary distribution
plt.hist(df["salary"], bins=50)
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()

# Others
# Salary by gender
sns.boxplot(x="gender", y="salary", data=df)
plt.title("Salary by Gender")
plt.show()

# Salary by education
sns.boxplot(x="education_level", y="salary", data=df)
plt.title("Salary by Education Level")
plt.show()

# Salary by industry
sns.boxplot(x="industry", y="salary", data=df)
plt.title("Salary by Industry")
plt.show()

# Correlation heatmap (numerical features)
num_cols = df.select_dtypes(include=np.number)
sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Part 2
print("PART @2")

# Fill numeric misses with median
num_features = df.select_dtypes(include=np.number).columns
df[num_features] = df[num_features].fillna(df[num_features].median())

# Fill categorical misses with mode
cat_features = df.select_dtypes(exclude=np.number).columns
df[cat_features] = df[cat_features].fillna(df[cat_features].mode().iloc[0])

# Outlier handling using IQR for salary
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1

df = df[(df["salary"] >= Q1 - 1.5*IQR) & (df["salary"] <= Q3 + 1.5*IQR)]


# Part 3
print("PART @3")
X = df.drop("salary", axis=1)
y = df["salary"]

categorical_cols = X.select_dtypes(exclude=np.number).columns
numerical_cols = X.select_dtypes(include=np.number).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)
print("PART @3 - DONE")


# Part 4
print("PART @4")
# Stratify by gender
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=X["gender"], random_state=42
)
print("PART @4 - DONE")


# Part 5
print("PART @5")
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
print("PART @5 - DONE")


# Part 6
print("PART @6")
X_processed = preprocessor.fit_transform(X)
X_processed = X_processed.toarray()
X_sm = sm.add_constant(X_processed)

ols_model = sm.OLS(y, X_sm).fit()
print(ols_model.summary())
print("PART @6 - DONE")
print()


# Part 7
print("PART @7")
feature_names = preprocessor.get_feature_names_out()
feature_names = np.insert(feature_names, 0, "Intercept")

coef_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": ols_model.params,
    "P-value": ols_model.pvalues
})

print(coef_table.sort_values(by="Coefficient", key=abs, ascending=False).head(5))


print("PART @7 - DONE")
print()


# Part 8
print("PART @8")
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)
print("PART @8 - DONE")
print()


# Part 9
print("PART @9")
residuals = y_test - y_pred

# Residuals vs Fitted
plt.scatter(y_pred, residuals)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# Q-Q plot
sm.qqplot(residuals, line="45")
plt.title("Q-Q Plot of Residuals")
plt.show()
print("PART @9 - DONE")


# Part 10 (AI Code)
# Add predictions and residuals to test set
test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred
test_df["residual"] = test_df["y_true"] - test_df["y_pred"]

def group_metrics(df, group_col, group_a, group_b):
    df_a = df[df[group_col] == group_a]
    df_b = df[df[group_col] == group_b]

    metrics = {}

    # (a) Mean Salary Prediction Difference
    # This remains the difference in raw average predicted salary
    metrics["Mean Prediction Difference"] = (
        df_a["y_pred"].mean() - df_b["y_pred"].mean()
    )

    # (b) Mean Absolute Error per group
    metrics["MAE_" + group_a] = np.mean(np.abs(df_a["residual"]))
    metrics["MAE_" + group_b] = np.mean(np.abs(df_b["residual"]))

    # --- DEFINE THRESHOLD ---
    # Moved up so it can be used for DPD and DIR as well.
    # We define "positive outcome" as being in the top 25% of earners.
    threshold = df["y_true"].quantile(0.75)

    # Calculate Selection Rates (SR): The % of the group PREDICTED to have high salary
    # DPD and DIR are based on these rates, not the raw means.
    sr_a = np.mean(df_a["y_pred"] >= threshold)
    sr_b = np.mean(df_b["y_pred"] >= threshold)

    # (c) Demographic Parity Difference (DPD)
    # Difference in selection rates (P(pred > thresh | A) - P(pred > thresh | B))
    metrics["DPD"] = sr_a - sr_b

    # (d) Equal Opportunity Difference (EOD)
    # TPR Difference: P(pred > thresh | true > thresh, A) - P(pred > thresh | true > thresh, B)
    tpr_a = np.mean(df_a[df_a["y_true"] >= threshold]["y_pred"] >= threshold)
    tpr_b = np.mean(df_b[df_b["y_true"] >= threshold]["y_pred"] >= threshold)
    metrics["EOD"] = tpr_a - tpr_b

    # (e) Predictive Equality
    # FPR Difference
    fpr_a = np.mean(df_a[df_a["y_true"] < threshold]["y_pred"] >= threshold)
    fpr_b = np.mean(df_b[df_b["y_true"] < threshold]["y_pred"] >= threshold)
    metrics["Predictive Equality"] = fpr_a - fpr_b

    # (f) Disparate Impact Ratio (DIR)
    # Ratio of selection rates (SR_a / SR_b)
    # Added a small check to avoid division by zero error
    metrics["DIR"] = sr_a / sr_b if sr_b > 0 else 0.0

    return metrics



metrics_mf = group_metrics(test_df, "gender", "Male", "Female")
metrics_mo = group_metrics(test_df, "gender", "Male", "Other")

print("Male vs Female:", metrics_mf)
print("Male vs Other:", metrics_mo)

plt.figure()
sns.boxplot(x="gender", y="residual", data=test_df)
plt.title("Residual Distribution by Gender")
plt.savefig("part10_residual_distribution_by_gender.png",
            dpi=300, bbox_inches="tight")
plt.close()



# Part 11
res_male = test_df[test_df["gender"] == "Male"]["residual"]
res_female = test_df[test_df["gender"] == "Female"]["residual"]

t_stat, p_val = ttest_ind(res_male, res_female, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_val)


# Part 12
mean_residuals = test_df.groupby("gender")["residual"].mean()
print(mean_residuals)
estimations = ["Overestimated" if val < 0 else "Underestimated" for val in mean_residuals]
print(estimations)


# Part 13
linear_model = model.named_steps["regressor"]

# 1. Preparing Background Data (MUST BE TRAINING DATA)
# We need the training distribution to serve as the baseline
X_train_processed = model.named_steps["preprocessor"].transform(X_train)
X_train_processed = X_train_processed.toarray()

# 2. Preparing Test Data (For which we want explanations)
X_test_processed = model.named_steps["preprocessor"].transform(X_test)
X_test_processed = X_test_processed.toarray()

# 3. Initializing the Explainer with TRAINING data
explainer = shap.LinearExplainer(linear_model, X_train_processed)

# 4. Calculating SHAP values for TEST data
shap_values = explainer.shap_values(X_test_processed)

feature_names = model.named_steps["preprocessor"].get_feature_names_out()


plt.figure()
shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names,
    show=False
)
plt.savefig("part13_shap_summary_plot.png",
            dpi=300, bbox_inches="tight")
plt.close()



for feature in feature_names[:3]:
    plt.figure()
    shap.dependence_plot(
        feature,
        shap_values,
        X_test_processed,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(f"part13_shap_dependence_{feature}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

