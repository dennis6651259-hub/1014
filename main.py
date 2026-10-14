
# ============================================================
# 50 STARTUPS - CRISP-DM MULTIPLE LINEAR REGRESSION
#
# Model B: 5 Features + Intercept
#
# Feature Selection:
# 1. LassoCV
# 2. RFECV
# 3. Sequential Forward Selection (SFS)
# 4. ElasticNetCV
# 5. Exhaustive Feature Selection
#
# Plus: Baseline, VIF, CV, Metrics, Plots, Top 10 Outliers
#
# pip install numpy pandas matplotlib scikit-learn statsmodels
# ============================================================

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score
)
from sklearn.linear_model import (
    LinearRegression, LassoCV, ElasticNetCV
)
from sklearn.feature_selection import (
    RFECV, SequentialFeatureSelector
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)
import statsmodels.api as sm


RANDOM_STATE = 42
CV = KFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# PHASE 1: BUSINESS UNDERSTANDING
# ============================================================

print("PHASE 1: BUSINESS UNDERSTANDING")
print("Goal: Predict startup profit")
print("Compare 5 feature selection methods against OLS")


# ============================================================
# PHASE 2: DATA UNDERSTANDING
# ============================================================

url = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)

df = pd.read_csv(url)

print("\nPHASE 2: DATA UNDERSTANDING")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())


# ============================================================
# PHASE 3: DATA PREPARATION
# ============================================================

# California is reference category
df_enc = pd.get_dummies(df, columns=["State"], dtype=int)

features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York"
]

X = df_enc[features].astype(float)
y = df_enc["Profit"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nPHASE 3: DATA PREPARATION")
print("Features:", features)
print("Train size:", len(X_train))
print("Test size:", len(X_test))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metrics(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)

    adjusted = (
        1 - (1-r2)*(n-1)/(n-p-1)
        if n > p+1 else np.nan
    )

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2,
        "Adjusted_R2": adjusted
    }


def evaluate_subset(name, selected_features):
    """
    Fit OLS using the selected features.
    Cross-validation scores here evaluate a FIXED subset,
    not the full feature-selection procedure.
    """
    selected_features = list(selected_features)

    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)

    predictions = model.predict(X_test[selected_features])

    cv_rmse = -cross_val_score(
        LinearRegression(),
        X_train[selected_features],
        y_train,
        cv=CV,
        scoring="neg_root_mean_squared_error"
    )

    result = metrics(
        y_test, predictions, len(selected_features)
    )

    result.update({
        "Method": name,
        "N_Features": len(selected_features),
        "Selected_Features": ", ".join(selected_features),
        "CV_RMSE": cv_rmse.mean()
    })

    return result, model, predictions


results = []
fitted_models = {}


def register(name, selected):
    result, model, predictions = evaluate_subset(name, selected)
    results.append(result)
    fitted_models[name] = {
        "model": model,
        "features": list(selected),
        "predictions": predictions
    }
    print(f"\n{name}")
    print("Selected:", list(selected))
    print("CV RMSE:", round(result["CV_RMSE"], 2))
    print("Test R2:", round(result["R2"], 4))


# ============================================================
# PHASE 4: MODELING
# ============================================================

print("\nPHASE 4: MODELING")


# ------------------------------------------------------------
# MODEL 0: BASELINE OLS
# ------------------------------------------------------------

register("Baseline OLS", features)


# ------------------------------------------------------------
# MODEL 1: LassoCV Feature Selection
# ------------------------------------------------------------

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        cv=CV,
        max_iter=100000,
        random_state=RANDOM_STATE
    ))
])

lasso_pipe.fit(X_train, y_train)

lasso_coef = lasso_pipe.named_steps["lasso"].coef_
lasso_selected = [
    f for f, c in zip(features, lasso_coef)
    if abs(c) > 1e-8
]

if not lasso_selected:
    lasso_selected = [features[np.argmax(abs(lasso_coef))]]

register("LassoCV", lasso_selected)


# ------------------------------------------------------------
# MODEL 2: RFECV
# ------------------------------------------------------------

rfecv = RFECV(
    estimator=LinearRegression(),
    step=1,
    cv=CV,
    scoring="neg_root_mean_squared_error",
    min_features_to_select=1
)

# Scaling keeps numerical features comparable.
# For OLS, coefficient magnitudes otherwise depend on units.
rfecv_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", rfecv)
])

rfecv_pipe.fit(X_train, y_train)

rfecv_selected = list(
    X_train.columns[
        rfecv_pipe.named_steps["selector"].support_
    ]
)

register("RFECV", rfecv_selected)


# ------------------------------------------------------------
# MODEL 3: Sequential Forward Selection
# ------------------------------------------------------------

# Choose best number of features (1..5)
# using the same training CV folds.
# For k=5, all features are retained.

best_sfs_score = np.inf
best_sfs_features = None

for k in range(1, len(features) + 1):

    if k == len(features):
        selected = features
    else:
        sfs = SequentialFeatureSelector(
            estimator=LinearRegression(),
            n_features_to_select=k,
            direction="forward",
            scoring="neg_root_mean_squared_error",
            cv=CV
        )

        sfs.fit(X_train, y_train)
        selected = list(X_train.columns[sfs.get_support()])

    score = -cross_val_score(
        LinearRegression(),
        X_train[selected],
        y_train,
        cv=CV,
        scoring="neg_root_mean_squared_error"
    ).mean()

    if score < best_sfs_score:
        best_sfs_score = score
        best_sfs_features = selected

register("Forward SFS", best_sfs_features)


# ------------------------------------------------------------
# MODEL 4: ElasticNetCV
# ------------------------------------------------------------

elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic", ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=CV,
        max_iter=100000,
        random_state=RANDOM_STATE
    ))
])

elastic_pipe.fit(X_train, y_train)

elastic_coef = elastic_pipe.named_steps["elastic"].coef_

elastic_selected = [
    f for f, c in zip(features, elastic_coef)
    if abs(c) > 1e-8
]

if not elastic_selected:
    elastic_selected = [
        features[np.argmax(abs(elastic_coef))]
    ]

register("ElasticNetCV", elastic_selected)


# ------------------------------------------------------------
# MODEL 5: Exhaustive Feature Selection
# ------------------------------------------------------------

best_exhaustive_score = np.inf
best_exhaustive_features = None

for k in range(1, len(features) + 1):
    for subset in itertools.combinations(features, k):

        selected = list(subset)

        cv_rmse = -cross_val_score(
            LinearRegression(),
            X_train[selected],
            y_train,
            cv=CV,
            scoring="neg_root_mean_squared_error"
        ).mean()

        if cv_rmse < best_exhaustive_score:
            best_exhaustive_score = cv_rmse
            best_exhaustive_features = selected

register("Exhaustive", best_exhaustive_features)


# ============================================================
# PHASE 5: EVALUATION
# ============================================================

print("\nPHASE 5: EVALUATION")

results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "Method",
        "N_Features",
        "Selected_Features",
        "CV_RMSE",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "Adjusted_R2"
    ]
]

results_df = results_df.sort_values("CV_RMSE")

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "feature_selection_comparison.csv",
    index=False
)

best_method = results_df.iloc[0]["Method"]
best_features = fitted_models[best_method]["features"]
best_model = fitted_models[best_method]["model"]

print("\nBest method by training CV RMSE:", best_method)
print("Selected features:", best_features)


# ============================================================
# VIF CHECK
# ============================================================

print("\n=== VIF: SELECTED FEATURES ===")

X_vif = sm.add_constant(
    X_train[best_features],
    has_constant="add"
)

vif_table = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})

print(vif_table.round(3))


# ============================================================
# PLOT 1: FEATURE SELECTION COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    results_df["Method"],
    results_df["CV_RMSE"]
)

plt.xlabel("Cross-Validated RMSE (Lower is Better)")
plt.title("50 Startups: Feature Selection Comparison")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_selection_comparison.png", dpi=300)
plt.show()


# ============================================================
# PLOT 2: ACTUAL VS PREDICTED (TEST SET)
# ============================================================

test_pred = best_model.predict(X_test[best_features])

plt.figure(figsize=(8, 7))

plt.scatter(
    y_test / 1000,
    test_pred / 1000,
    s=80,
    alpha=0.8
)

lower = min(y_test.min(), test_pred.min()) / 1000
upper = max(y_test.max(), test_pred.max()) / 1000

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Perfect Prediction (y=x)"
)

plt.xlabel("Actual Profit ($ thousands)")
plt.ylabel("Predicted Profit ($ thousands)")
plt.title(f"Actual vs Predicted: {best_method}")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()


# ============================================================
# TOP 10 OUTLIERS: DESCRIPTIVE FULL-DATA REFIT
# ============================================================

# This is NOT a held-out evaluation.
# Refit the selected feature set to all 50 observations
# for descriptive residual analysis.

full_model = LinearRegression()
full_model.fit(X[best_features], y)

outliers = df.copy()
outliers["Predicted_Profit"] = full_model.predict(
    X[best_features]
)

outliers["Residual"] = (
    outliers["Profit"] - outliers["Predicted_Profit"]
)

outliers["Absolute_Residual"] = outliers["Residual"].abs()

top10 = outliers.nlargest(
    10, "Absolute_Residual"
).copy()

top10.insert(
    0, "Rank",
    np.arange(1, len(top10) + 1)
)

top10.insert(1, "Original_Index", top10.index)

print("\n=== TOP 10 ABSOLUTE RESIDUALS ===")
print(
    top10[
        [
            "Rank", "Original_Index", "State",
            "Profit", "Predicted_Profit",
            "Residual", "Absolute_Residual"
        ]
    ].round(2).to_string(index=False)
)

top10.to_csv("top10_outliers.csv", index=False)


# ============================================================
# PLOT 3: SCATTER + TOP 10 OUTLIERS
# ============================================================

colors = {
    "California": "blue",
    "Florida": "green",
    "New York": "red"
}

plt.figure(figsize=(10, 8))

for state, color in colors.items():
    sub = outliers[outliers["State"] == state]

    plt.scatter(
        sub["Profit"] / 1000,
        sub["Predicted_Profit"] / 1000,
        label=state,
        color=color,
        alpha=0.75
    )

# Perfect prediction reference
min_val = min(
    outliers["Profit"].min(),
    outliers["Predicted_Profit"].min()
) / 1000

max_val = max(
    outliers["Profit"].max(),
    outliers["Predicted_Profit"].max()
) / 1000

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "k--",
    label="y = x"
)

# Descriptive trend line
slope, intercept = np.polyfit(
    outliers["Profit"] / 1000,
    outliers["Predicted_Profit"] / 1000,
    1
)

line_x = np.linspace(min_val, max_val, 200)
plt.plot(
    line_x,
    slope * line_x + intercept,
    color="orange",
    linewidth=2,
    label="Scatter Trend Line"
)

# Top 10 largest residuals
for _, row in top10.iterrows():
    xx = row["Profit"] / 1000
    yy = row["Predicted_Profit"] / 1000

    plt.scatter(
        xx, yy,
        s=220,
        facecolors="none",
        edgecolors="red",
        linewidths=2
    )

    plt.annotate(
        str(int(row["Rank"])),
        (xx, yy),
        xytext=(6, 7),
        textcoords="offset points",
        fontweight="bold"
    )

plt.xlabel("Actual Profit ($ thousands)")
plt.ylabel("Predicted Profit ($ thousands)")
plt.title(
    f"50 Startups: {best_method}\n"
    "Top 10 Largest Absolute Residuals"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("top10_outliers_plot.png", dpi=300)
plt.show()


# ============================================================
# PHASE 6: DEPLOYMENT
# ============================================================

print("\nPHASE 6: DEPLOYMENT")

# New startup located in California
new_startup = pd.DataFrame([{
    "R&D Spend": 100000,
    "Administration": 120000,
    "Marketing Spend": 150000,
    "State_Florida": 0,
    "State_New York": 0
}])

# Deployment model refitted on all available data
predicted_profit = full_model.predict(
    new_startup[best_features]
)[0]

print(f"Predicted new startup profit: ${predicted_profit:,.2f}")

print("\nFiles saved:")
print("feature_selection_comparison.csv")
print("feature_selection_comparison.png")
print("actual_vs_predicted.png")
print("top10_outliers.csv")
print("top10_outliers_plot.png")
