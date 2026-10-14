import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
df = pd.read_csv(url)

# One-hot encode State (California dropped as reference to match 5 features)
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

# Split matching the benchmark: random_state=0, test_size=0.20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=0
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

def evaluate_subset(features_subset):
    sub = list(features_subset)
    model = LinearRegression()
    model.fit(X_train[sub], y_train)
    y_pred = model.predict(X_test[sub])
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

algorithms = {}

# 1. Lasso Ranking (by standardized coefficient magnitude)
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=100000).fit(X_train_scaled, y_train)
lasso_rank = [f for _, f in sorted(zip(abs(lasso_cv.coef_), features), reverse=True)]
algorithms["Lasso (L1 Ranking)"] = [lasso_rank[:k] for k in range(1, 6)]

# 2. Forward SFS
sfs_subsets = []
for k in range(1, 6):
    if k == 5:
        sfs_subsets.append(features)
    else:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward")
        sfs.fit(X_train, y_train)
        sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
algorithms["Forward SFS"] = sfs_subsets

# 3. Recursive Feature Elimination (RFE)
rfe_subsets = []
for k in range(1, 6):
    if k == 5:
        rfe_subsets.append(features)
    else:
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_scaled, y_train)
        rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
algorithms["RFE"] = rfe_subsets

# 4. ElasticNet Ranking
enet_cv = ElasticNetCV(cv=5, random_state=0, max_iter=100000).fit(X_train_scaled, y_train)
enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
algorithms["ElasticNet Ranking"] = [enet_rank[:k] for k in range(1, 6)]

# 5. Exhaustive (Best Subset Selection per k)
exhaustive_subsets = []
for k in range(1, 6):
    best_rmse = float("inf")
    best_sub = None
    for sub in itertools.combinations(features, k):
        rmse, _ = evaluate_subset(sub)
        if rmse < best_rmse:
            best_rmse = rmse
            best_sub = list(sub)
    exhaustive_subsets.append(best_sub)
algorithms["Exhaustive Search"] = exhaustive_subsets

# Evaluate and compile results
all_dfs = {}
for name, subsets in algorithms.items():
    records = []
    for k, sub in enumerate(subsets, start=1):
        rmse, r2 = evaluate_subset(sub)
        records.append({
            "Number of Features": k,
            "Selected Features": str(sub),
            "RMSE": rmse,
            "R-squared": r2
        })
    df_res = pd.DataFrame(records)
    all_dfs[name] = df_res
    csv_filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_progression.csv"
    df_res.to_csv(csv_filename, index=False)
    print(f"\n===== {name} =====")
    print(df_res.to_string(index=False))

    # Plot individual 2-panel chart matching user's image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: RMSE
    ax1.plot(df_res["Number of Features"], df_res["RMSE"], marker='o', linewidth=2, color='#1f77b4')
    ax1.set_title("RMSE by Number of Features", fontsize=12, pad=10)
    ax1.set_xlabel("Number of Features", fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=11)
    ax1.set_xticks(range(1, 6))
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Right: R-squared
    ax2.plot(df_res["Number of Features"], df_res["R-squared"], marker='o', linewidth=2, color='#1f77b4')
    ax2.set_title("R-squared by Number of Features", fontsize=12, pad=10)
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("R-squared", fontsize=11)
    ax2.set_xticks(range(1, 6))
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_plot.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

# Combined Comparison Plot across all 5 algorithms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

for (name, df_res), color, marker in zip(all_dfs.items(), colors, markers):
    ax1.plot(df_res["Number of Features"], df_res["RMSE"], marker=marker, linewidth=1.8, label=name, color=color)
    ax2.plot(df_res["Number of Features"], df_res["R-squared"], marker=marker, linewidth=1.8, label=name, color=color)

ax1.set_title("RMSE Comparison across 5 Feature Selection Methods", fontsize=12, pad=10)
ax1.set_xlabel("Number of Features", fontsize=11)
ax1.set_ylabel("RMSE (Lower is Better)", fontsize=11)
ax1.set_xticks(range(1, 6))
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.set_title("R-squared Comparison across 5 Feature Selection Methods", fontsize=12, pad=10)
ax2.set_xlabel("Number of Features", fontsize=11)
ax2.set_ylabel("R-squared (Higher is Better)", fontsize=11)
ax2.set_xticks(range(1, 6))
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("all_5_algorithms_comparison_plot.png", dpi=300)
plt.close()

print("\nAll plots and CSVs successfully generated!")
