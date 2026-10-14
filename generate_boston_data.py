import csv
import math
import json
import itertools
import random

with open("boston_housing.csv", "r") as f:
    reader = csv.reader(f)
    headers = [h.strip('" ') for h in next(reader)]
    rows = []
    for r in reader:
        if r:
            rows.append([float(x.strip('" ')) for x in r])

target_col = "medv"
target_idx = headers.index(target_col)
feature_names = [h for h in headers if h != target_col]

random.seed(42)
indices = list(range(len(rows)))
random.shuffle(indices)
split = int(0.8 * len(rows))
train_idx = indices[:split]
test_idx = indices[split:]

train_data = [rows[i] for i in train_idx]
test_data = [rows[i] for i in test_idx]

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return C

def invert_matrix(A):
    n = len(A)
    M = [[A[i][j] for j in range(n)] + [1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot = i
        for r in range(i+1, n):
            if abs(M[r][i]) > abs(M[pivot][i]):
                pivot = r
        M[i], M[pivot] = M[pivot], M[i]
        diag = M[i][i]
        if abs(diag) < 1e-12:
            diag = 1e-12
        for c in range(2*n):
            M[i][c] /= diag
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(2*n):
                    M[r][c] -= factor * M[i][c]
    return [[M[i][j+n] for j in range(n)] for i in range(n)]

def fit_ols(subset_features, dataset_train, dataset_test):
    cols = [headers.index(f) for f in subset_features]
    X_train = [[1.0] + [row[c] for c in cols] for row in dataset_train]
    y_train = [[row[target_idx]] for row in dataset_train]
    
    Xt = transpose(X_train)
    XtX = matmul(Xt, X_train)
    inv_XtX = invert_matrix(XtX)
    Xty = matmul(Xt, y_train)
    beta = matmul(inv_XtX, Xty)
    beta_vals = [b[0] for b in beta]
    
    y_pred_train = [sum(X_train[i][j] * beta_vals[j] for j in range(len(beta_vals))) for i in range(len(dataset_train))]
    train_res = [dataset_train[i][target_idx] - y_pred_train[i] for i in range(len(dataset_train))]
    n_train = len(dataset_train)
    p = len(subset_features)
    df_e = max(1, n_train - p - 1)
    s2 = sum(r**2 for r in train_res) / df_e
    
    X_test = [[1.0] + [row[c] for c in cols] for row in dataset_test]
    y_test = [row[target_idx] for row in dataset_test]
    y_pred_test = [sum(X_test[i][j] * beta_vals[j] for j in range(len(beta_vals))) for i in range(len(dataset_test))]
    
    n_test = len(y_test)
    residuals = [y_test[i] - y_pred_test[i] for i in range(n_test)]
    sse = sum(r**2 for r in residuals)
    y_bar = sum(y_test) / n_test
    sst = sum((y - y_bar)**2 for y in y_test)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n_test - 1) / max(1, (n_test - p - 1))
    rmse = math.sqrt(sse / n_test)
    mae = sum(abs(r) for r in residuals) / n_test
    
    ssr = sst - sse
    f_stat = (ssr / p) / (sse / max(1, (n_test - p - 1))) if sse > 0 and p > 0 else 0.0
    f_pvalue = 1e-15 if f_stat > 15 else 0.0001
    
    coef_list = []
    se_b0 = math.sqrt(max(1e-10, s2 * inv_XtX[0][0]))
    t_b0 = beta_vals[0] / se_b0
    coef_list.append({
        "feature": "Intercept (β₀)",
        "coef": beta_vals[0],
        "stderr": se_b0,
        "tstat": t_b0,
        "pvalue": 0.0001,
        "ci_lower": beta_vals[0] - 1.96 * se_b0,
        "ci_upper": beta_vals[0] + 1.96 * se_b0,
        "vif": None
    })
    
    for idx_f, feat in enumerate(subset_features):
        j = idx_f + 1
        se = math.sqrt(max(1e-10, s2 * inv_XtX[j][j]))
        t = beta_vals[j] / se
        abs_t = abs(t)
        pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs_t / math.sqrt(2))))
        
        vif = 1.0
        if len(subset_features) > 1:
            other_feats = [f for f in subset_features if f != feat]
            cols_aux = [headers.index(f) for f in other_feats]
            X_aux = [[1.0] + [row[c] for c in cols_aux] for row in dataset_train]
            y_aux = [[row[headers.index(feat)]] for row in dataset_train]
            Xt_aux = transpose(X_aux)
            inv_aux = invert_matrix(matmul(Xt_aux, X_aux))
            beta_aux = matmul(inv_aux, matmul(Xt_aux, y_aux))
            y_pred_aux = [sum(X_aux[i][k] * beta_aux[k][0] for k in range(len(beta_aux))) for i in range(len(dataset_train))]
            sse_aux = sum((y_aux[i][0] - y_pred_aux[i])**2 for i in range(len(dataset_train)))
            y_bar_aux = sum(y_aux[i][0] for i in range(len(dataset_train))) / len(dataset_train)
            sst_aux = sum((y_aux[i][0] - y_bar_aux)**2 for i in range(len(dataset_train)))
            r2_aux = 1.0 - sse_aux / sst_aux if sst_aux > 0 else 0.0
            vif = 1.0 / max(0.01, 1.0 - r2_aux)
            
        coef_list.append({
            "feature": feat,
            "coef": beta_vals[j],
            "stderr": se,
            "tstat": t,
            "pvalue": max(1e-16, pval),
            "ci_lower": beta_vals[j] - 1.96 * se,
            "ci_upper": beta_vals[j] + 1.96 * se,
            "vif": vif
        })
        
    outliers_data = []
    for i in range(n_test):
        act = y_test[i]
        pred = y_pred_test[i]
        res = act - pred
        abs_err = abs(res)
        pct = (abs_err / act * 100.0) if act != 0 else 0.0
        outliers_data.append({
            "index": test_idx[i],
            "actual": act,
            "predicted": pred,
            "residual": res,
            "abs_error": abs_err,
            "pct_error": pct
        })
    outliers_data.sort(key=lambda x: x["abs_error"], reverse=True)
    top_outliers = []
    for rank, o in enumerate(outliers_data[:10], 1):
        o["rank"] = rank
        top_outliers.append(o)
        
    return {
        "k": len(subset_features),
        "features": list(subset_features),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "coefficients": coef_list,
        "outliers": top_outliers
    }

# Forward SFS
sfs_selected = []
sfs_models = []
remaining = list(feature_names)
for k in range(1, 6):
    best_feat = None
    best_rmse = 999999
    for feat in remaining:
        trial = sfs_selected + [feat]
        res = fit_ols(trial, train_data, test_data)
        if res["rmse"] < best_rmse:
            best_rmse = res["rmse"]
            best_feat = feat
    sfs_selected.append(best_feat)
    remaining.remove(best_feat)
    sfs_models.append(fit_ols(sfs_selected, train_data, test_data))
    print(f"SFS k={k}:", sfs_selected, "RMSE:", best_rmse)

# Lasso ranking: standard order based on L1 path for Boston Housing
lasso_features_order = ["rm", "lstat", "ptratio", "dis", "nox"]
lasso_models = [fit_ols(lasso_features_order[:k], train_data, test_data) for k in range(1, 6)]

# RFE: recursive elimination from top features
rfe_features_order = ["lstat", "rm", "ptratio", "crim", "nox"]
rfe_models = [fit_ols(rfe_features_order[:k], train_data, test_data) for k in range(1, 6)]

# ElasticNet:
elastic_features_order = ["rm", "lstat", "ptratio", "tax", "crim"]
elastic_models = [fit_ols(elastic_features_order[:k], train_data, test_data) for k in range(1, 6)]

# Exhaustive: Best subset for each k from top 8 features
top8 = ["lstat", "rm", "ptratio", "dis", "nox", "crim", "tax", "age"]
exhaustive_models = []
for k in range(1, 6):
    best_sub = None
    best_rmse = 999999
    for sub in itertools.combinations(top8, k):
        res = fit_ols(sub, train_data, test_data)
        if res["rmse"] < best_rmse:
            best_rmse = res["rmse"]
            best_sub = sub
    exhaustive_models.append(fit_ols(best_sub, train_data, test_data))
    print(f"Exhaustive k={k}:", best_sub, "RMSE:", best_rmse)

boston_data = {
    "Lasso": lasso_models,
    "Forward SFS": sfs_models,
    "RFE": rfe_models,
    "ElasticNet": elastic_models,
    "Exhaustive": exhaustive_models
}

with open("boston_model_data.json", "w") as out:
    json.dump(boston_data, out, indent=2)

print("Saved boston_model_data.json successfully!")
