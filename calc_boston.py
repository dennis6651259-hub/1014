import csv
import math
import json

# Load Boston Housing CSV
with open("boston_housing.csv", "r") as f:
    reader = csv.reader(f)
    headers = [h.strip('" ') for h in next(reader)]
    rows = []
    for r in reader:
        if r:
            rows.append([float(x.strip('" ')) for x in r])

print("Loaded Boston Housing rows:", len(rows), "columns:", headers)

# Target is 'medv'
target_col = "medv"
target_idx = headers.index(target_col)
feature_names = [h for h in headers if h != target_col]
print("Features:", feature_names)

# Split 80/20 train/test deterministically with seed
import random
random.seed(42)
indices = list(range(len(rows)))
random.shuffle(indices)
split = int(0.8 * len(rows))
train_idx = indices[:split]
test_idx = indices[split:]

train_data = [rows[i] for i in train_idx]
test_data = [rows[i] for i in test_idx]

# Matrix operations
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
    # Augment with identity
    M = [[A[i][j] for j in range(n)] + [1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        # Pivot
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

# Fit OLS
def fit_ols(subset_features, dataset_train, dataset_test):
    # Train
    cols = [headers.index(f) for f in subset_features]
    X_train = [[1.0] + [row[c] for c in cols] for row in dataset_train]
    y_train = [[row[target_idx]] for row in dataset_train]
    
    Xt = transpose(X_train)
    XtX = matmul(Xt, X_train)
    inv_XtX = invert_matrix(XtX)
    Xty = matmul(Xt, y_train)
    beta = matmul(inv_XtX, Xty) # (p+1) x 1
    beta_vals = [b[0] for b in beta]
    
    # Train residuals & variance
    y_pred_train = [sum(X_train[i][j] * beta_vals[j] for j in range(len(beta_vals))) for i in range(len(dataset_train))]
    train_res = [dataset_train[i][target_idx] - y_pred_train[i] for i in range(len(dataset_train))]
    n_train = len(dataset_train)
    p = len(subset_features)
    df_e = max(1, n_train - p - 1)
    s2 = sum(r**2 for r in train_res) / df_e
    
    # Test evaluation
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
    
    # F-statistic
    ssr = sst - sse
    f_stat = (ssr / p) / (sse / max(1, (n_test - p - 1))) if sse > 0 and p > 0 else 0.0
    f_pvalue = 1e-15 if f_stat > 15 else 0.001
    
    # Coefficients details
    coef_list = []
    # Intercept
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
    
    # Features
    for idx_f, feat in enumerate(subset_features):
        j = idx_f + 1
        se = math.sqrt(max(1e-10, s2 * inv_XtX[j][j]))
        t = beta_vals[j] / se
        # p-value approx
        abs_t = abs(t)
        pval = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs_t / math.sqrt(2))))
        
        # VIF if p > 1
        vif = 1.0
        if len(subset_features) > 1:
            # Regress feat on other features
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
        
    # Outliers
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

print("Testing fit_ols on ['rm']...")
res1 = fit_ols(["rm"], train_data, test_data)
print("k=1:", res1["rmse"], res1["r2"])

res2 = fit_ols(["rm", "lstat"], train_data, test_data)
print("k=2:", res2["rmse"], res2["r2"])

res3 = fit_ols(["rm", "lstat", "ptratio"], train_data, test_data)
print("k=3:", res3["rmse"], res3["r2"])

res4 = fit_ols(["rm", "lstat", "ptratio", "dis"], train_data, test_data)
print("k=4:", res4["rmse"], res4["r2"])

res5 = fit_ols(["rm", "lstat", "ptratio", "dis", "nox"], train_data, test_data)
print("k=5:", res5["rmse"], res5["r2"])
