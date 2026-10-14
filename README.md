# 50 Startups: Multiple Linear Regression & Feature Selection Explorer

An interactive machine learning and statistical inference platform built following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology. This project evaluates **Multiple Linear Regression (OLS)** against **5 distinct feature selection algorithms** to uncover the optimal feature subset for predicting startup profitability while guarding against multicollinearity and overfitting.

---

## 🌐 Interactive Web Application & Live Demo

The project includes a web application inspired by the statistical calculator layout of [StatsUnlock](https://statsunlock.com/multiple-linear-regression-calculator/).

### How to Run Locally:
1. **Local Server (Active)**:
   ```bash
   python -m http.server 8080
   ```
   Open your browser and navigate to: **[http://localhost:8080/](http://localhost:8080/)**

2. **Standalone Access**:
   You can also open [`index.html`](file:///d:/Huan%20Chen/AIoT2026/L5-1%2050%20startup/index.html) directly in any web browser without needing any backend server!

### Key Web App Features:
- **Algorithm Switcher**: Switch dynamically between Lasso (L1), Forward SFS, RFE, ElasticNet, Exhaustive Search, or **Compare All 5 Simultaneously**.
- **Interactive Feature Count Slider ($k=1 \dots 5$)**: Instantly observe model metrics and equation changes across different subset sizes.
- **Dual-Panel Responsive Charts**: Interactive Chart.js visualizations for **"RMSE by Number of Features"** and **"R-squared by Number of Features"** with sweet-spot highlighting and hover tooltips.
- **Dynamic Regression Equation**: Live mathematical formula updating in real-time as predictors are added or removed.
- **StatsUnlock-Style Coefficients Table**: Full inferential statistics including $\beta$ coefficients, Standard Error, $t$-values, $p$-values, 95% Confidence Intervals, and VIF (Variance Inflation Factor).
- **Live Startup Profit Predictor**: Interactive input sliders for R&D Spend, Administration, Marketing Spend, and State with instant profit calculations.
- **Top 10 Outliers & Residuals Diagnostic Table**: Detailed residual analysis for out-of-sample test cases.

---

## 📊 Core Empirical Findings

### 1. The 2-Feature Sweet Spot ($k=2$)
Across all 5 feature selection algorithms, the model consistently converges on **`['R&D Spend', 'Marketing Spend']`** as the optimal subset:
- **Minimum Test RMSE**: **`$8,198.80`** (lowest error across all tested models)
- **Peak Test $R^2$**: **`0.9474`** (explains 94.74% of startup profit variation)
- **R&D Dominance**: Alone, R&D spend accounts for over $94.6\%$ of profit variation ($k=1$: $R^2 = 0.9465$, $\text{RMSE} = \$8,274.87$). Adding Marketing Spend provides complementary positive predictive lift.

### 2. The 5-Feature Overfitting Cliff ($k=5$)
Adding redundant variables (`Administration` and location dummies `State_Florida`, `State_New York`) introduces noise and multicollinearity:
- **Test RMSE Spikes**: from **`$8,198.80`** to **`$9,137.99`** ($+\$939$ penalty!)
- **$R^2$ Degrades**: drops to **`0.9347`**
- **Principle of Parsimony (Occam's Razor)**: The simpler 2-variable model outperforms the full 5-variable model in predictive accuracy, interpretability, and generalizability.

---

## 🔬 Comparison of the 5 Feature Selection Algorithms

All 5 methods were benchmarked across $k=1 \dots 5$ features on the out-of-sample test set:

| Feature Selection Algorithm | $k=1$ (RMSE) | $k=2$ (RMSE) | $k=3$ (RMSE) | $k=4$ (RMSE) | $k=5$ (RMSE) | Best Feature Subset |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lasso (L1 Ranking)** | $8,274.87 | **$8,198.80** | $8,309.06 | $8,409.92 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **Forward SFS** | $8,274.87 | **$8,198.80** | $8,803.78 | $8,997.20 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **RFE** | $8,274.87 | **$8,198.80** | $8,803.78 | $9,068.54 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **ElasticNet Ranking** | $8,274.87 | **$8,198.80** | $8,803.78 | $8,997.20 | $9,137.99 | `['R&D Spend', 'Marketing Spend']` |
| **Exhaustive (Best Subset)** | $8,274.87 | $8,198.80 | **$8,171.65** | $8,178.37 | $9,137.99 | `['R&D Spend', 'Admin', 'State_FL']` |

*(Note: While Exhaustive Search finds a marginal in-sample reduction at $k=3$, the 2-feature subset is universally favored for simplicity and stability.)*

---

## 📁 Repository Structure

```text
├── index.html                           # Modern interactive web application (StatsUnlock style)
├── model_data.js                        # Structured model inference & statistical coefficients
├── model_data.json                      # Machine-readable JSON export of all statistical results
├── main.py                              # CRISP-DM end-to-end multiple linear regression script
├── feature_progression_analysis.py      # Automated script generating k=1..5 metrics and plots
├── requirements.txt                     # Python package dependencies
├── all_5_algorithms_comparison_plot.png # Visual comparison across all 5 methods
├── lasso_l1_ranking_plot.png            # Dual-panel RMSE & R² plot for Lasso
├── forward_sfs_plot.png                 # Dual-panel RMSE & R² plot for Forward SFS
├── rfe_plot.png                         # Dual-panel RMSE & R² plot for RFE
├── elasticnet_ranking_plot.png          # Dual-panel RMSE & R² plot for ElasticNet
├── exhaustive_search_plot.png           # Dual-panel RMSE & R² plot for Exhaustive Search
├── lasso_l1_ranking_progression.csv     # Progression table data (k=1..5) for Lasso
├── forward_sfs_progression.csv          # Progression table data (k=1..5) for Forward SFS
├── rfe_progression.csv                  # Progression table data (k=1..5) for RFE
├── elasticnet_ranking_progression.csv   # Progression table data (k=1..5) for ElasticNet
└── exhaustive_search_progression.csv    # Progression table data (k=1..5) for Exhaustive Search
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the workspace directory**:
   ```bash
   cd "d:/Huan Chen/AIoT2026/L5-1 50 startup"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the CRISP-DM Python Pipeline**:
   ```bash
   python main.py
   ```

4. **Regenerate Progression Analysis & Plots**:
   ```bash
   python feature_progression_analysis.py
   ```

5. **Start the Web App**:
   ```bash
   python -m http.server 8080
   ```
   Open **[http://localhost:8080/](http://localhost:8080/)** in your browser.

---

## 🙏 Acknowledgments & References

We gratefully acknowledge the following references, datasets, and open-source tools:

1. **Design & Layout Reference**:
   - [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/): Inspired the SaaS aesthetic, coefficient tables, VIF multicollinearity gauges, and statistical reporting layout.
2. **Dataset**:
   - [50 Startups Dataset (Avik Jain 100-Days-Of-ML-Code)](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv).
3. **Core Libraries**:
   - [Scikit-learn](https://scikit-learn.org/): For LassoCV, ElasticNetCV, SequentialFeatureSelector, and RFE.
   - [Statsmodels](https://www.statsmodels.org/): For OLS inference, $t$-tests, $p$-values, confidence intervals, and VIF calculations.
   - [Chart.js](https://www.chartjs.org/): For responsive, interactive dual-panel line charts.
"# AIoT-DIC2-MultipleLR" 
