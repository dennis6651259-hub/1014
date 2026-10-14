# 50 Startups: Multiple Linear Regression & Feature Selection Explorer

An interactive machine learning, statistical inference, and full-stack web platform built with the **Flask framework** and following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

This project evaluates **Multiple Linear Regression (OLS)** against **5 distinct feature selection algorithms** to uncover the optimal feature subset for predicting startup profitability while guarding against multicollinearity and overfitting.

---

## 🌐 Live Demo & Interactive Web Application

🔗 **Live Demo**: **[https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)**

![50 Startups Interactive Regression & Feature Selection Explorer](dashboard_demo.png)

The web application is styled with the professional SaaS statistical calculator layout inspired by [StatsUnlock](https://statsunlock.com/multiple-linear-regression-calculator/).

### How to Run Locally with Flask:
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to: **[http://localhost:5000/](http://localhost:5000/)**

### Flask REST API Endpoints:
- `GET /`: Renders the main interactive regression explorer dashboard.
- `GET /api/models`: Returns the complete 5-algorithm feature selection evaluation data ($k=1 \dots 5$).
- `POST /api/predict`: Live inference endpoint calculating predicted profit and formula contribution breakdown:
  ```json
  {
    "algo": "Lasso",
    "k": 2,
    "rd_spend": 100000,
    "marketing_spend": 150000,
    "admin_spend": 120000,
    "state": "California"
  }
  ```
- `GET /api/summary`: Returns summary metrics across all algorithms.

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

---

## 📁 Repository Structure

```text
├── dashboard_demo.png                  # Dashboard screenshot in README
├── app.py                              # Flask Web Application (WSGI entrypoint & REST API)
├── api/
│   └── index.py                        # Vercel Serverless Function entry point
├── templates/
│   └── index.html                      # StatsUnlock-style Jinja2 web template
├── static/
│   └── model_data.js                   # Static data asset for instant frontend fallback
├── model_data.json                     # Comprehensive model statistics, weights, and tests
├── vercel.json                         # Vercel deployment configuration
├── requirements.txt                    # Project dependencies (Flask, NumPy, Pandas, etc.)
├── main.py                             # CRISP-DM CLI multiple linear regression script
├── feature_progression_analysis.py     # Script generating k=1..5 metrics and charts
├── all_5_algorithms_comparison_plot.png# Comparison chart of all 5 algorithms
└── README.md                           # Documentation and project manual
```

---

## ☁️ Deployment

### 1. GitHub Pages (Active)
🔗 **[https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)**
Automated deployment via `.github/workflows/pages.yml`.

### 2. Vercel
Configured for zero-config Vercel Python serverless deployment via `api/index.py` and `vercel.json`.

---

## 🙏 Acknowledgments & References

We gratefully acknowledge the following references, datasets, and open-source tools:

1. **Design & Layout Reference**:
   - [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/): Inspired the SaaS aesthetic, coefficient tables, VIF multicollinearity gauges, and statistical reporting layout.
2. **Dataset**:
   - [50 Startups Dataset (Avik Jain 100-Days-Of-ML-Code)](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv).
3. **Core Libraries**:
   - [Flask](https://flask.palletsprojects.com/): Python micro web framework.
   - [Scikit-learn](https://scikit-learn.org/): For LassoCV, ElasticNetCV, SequentialFeatureSelector, and RFE.
   - [Statsmodels](https://www.statsmodels.org/): For OLS inference, $t$-tests, $p$-values, confidence intervals, and VIF calculations.
   - [Chart.js](https://www.chartjs.org/): For responsive, interactive dual-panel line charts.
