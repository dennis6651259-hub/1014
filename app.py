import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load precomputed model registry
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DATA_PATH = os.path.join(BASE_DIR, "model_data.json")

try:
    with open(MODEL_DATA_PATH, "r", encoding="utf-8") as f:
        MODEL_DATA = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model_data.json: {e}")
    MODEL_DATA = {}


@app.route("/")
def index():
    """Render the main StatsUnlock-inspired regression explorer dashboard."""
    return render_template("index.html", initial_algo="Lasso", initial_k=2)


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return the complete 5-algorithm feature selection evaluation data."""
    algo = request.args.get("algo")
    if algo and algo in MODEL_DATA:
        return jsonify({algo: MODEL_DATA[algo]})
    return jsonify(MODEL_DATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live prediction endpoint: Calculates startup profit based on user inputs
    and the selected algorithm/k-feature subset.
    """
    try:
        data = request.get_json(force=True) or {}
        algo = data.get("algo", "Lasso")
        k = int(data.get("k", 2))
        rd_spend = float(data.get("rd_spend", 100000))
        marketing_spend = float(data.get("marketing_spend", 150000))
        admin_spend = float(data.get("admin_spend", 120000))
        state = data.get("state", "California")

        if algo not in MODEL_DATA or k < 1 or k > len(MODEL_DATA[algo]):
            return jsonify({"error": "Invalid algorithm or k parameter"}), 400

        model_spec = MODEL_DATA[algo][k - 1]
        coefficients = model_spec["coefficients"]
        active_features = model_spec["features"]

        predicted_profit = 0.0
        breakdown = []

        for c in coefficients:
            feat_name = c["feature"]
            coef_val = c["coef"]

            if "Intercept" in feat_name:
                predicted_profit += coef_val
                breakdown.append({
                    "feature": "Intercept",
                    "value": 1.0,
                    "coef": coef_val,
                    "contribution": coef_val
                })
            elif feat_name == "R&D Spend":
                contrib = coef_val * rd_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "R&D Spend",
                    "value": rd_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Marketing Spend":
                contrib = coef_val * marketing_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "Marketing Spend",
                    "value": marketing_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Administration":
                contrib = coef_val * admin_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "Administration",
                    "value": admin_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "State_Florida":
                val = 1.0 if state == "Florida" else 0.0
                contrib = coef_val * val
                predicted_profit += contrib
                breakdown.append({
                    "feature": "State_Florida",
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "State_New York":
                val = 1.0 if state == "New York" else 0.0
                contrib = coef_val * val
                predicted_profit += contrib
                breakdown.append({
                    "feature": "State_New York",
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })

        return jsonify({
            "status": "success",
            "algorithm": algo,
            "k": k,
            "active_features": active_features,
            "predicted_profit": predicted_profit,
            "rmse": model_spec["rmse"],
            "r2": model_spec["r2"],
            "breakdown": breakdown
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/summary", methods=["GET"])
def summary():
    """Return key comparison metrics across all 5 algorithms."""
    summary_data = []
    for algo, subsets in MODEL_DATA.items():
        for item in subsets:
            summary_data.append({
                "algorithm": algo,
                "k": item["k"],
                "features": item["features"],
                "rmse": item["rmse"],
                "r2": item["r2"],
                "mae": item["mae"]
            })
    return jsonify(summary_data)


# Local development server entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" * Starting 50 Startups Flask App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
