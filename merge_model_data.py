import json

with open("model_data.json", "r") as f:
    startup_data = json.load(f)

with open("boston_model_data.json", "r") as f:
    boston_data = json.load(f)

combined_content = f"""// Unified Model Data for 50 Startups & Boston Housing
window.STARTUP_MODEL_DATA = {json.dumps(startup_data, indent=2)};
window.BOSTON_MODEL_DATA = {json.dumps(boston_data, indent=2)};
window.MODEL_DATA = window.STARTUP_MODEL_DATA;
"""

with open("model_data.js", "w", encoding="utf-8") as f:
    f.write(combined_content)

print("Updated model_data.js successfully! Length:", len(combined_content))
