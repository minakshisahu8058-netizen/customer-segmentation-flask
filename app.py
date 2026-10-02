"""Step 2: Flask web app. User enters R, F, M -> app predicts the customer segment."""
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
bundle = joblib.load("segment_model.pkl")

ADVICE = {
    "Champions": "Best customers. Reward them with loyalty perks and early access.",
    "Loyal Customers": "Buy regularly. Upsell and ask for reviews or referrals.",
    "At-Risk Customers": "Used to buy but slowed down. Send a win-back offer now.",
    "Lost Customers": "Inactive for long. Try a low-cost reactivation email.",
}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            r = float(request.form["recency"])
            f = float(request.form["frequency"])
            m = float(request.form["monetary"])
            x = bundle["scaler"].transform(np.log1p([[r, f, m]]))
            cluster = int(bundle["model"].predict(x)[0])
            seg = bundle["labels"][cluster]
            result = {"segment": seg, "advice": ADVICE[seg]}
        except (ValueError, KeyError):
            result = {"segment": "Invalid input", "advice": "Please enter valid positive numbers."}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
