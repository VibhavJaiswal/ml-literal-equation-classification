from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load the trained Random Forest model
model = joblib.load("random_forest_best_model.pkl")

# ✅ Load the saved TF-IDF Vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Receive JSON data
        text_input = data["features"]  # Expecting a list of equations

        # ✅ Transform input using the saved vectorizer
        X_transformed = vectorizer.transform(text_input)

        # ✅ Make predictions
        prediction = model.predict(X_transformed).tolist()

        return jsonify({"predictions": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
