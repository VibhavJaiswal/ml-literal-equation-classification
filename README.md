# ml-literal-equation-classification

Literal Equation Classifier

A machine learning-based classifier to identify literal equations using TF-IDF vectorization and a Random Forest model. The project includes training, testing, and deploying the model via a Flask API.
📌 Features

✅ Preprocesses equations using TF-IDF
✅ Trains multiple ML models (Random Forest, Logistic Regression, LightGBM, XGBoost, etc.)
✅ Evaluates models and selects the best one based on F1-score
✅ Deploys the model using Flask for API-based predictions

📌 Training the Model

If you want to train the model from scratch:

    Open literal_equation_classifier.ipynb in Jupyter Notebook.
    Run all the cells to:
        Load and preprocess data.
        Train multiple machine learning models.
        Save the best-performing model (random_forest_best_model.pkl).
        Save the TF-IDF vectorizer (tfidf_vectorizer.pkl).

📌 Running the Flask API

    Ensure the saved model and vectorizer are in the project folder.
    Run the Flask server:

python app.py

📌 Testing the API

You can test the API using Postman or cURL.
✅ Using Postman

    Open Postman.
    Set Method to POST.
    Enter URL
    Go to the Body tab → Select raw → Choose JSON.
    Send the following JSON request:

{
    "features": ["F = ma", "2 + 3 = 5", "P = IV", "E = mc^2"]
}

Click Send and check the response.

📌 Model Details

    Vectorization: TF-IDF (max_features=50000)
    Best Model: Random Forest (n_estimators=30, random_state=42)
    Dataset: Preprocessed CSV of equations

📌 Future Enhancements

🚀 Optimize Hyperparameters using Grid Search
🚀 Expand to More Complex Equations
🚀 Deploy API to Cloud (AWS/GCP/Heroku)