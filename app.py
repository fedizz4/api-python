from flask import Flask, request, jsonify
import joblib
import pandas as pd
from scipy.sparse import hstack

app = Flask(__name__)

# Chargement des objets sauvegardés
model = joblib.load("modele_va_quitter.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
encoders = joblib.load("label_encoders.pkl")

@app.route("/")
def home():
    return "API de prédiction - Employés"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # dictionnaire

    df = pd.DataFrame([data])

    # Encodage
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Texte
    texte_vect = vectorizer.transform(df["Commentaire du sondage"])
    df = df.drop(columns=["Commentaire du sondage"])

    # Combinaison
    final_input = hstack([df.values, texte_vect])

    prediction = model.predict(final_input)[0]
    return jsonify({"va_quitter": int(prediction)})

if __name__ == "__main__":
    app.run()
