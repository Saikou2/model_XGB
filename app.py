from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le scaler
scaler = StandardScaler()
SCALER_PATH = 'scaler.pkl'
MODEL_PATH = 'xgb_model.pkl'

# Fonction pour charger le modèle et le scaler
def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

# Initialiser l'application Flask
app = Flask(__name__)
CORS(app)  # Activation de CORS

# Charger le modèle et le scaler
model = load_model()
scaler = load_scaler()

# Route de la racine
@app.route('/')
def home():
    return "API de prédiction active !"

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame avec les colonnes nécessaires, même si certaines sont manquantes
        df = pd.DataFrame(data, index=[0])

        # Liste des 77 colonnes attendues
        expected_columns = ['feature1', 'feature2', 'feature3', ..., 'feature77']
        
        # Si des colonnes sont manquantes, les ajouter avec des valeurs par défaut (par exemple 0)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Remplace par une valeur par défaut si nécessaire

        # Appliquer la même transformation que lors de l'entraînement
        df[df.columns] = scaler.transform(df[df.columns])

        # Prédiction
        prediction = model.predict(df)

        # Retourner la prédiction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    print("Démarrage de l'application Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
