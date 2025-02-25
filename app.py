from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_swagger_ui import get_swaggerui_blueprint
import os

# Charger le scaler et le modèle
SCALER_PATH = 'scaler.pkl'
MODEL_PATH = 'xgb_model.pkl'

# Fonction pour charger le modèle et le scaler
def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

# Initialiser l'application Flask
app = Flask(__name__, static_folder='static')
CORS(app)  # Activation de CORS

# Charger le modèle et le scaler
model = load_model()
scaler = load_scaler()

# Définir la documentation Swagger
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'  # Localisation du fichier JSON Swagger
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "API de prédiction"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Route de la racine
@app.route('/')
def home():
    return "API de prédiction active !"

# Route pour servir le fichier swagger.json
@app.route('/static/swagger.json')
def swagger_json():
    swagger_path = os.path.join(app.static_folder, 'swagger.json')
    if os.path.exists(swagger_path):
        return send_from_directory(app.static_folder, 'swagger.json')
    else:
        return jsonify({'error': 'swagger.json not found'}), 404

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données JSON de la requête
        data = request.get_json()

        # Créer un DataFrame avec les colonnes nécessaires, même si certaines sont manquantes
        df = pd.DataFrame(data, index=[0])

        # Liste des colonnes attendues
        expected_columns = ['feature1', 'feature2', 'feature3', ..., 'feature77']

        # Si des colonnes sont manquantes, les ajouter avec des valeurs par défaut (par exemple 0)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Remplace par une valeur par défaut si nécessaire

        # Appliquer la même transformation que lors de l'entraînement
        df[df.columns] = scaler.transform(df[df.columns])

        # Prédiction
        prediction = model.predict(df)

        # Convertir la prédiction en "clean" ou "menace"
        if prediction[0] == 0:
            prediction_label = "menace"
        else:
            prediction_label = "clean"

        # Retourner la prédiction sous forme de texte
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Démarrage de l'application Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)
