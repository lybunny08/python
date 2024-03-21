from inference_sdk import InferenceHTTPClient
import base64

# Initialisation du client d'inférence avec l'URL de l'API et la clé API
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="QHL5psNHyJiFsvWBeMeo"
)

# Chemin de l'image à utiliser pour l'inférence
image_path = "E:/identify/assets/images/dog.jpg"

# ID du modèle à utiliser pour l'inférence
model_id = "dog-breed-identifier-nc37x/1"

# Effectuer l'inférence en utilisant le client
try:
    # Charger l'image depuis le chemin spécifié
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')  # Convertir en base64 et décoder en UTF-8

    # Appeler la méthode infer du client avec les données de l'image encodées en base64 et l'ID du modèle
    result = CLIENT.infer(image_data, model_id=model_id)

    # Afficher les résultats de l'inférence
    print(result)
except Exception as e:
    print(f"Error during inference: {str(e)}")
