from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import base64

# Initialiser l'objet InferenceHTTPClient pour les chiens
CLIENT_DOG = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="QHL5psNHyJiFsvWBeMeo"
)

# Initialiser l'objet InferenceHTTPClient pour les chats
CLIENT_CAT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="QHL5psNHyJiFsvWBeMeo"
)

app = Flask(__name__)

# Définir l'endpoint pour effectuer l'inférence pour les chiens
@app.route('/infer-dog', methods=['POST'])
def infer_dog_breed():
    try:
        # Récupérer les données d'image encodées en base64 depuis la requête
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'Image data not provided'}), 400

        # Appeler la méthode infer de l'objet InferenceHTTPClient pour les chiens
        result_dog = CLIENT_DOG.infer(image_data, model_id="dog-breed-identifier-nc37x/1")

        # Renvoyer les résultats d'inférence pour les chiens au client
        return jsonify(result_dog)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Définir l'endpoint pour effectuer l'inférence pour les chats
@app.route('/infer-cat', methods=['POST'])
def infer_cat_breed():
    try:
        # Récupérer les données d'image encodées en base64 depuis la requête
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'Image data not provided'}), 400

        # Appeler la méthode infer de l'objet InferenceHTTPClient pour les chats
        result_cat = CLIENT_CAT.infer(image_data, model_id="cat-breeds-2n7zk/1")

        # Renvoyer les résultats d'inférence pour les chats au client
        return jsonify(result_cat)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
