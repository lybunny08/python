from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/infer-dog-breed', methods=['POST'])
def infer_dog_breed():
    # Obtenir l'image depuis la requête Flutter
    image_data = request.data

    # Appeler l'API de Roboflow pour effectuer l'inférence
    roboflow_api_url = 'https://outline.roboflow.com/dog-breed-identifier-nc37x/1'
    api_key = 'QHL5psNHyJiFsvWBeMeo'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(roboflow_api_url, headers=headers, data=image_data)

    if response.status_code == 200:
        # Renvoyer les résultats d'inférence à Flutter
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Failed to perform inference'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
