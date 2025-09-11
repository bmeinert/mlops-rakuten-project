from flask import Flask, request, jsonify
import requests
import os
import base64

app = Flask(__name__)

AIRFLOW_API_URL = os.getenv('_AIRFLOW_API_URL', 'http://airflow-webserver:8080/api/v1/dags/ml_pipeline_mixed_experiment_dvc/dagRuns')
AIRFLOW_USERNAME = os.getenv('_AIRFLOW_WWW_USER_USERNAME', 'default_user') 
AIRFLOW_PASSWORD = os.getenv('_AIRFLOW_WWW_USER_PASSWORD', 'default_pass') 

@app.route('/alertmanager-webhook', methods=['POST'])
def handle_webhook():
    try:
        alert_data = request.json
        print(f"Received alert from Alertmanager: {alert_data}")

        payload_for_airflow = {"conf": {}} 

        auth_string = f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}"
        encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {encoded_auth_string}'
        }

        print(f"Sending to Airflow API: {AIRFLOW_API_URL} with payload: {payload_for_airflow}")

        response = requests.post(AIRFLOW_API_URL, json=payload_for_airflow, headers=headers)

        print(f"Airflow API response status: {response.status_code}")
        print(f"Airflow API response body: {response.text}")

        return jsonify(response.json()), response.status_code

    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)