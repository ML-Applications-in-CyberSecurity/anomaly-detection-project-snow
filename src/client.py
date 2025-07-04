import socket
import json
import pandas as pd
import joblib
import requests

HOST = 'localhost'
PORT = 9999

model = joblib.load("anomaly_model.joblib")


def pre_process_data(data):
    df = pd.DataFrame([data])
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2, 'UNKNOWN': 3}
    df['protocol'] = df['protocol'].map(protocol_map)
    return df


def describe_anomaly_with_llm(data):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 164813310ae80d158b06b43fb27c8279131046a3f667d04364c7ce51076237a7",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",  # ✅ مدل قابل استفاده برای اکانت‌های رایگان
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that labels sensor anomalies."},
            {"role": "user",
             "content": f"Sensor reading: {data}\nDescribe the type of anomaly and suggest a possible cause."}
        ],
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": 200,
        "stream": False
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response_json = response.json()
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        elif "error" in response_json:
            print("🚫 API Error:", response_json["error"]["message"])
            return "API Error: " + response_json["error"]["message"]
        else:
            print("⚠️ Unexpected response format:", response_json)
            return "Unknown error"
    except ValueError:
        print("❌ Failed to parse response as JSON")
        print(response.text)
        return "Invalid response"


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                processed = pre_process_data(data)
                prediction = model.predict(processed)[0]

                if prediction == -1:
                    label = describe_anomaly_with_llm(data)
                    print(f"\n🚨 Anomaly Detected!\nLabel & Reason: {label}\n")
                else:
                    print("✅ Normal data.\n")

            except json.JSONDecodeError:
                print("Error decoding JSON.")
