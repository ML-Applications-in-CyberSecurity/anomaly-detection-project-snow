import socket
import json
import pandas as pd
import joblib
import requests
import csv
import os
import random

HOST = 'localhost'
PORT = 9999

LLM_PROMPT_TEMPLATES = [
    {
        "system": "You are a network security expert. Provide a concise label and the most probable cause for network anomalies based on sensor data.",
        "user": "Sensor data: {data}\nWhat kind of anomaly is this? What is its most likely cause? Be brief and to the point."
    },
    {
        "system": "You are an AI assistant specialized in IoT sensor anomaly detection. Analyze the provided sensor reading.",
        "user": "Sensor reading: {data}\nIdentify the anomaly type and elaborate on its potential root causes."
    },
    {
        "system": "You are a helpful assistant that labels sensor anomalies.",
        "user": "Sensor reading: {data}\nDescribe the type of anomaly and suggest a possible cause."
    }
]

model = joblib.load("anomaly_model.joblib")

CSV_FILE = "anomalies_log.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["src_port", "dst_port", "packet_size", "duration_ms", "protocol", "confidence_score", "llm_explanation"])


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

    selected_template = random.choice(LLM_PROMPT_TEMPLATES)
    system_message = selected_template["system"]
    user_message = selected_template["user"].format(data=data)

    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
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
                score = model.decision_function(processed)[0]  # confidence score

                if prediction == -1:
                    label = describe_anomaly_with_llm(data)
                    print(f"\n🚨 Anomaly Detected!")
                    print(f"Confidence Score: {score:.4f}")
                    print(f"Label & Reason: {label}\n")

                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            data["src_port"],
                            data["dst_port"],
                            data["packet_size"],
                            data["duration_ms"],
                            data["protocol"],
                            score,
                            label.strip().replace('\n', ' '),
                        ])
                else:
                    print(f"✅ Normal data. Confidence Score: {score:.4f}\n")

            except json.JSONDecodeError:
                print("Error decoding JSON.")