import socket
import json
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import matplotlib

matplotlib.use('TkAgg')  # یا Agg اگر فقط ذخیره می‌خوای

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

    all_data = []  # لیست همه ویژگی‌ها
    all_labels = []  # لیست برچسب‌ها (1=normal, -1=anomaly)

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

                all_data.append(processed.iloc[0].tolist())  # یک ردیف به صورت لیست
                all_labels.append(prediction)

                if prediction == -1:
                    label = describe_anomaly_with_llm(data)
                    print(f"\n🚨 Anomaly Detected!\nLabel & Reason: {label}\n")
                else:
                    print("✅ Normal data.\n")

            except json.JSONDecodeError:
                print("Error decoding JSON.")

            if len(all_data) % 50 == 0:
                try:
                    df_viz = pd.DataFrame(all_data, columns=processed.columns)
                    df_viz['label'] = all_labels

                    # اعمال PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(df_viz.drop(columns=['label']))

                    df_viz['PC1'] = pca_result[:, 0]
                    df_viz['PC2'] = pca_result[:, 1]
                    df_viz['label'] = df_viz['label'].map({1: "normal", -1: "anomaly"})

                    # ترسیم
                    plt.figure(figsize=(8, 5))
                    sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="label",
                                    palette={"normal": "green", "anomaly": "red"})
                    plt.title("Live PCA Visualization of Sensor Data")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"visualization_{len(all_data)}.png")
                    plt.close()

                except Exception as e:
                    print(f"📉 Visualization error: {e}")
