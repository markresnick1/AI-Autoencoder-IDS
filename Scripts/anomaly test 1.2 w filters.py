import pandas as pd
import re
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

def extract_feature_ip_analysis(message):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, message)
    unique_ips = set(ips)
    return len(unique_ips)

def extract_feature_text_complexity(message):
    return len(message)

def extract_feature_protocol_port(message):
    return message.count('HTTP') + message.count('HTTPS')

def load_and_preprocess_logs(file_path):
    data = pd.read_csv(file_path, sep="\t", names=['date_and_time', 'log_type', 'ip_address', 'message'])
    original_data = data.copy()
    data['ip_analysis'] = data['message'].apply(extract_feature_ip_analysis)
    data['text_complexity'] = data['message'].apply(extract_feature_text_complexity)
    data['protocol_port_info'] = data['message'].apply(extract_feature_protocol_port)
    scaler = StandardScaler()
    feature_columns = ['ip_analysis', 'text_complexity', 'protocol_port_info']
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data[feature_columns].values, original_data

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def detect_anomalies(processed_data, original_data, model, threshold):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(processed_data)
    data_tensor = torch.Tensor(data_scaled)

    with torch.no_grad():
        reconstructed = model(data_tensor).numpy()
        loss = ((data_scaled - reconstructed) ** 2).mean(axis=1)

    anomaly_indices = loss > threshold
    anomalies = original_data[anomaly_indices]
    return anomalies

def main():
    test_file_path = 'E:\\CISO AI\\Syslogd\\Logs\\SyslogCatchAll-2024-01-04.txt'
    processed_data, original_data = load_and_preprocess_logs(test_file_path)

    input_size = processed_data.shape[1]
    model = Autoencoder(input_size)
    model.load_state_dict(torch.load('autoencoder_model_1.1.pth'))
    model.eval()

    threshold = 0.8

    anomalies = detect_anomalies(processed_data, original_data, model, threshold)

    filtered_anomalies = anomalies[~anomalies['message'].str.contains("/usr/sbin/newsyslog")]

    output_file_path = 'E:\\CISO AI\\Anomaly test output\\anomalies_SyslogCatchAll-2024-01-04.txt'
    filtered_anomalies[['date_and_time', 'ip_address', 'message']].to_csv(output_file_path, index=False, sep='\t')

if __name__ == "__main__":
    main()