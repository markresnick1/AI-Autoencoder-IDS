import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Load and Preprocess Data
# -----------------------------
# -----------------------------
# Data Preprocessing Functions
# -----------------------------
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
    data['ip_analysis'] = data['message'].apply(extract_feature_ip_analysis)
    data['text_complexity'] = data['message'].apply(extract_feature_text_complexity)
    data['protocol_port_info'] = data['message'].apply(extract_feature_protocol_port)
    scaler = StandardScaler()
    feature_columns = ['ip_analysis', 'text_complexity', 'protocol_port_info']
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data[feature_columns].values

# -----------------------------
# Load Trained Model
# -----------------------------
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

# -----------------------------
# Anomaly Detection Function
# -----------------------------
def detect_anomalies(data, model, threshold):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_tensor = torch.Tensor(data_scaled)

    with torch.no_grad():
        reconstructed = model(data_tensor).numpy()
        loss = ((data_scaled - reconstructed) ** 2).mean(axis=1)

    # Convert boolean mask to indices
    anomaly_indices = loss > threshold
    anomalies = data[anomaly_indices]
    return anomalies, anomaly_indices


# -----------------------------
# Main Execution
# -----------------------------
def main():
    # Load and preprocess the test data
    test_file_path = 'E:\\CISO AI\\Syslogd\\Logs\\anomalies_0.txt'
    test_data = load_and_preprocess_logs(test_file_path)

    # Determine the input size for the autoencoder
    input_size = test_data.shape[1]  # Number of features in preprocessed data

    # Initialize and load the model
    model = Autoencoder(input_size)
    model.load_state_dict(torch.load('autoencoder_model_updated.pth'))
    model.eval()

    # Set a threshold for anomaly detection
    threshold = 0.1  # Adjust this based on your requirements

    # Detect anomalies
    anomalies, anomaly_indices = detect_anomalies(test_data, model, threshold)

    # Convert the anomalies to a DataFrame
    anomaly_data = pd.DataFrame(test_data[anomaly_indices],
                                columns=['ip_analysis', 'text_complexity', 'protocol_port_info'])

    # Save detected anomalies to a file
    output_file_path = 'E:\\CISO AI\\Anomaly test output\\anomalies.txt'
    anomaly_data.to_csv(output_file_path, index=False, sep='\t')


if __name__ == "__main__":
    main()