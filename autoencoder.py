# Import necessary libraries
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# Autoencoder Model Definition
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
# Main Execution
# -----------------------------
def main():
    # Load and preprocess the data
    file_path = 'E:\\CISO AI\\Syslogd\\Logs\\SyslogCatchAll-2023-12-17.txt'
    processed_data = load_and_preprocess_logs(file_path)

    # Convert to PyTorch tensors and split the dataset
    tensor_data = torch.Tensor(processed_data)
    train_data, val_data = train_test_split(tensor_data, test_size=0.2)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=64, shuffle=False)

    # Initialize the model
    input_size = processed_data.shape[1]
    model = Autoencoder(input_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for inputs, in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Optional: Validation step
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs), inputs).item() for inputs, in val_loader) / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'autoencoder_model.pth')



if __name__ == "__main__":
    main()