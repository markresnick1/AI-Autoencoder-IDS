import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

def train_model(data_loader, model, criterion, optimizer, num_epochs=400): #set number of epochs
    for epoch in range(num_epochs):
        model.train()
        for inputs, in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def main():
    file_path = 'E:\\CISO AI\\Syslogd\\Logs\\SyslogCatchAll-2023-12-24.txt'
    processed_data = load_and_preprocess_logs(file_path)

    tensor_data = torch.Tensor(processed_data)
    train_loader = DataLoader(TensorDataset(tensor_data), batch_size=64, shuffle=True)

    input_size = processed_data.shape[1]
    model = Autoencoder(input_size)

    model.load_state_dict(torch.load('autoencoder_model_updated.pth'))
    model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer)

    torch.save(model.state_dict(), 'autoencoder_model_updated.pth')

if __name__ == "__main__":
    main()
