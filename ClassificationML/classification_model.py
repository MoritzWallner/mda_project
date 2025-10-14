import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.data, self.labels = self.init_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def init_data(self):
        data = []
        labels = []
        columns = None
        # get all files in the data folder, even in subfolders
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.pkl'):
                    vehicle_id, track_id, modality, modality_precission, file_data = pd.read_pickle(os.path.join(root, file))
                    file_data = file_data.reset_index(drop=True)
                    file_data.fillna(method='ffill', inplace=True)
                    file_data.fillna(method='bfill', inplace=True)
                    file_data['time_passed'] = file_data['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().cumsum()
                    file_data.drop(columns=['time'], inplace=True, errors='ignore')
                    if columns is None:
                        columns = file_data.columns
                        common_cols = columns
                    else:
                        if not len(columns) == len(file_data.columns):
                            common_cols = columns.intersection(file_data.columns)
                            columns = common_cols

                    file_data['label'] = modality
                    #file_data = np.array(file_data, dtype=np.float32)
                    data.append(file_data)
                    labels.append(file_data['label'])


        data_cat = np.concatenate([d[common_cols] for d in data], axis=0, dtype=np.float32)
        labels = np.concatenate(labels, axis=0)
        # convert labels to integers
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        # print the classes
        print("Classes:", le.classes_) 
        # normalize data
        scaler = preprocessing.StandardScaler()
        data_cat = scaler.fit_transform(data_cat)
        # convert to torch tensors
        data = torch.tensor(data_cat, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels
    

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def train_model(model, dataset_train, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)
    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        losses = []
        acc = 0
        t0 = time.time()
        i = 0
        for inputs, labels in dataloader_train:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            acc += (outputs.argmax(dim=1) == labels.to(device)).float().mean()
            i += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader_test:
                outputs = model(inputs.to(device))
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels.to(device)).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {np.mean(losses):.4f}, Training Accuracy: {acc/i:.4f}')
        print(f'Accuracy of the model on the validation data: {accuracy:.2f}%')
        print(f'Time taken for epoch {epoch+1}: {time.time() - t0:.2f} seconds')


if __name__ == "__main__":
    dataset = Dataset(data_folder='data/raw')
    input_size = dataset.data.shape[1]
    hidden_size = 64
    output_size = len(torch.unique(dataset.labels))

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    model = Model(input_size, hidden_size, output_size)
    model.to(device)
    train_model(model, dataset, epochs=20, batch_size=64, learning_rate=0.001, device=device)