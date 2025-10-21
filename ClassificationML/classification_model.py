import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, test_run=False):
        self.data_folder = data_folder
        self.data, self.labels = self.init_data(test_run)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def init_data(self, test_run):
        data = []
        labels = []
        
        if test_run:
            with open('outputs/common_columns.txt', 'r') as f:
                common_cols = [line.strip() for line in f.readlines()]
            columns = common_cols
        else:
            common_cols = []
            columns = None
        # get all files in the data folder, even in subfolders
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.pkl'):
                    vehicle_id, track_id, modality, modality_precission, file_data = pd.read_pickle(os.path.join(root, file))
                    file_data = file_data.reset_index(drop=True)
                    file_data.ffill(inplace=True)
                    file_data.bfill(inplace=True)
                    file_data['time_passed'] = file_data['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().cumsum()
                    file_data.drop(columns=['time'], inplace=True, errors='ignore')
                    if columns is None and not test_run:
                        columns = file_data.columns
                        common_cols = columns
                    elif not test_run:
                        if not len(columns) == len(file_data.columns):
                            common_cols = columns.intersection(file_data.columns)
                            columns = common_cols
                    file_data['label'] = modality if modality is not None else 'car'
                    data.append(file_data)
                    labels.append(file_data['label'])

        # save common columns to a text file
        if not test_run:
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/common_columns.txt', 'w') as f:
                for col in common_cols:
                    f.write(f"{col}\n")
        data_cat = np.concatenate([d[common_cols] for d in data], axis=0, dtype=np.float32) 
        labels = np.concatenate(labels, axis=0)
        # convert labels to integers
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        # print the classes
        print("Classes:", le.classes_) 
        # normalize data
        scaler = preprocessing.MinMaxScaler()
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

def train_model(dataset, hidden_dim=64, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    X_train_val, X_test, y_train_val, y_test = train_test_split(dataset.data, dataset.labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
    dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
    dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    input_size = dataset.data.shape[1]
    output_size = len(torch.unique(dataset.labels))

    model = Model(input_size, hidden_dim, output_size).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tr_epoch_loss, tr_epoch_acc = [], []
    val_epoch_acc = []

    patience_counter = 0
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        losses, acc = [], []
        t0 = time.time()
        for inputs, labels in dataloader_train:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            losses.append(loss.item())
            acc.append((outputs.argmax(dim=1) == labels.to(device)).float().mean().item())
            # class imbalance handling (optional)
            """class_counts = torch.bincount(labels)
            class_weights = 1. / class_counts.float()
            sample_weights = class_weights[labels]
            loss = (loss * sample_weights.to(device)).mean()"""

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tr_epoch_loss.append(np.mean(losses))
        tr_epoch_acc.append(np.mean(acc))

        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader_val:
                outputs = model(inputs.to(device))
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels.to(device)).sum().item()

        accuracy = 100 * correct / total
        val_epoch_acc.append(accuracy)

        if accuracy > best_val_acc:
            best_val_acc = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {np.mean(losses):.4f}, Training Accuracy: {np.mean(acc):.4f}')
        print(f'Accuracy on validation data: {accuracy:.2f}%')
        print(f'Time Epoch {epoch+1}: {time.time() - t0:.2f} seconds')

        if patience_counter > 5:
            print("Early stopping triggered")
            break
    ## Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            outputs = model(inputs.to(device))
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

    ## Save the model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), f'models/classification_model.pth')

    # Plot training loss and accuracy
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(tr_epoch_loss, label='Training Loss')
    ax[0].set_title('Training Loss')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    ax[1].plot(tr_epoch_acc, label='Training Accuracy', color='orange')
    ax[1].set_title('Training Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    ax[2].plot(val_epoch_acc, label='Validation Accuracy', color='green')
    ax[2].set_title('Validation Accuracy')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()

    fig.supxlabel('Epochs')
    ax[2].set_xticks(np.arange(0, len(tr_epoch_loss), 1))
    fig.tight_layout()
    plt.savefig('outputs/training_plots.png')


def test_model(data_path, model_path, device='cpu'):
    dataset = Dataset(data_folder=data_path, test_run=True)
    sd = torch.load(model_path)

    input_size = dataset.data.shape[1]
    assert sd['fc1.weight'].shape[1] == input_size, "Model input size does not match dataset feature size."
    hidden_size = 128
    assert sd['fc2.weight'].shape[1] == hidden_size, "Model hidden size does not match."

    model = Model(input_size, hidden_size, output_size=3)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels.to(device)).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'outputs/confusion_matrix_{data_path.split("/")[-1]}.png')

if __name__ == "__main__":
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    train_dataset = Dataset(data_folder='data/raw')

    train_model(train_dataset, hidden_dim=64, epochs=20, batch_size=64, learning_rate=0.001, device=device)

    test_model(data_path='data/raw/User1_Smartphone', model_path='models/classification_model.pth', device=device)
    test_model(data_path='data/raw/User2_Logger', model_path='models/classification_model.pth', device=device)
