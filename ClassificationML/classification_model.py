import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import geopandas as gpd

def lat_lon_2_m(latitude_1, longitude_1, latitude_2, longitude_2):
    # Radius of the earth in m
    radius_earth = 6371009

    d_latitude = np.deg2rad(latitude_2 - latitude_1)
    d_longitude = np.deg2rad(longitude_2 - longitude_1)
    latitude_1 = np.deg2rad(latitude_1)
    latitude_2 = np.deg2rad(latitude_2)

    a = (np.sin(d_latitude / 2)) ** 2 + np.cos(latitude_1) * np.cos(latitude_2) * (np.sin(d_longitude / 2)) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = radius_earth * c

    return distance

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
        # get all files in the data folder, even in subfolders
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if file.endswith('.pkl'):
                    _, _, modality, _, file_data = pd.read_pickle(os.path.join(root, file))
                    file_data = file_data.reset_index(drop=True)
                    file_data.ffill(inplace=True)
                    file_data.bfill(inplace=True)
                    file_data['time_passed'] = pd.to_datetime(file_data['time']).diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().cumsum()
                    file_data['distance_covered'] = lat_lon_2_m(file_data['latitude'].to_numpy(), file_data['longitude'].to_numpy(), file_data['latitude'].shift(1).to_numpy(), file_data['longitude'].shift(1).to_numpy())
                    file_data.fillna(0, inplace=True)
                    file_data['distance_covered'].cumsum()
                    file_data.drop(columns=['time'], inplace=True, errors='ignore')
                    file_data['label'] = modality if modality is not None else 'car'
                    data.append(file_data)
                    labels.append(file_data['label'])

        data_cat = np.concatenate([d[['latitude', 'longitude', 'speed', 'time_passed', 'distance_covered']] for d in data], axis=0, dtype=np.float32) 
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
    
def train_model(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.labels, test_size=0.1, random_state=42)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train.numpy(), y_train.numpy())

    ## Test the model
    predictions = clf.predict(X_test.numpy())
    total_Labels = y_test.numpy()
    correct = np.sum(total_Labels == predictions)
    total = len(total_Labels)
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

    # Plot confusion matrix
    cm = confusion_matrix(total_Labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('outputs/confusion_matrix_test.png')

    ## Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/classification_model.joblib')


def test_model(data_path, model_path):
    dataset = Dataset(data_folder=data_path)
    clf = joblib.load(model_path)

    predictions = clf.predict(dataset.data.numpy())
    all_labels = dataset.labels.numpy()

    correct = np.sum(all_labels == predictions)
    total = len(all_labels)
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the data from {data_path}: {accuracy:.2f}%')

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    output_path = f'outputs/confusion_matrix_{data_path.split("/")[-1]}.png'
    plt.savefig(output_path)
    return output_path

if __name__ == "__main__":
    #train_dataset = Dataset(data_folder='data/raw/User1_Smartphone')

    #train_model(train_dataset)

    test_model(data_path='data/raw/User1_Smartphone', model_path='models/classification_model.joblib')
