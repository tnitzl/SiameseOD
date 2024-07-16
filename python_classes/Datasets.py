import numpy as np

import torch

from torchvision import transforms
from torch.utils.data import Dataset

from itertools import product
from typing import Tuple

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random

class Siamese_Dataset(Dataset):
    def __init__(self, anomaly: np.ndarray, normal:np.ndarray) -> None:
        super(Siamese_Dataset, self).__init__()

        ##ensemble
        ##weak classifier -> ca 10 eigenen durchläufe max 1000 - 10000

        data_normal = list(product(normal, repeat=2))
        data_anomaly = list(product(anomaly, repeat=2))
        data_mixed = list(product(normal, anomaly))

        data_similar = [(pair, 1) for pair in data_normal + data_anomaly]
        data_different = [(pair, 0) for pair in data_mixed]

        self.data = data_similar + data_different
        print(f'Der Siamese Datasatz wurde mit einer gesamtLänge von {len(self.data)} erstellt')

    def __len__(self) -> int:
        return len(self. data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (data1, data2), label = self.data[idx]
        data1 = torch.tensor(data1, dtype=torch.float32)
        data2 = torch.tensor(data2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return data1, data2, label
    
class Simple_Dataset(Dataset):

    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        super(Simple_Dataset, self).__init__()
        self.data = data
        self.label = label
    
    def __len__(self) -> int:
        assert(len(self.data), len(self.label))
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.tensor(self.data[idx], dtype = torch.float32)
        label = torch.tensor(self.label[idx], dtype= torch.float32)
        return data, label


class Random_Dataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, percent_labeld: float, len_dataset: int):

        #(0.25,0.25) (0.75, 0.75)

        # Noramlizie Data
        #Übergeben zwischen 0.25, 0.75
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x)
        x = minmax_scaler.transform(x)

        self.len_dataset = len_dataset

        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]

        anomaly = x[anomaly_idx]
        normal = x[normal_idx]
        num_samples_anomaly = int(percent_labeld * len(anomaly))
        num_samples_normal = int(percent_labeld * len(normal))
        if num_samples_anomaly < 5:
            num_samples_anomaly = 5

        random_indicies_anomaly = np.random.choice(anomaly.shape[0], num_samples_anomaly, replace=False)
        self.labeld_anoamly = anomaly[random_indicies_anomaly]
        random_indicies_normal = np.random.choice(normal.shape[0], num_samples_normal, replace=False)
        self.labeld_normal = normal[random_indicies_normal]

        all_indicies_anomaly = np.arange(anomaly.shape[0])
        unlabeld_indicies_anomaly = np.setdiff1d(all_indicies_anomaly, random_indicies_anomaly)
        self.unlabeld_anomaly = anomaly[unlabeld_indicies_anomaly]

        all_indicies_normal = np.arange(normal.shape[0])
        unlabeld_indicies_normal = np.setdiff1d(all_indicies_normal, random_indicies_normal)
        self.unlabeld_normal = normal[unlabeld_indicies_normal]

    def __len__(self) -> int:
        return self.len_dataset
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zahlen = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        zufallszahl = random.choice(zahlen)
        # choice anomaly anomaly
        if zufallszahl < 2:
            first_random_index = random.randint(0, len(self.labeld_anoamly) - 1)
            second_random_index = random.randint(0, len(self.labeld_anoamly) - 1)
            while second_random_index == first_random_index:
                second_random_index = random.randint(0, len(self.labeld_anoamly) - 1)
            out1 = torch.tensor(self.labeld_anoamly[first_random_index], dtype = torch.float32)
            out2 = torch.tensor(self.labeld_anoamly[second_random_index], dtype = torch.float32)
            lable = torch.tensor(1, dtype = torch.float32)
            return out1, out2, lable
        # labeld normal - normal
        elif zufallszahl < 4:
            first_random_index = random.randint(0, len(self.labeld_normal) - 1)
            second_random_index = random.randint(0, len(self.labeld_normal) - 1)
            while second_random_index == first_random_index:
                second_random_index = random.randint(0, len(self.labeld_normal) - 1)
            out1 = torch.tensor(self.labeld_normal[first_random_index], dtype = torch.float32)
            out2 = torch.tensor(self.labeld_normal[second_random_index], dtype = torch.float32)
            lable = torch.tensor(1, dtype = torch.float32)
            return out1, out2, lable
        #labeld normal - anomaly
        else:
            first_random_index = random.randint(0, len(self.labeld_anoamly) - 1)
            second_random_index = random.randint(0, len(self.labeld_normal) - 1)
            out1 = torch.tensor(self.labeld_anoamly[first_random_index], dtype = torch.float32)
            out2 = torch.tensor(self.labeld_normal[second_random_index], dtype = torch.float32)
            lable = torch.tensor(0, dtype = torch.float32)
            return out1, out2, lable
    
    def get_unlabeld_dataset(self) -> Simple_Dataset:
        data = np.append(self.unlabeld_normal, self.unlabeld_anomaly, axis = 0)
        label = [0] * len(self.unlabeld_normal) + [1] * len(self.unlabeld_anomaly)
        return Simple_Dataset(data = data, label = label)
    
    def get_labeld_dataset(self) -> Simple_Dataset:
        data = np.append(self.labeld_normal, self.labeld_anoamly, axis = 0)
        label = [0] * len(self.labeld_normal) + [1] * len(self.labeld_anoamly)
        return Simple_Dataset(data = data, label = label)
    
            


class Base_Dataset():
    def __init__(self, X: np.ndarray, y: np.ndarray, percent_labeld: float):

        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]

        anomaly = X[anomaly_idx]
        normal = X[normal_idx]

        num_samples_anomaly = int(percent_labeld * len(anomaly))
        num_samples_normal = int(percent_labeld * len(normal))
        if num_samples_normal >= 1000:
            num_samples_normal = 1000
        if num_samples_anomaly == 0:
            num_samples_anomaly = 5
        if num_samples_anomaly >= 1000:
            num_samples_anomaly = 1000

        random_indicies_anomaly = np.random.choice(anomaly.shape[0], num_samples_anomaly, replace=False)
        self.labeld_anoamly = anomaly[random_indicies_anomaly]
        random_indicies_normal = np.random.choice(normal.shape[0], num_samples_normal, replace=False)
        self.labeld_normal = normal[random_indicies_normal]

        all_indicies_anomaly = np.arange(anomaly.shape[0])
        unlabeld_indicies_anomaly = np.setdiff1d(all_indicies_anomaly, random_indicies_anomaly)
        self.unlabeld_anomaly = anomaly[unlabeld_indicies_anomaly]

        all_indicies_normal = np.arange(normal.shape[0])
        unlabeld_indicies_normal = np.setdiff1d(all_indicies_normal, random_indicies_normal)
        self.unlabeld_normal = normal[unlabeld_indicies_normal]

        print(f"Die gesamte Länge der Daten ist {len(X)}")
        print(f"Die Länge das Anomalydatensatzen ist {len(anomaly)} und der normalen daten ist: {len(normal)}")
        print(f"Es werden {percent_labeld * 100}% der Daten gelabeld")
        print(f'Es wurden zwei Datensätze erstellt, der erste mit der beiden Längen {len(self.labeld_anoamly)} und {len(self.labeld_normal)}')
        print(f'Die ungelabelden parts dazu sind {len(self.unlabeld_anomaly)} und {len(self.unlabeld_normal)}')

    def get_siamese_dataset(self) -> Siamese_Dataset:
        return Siamese_Dataset(anomaly = self.labeld_anoamly, normal = self.labeld_normal)
    
    def get_unlabeld_dataset(self) -> Simple_Dataset:
        data = np.append(self.unlabeld_normal, self.unlabeld_anomaly, axis = 0)
        label = [0] * len(self.unlabeld_normal) + [1] * len(self.unlabeld_anomaly)
        return Simple_Dataset(data = data, label = label)
    
    def get_labeld_dataset(self) -> Simple_Dataset:
        data = np.append(self.labeld_normal, self.labeld_anoamly, axis = 0)
        label = [0] * len(self.labeld_normal) + [1] * len(self.labeld_anoamly)
        return Simple_Dataset(data = data, label = label)
    
    