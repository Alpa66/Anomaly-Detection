import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_score, recall_score


class SWat_dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, target: pd.DataFrame,  window_size, device):
        self.data = dataframe
        self.window_size = window_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx: idx + self.window_size]
        features = torch.tensor(window.iloc[:,:].values).float().to(self.device)
        return features

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_val_dataloaders(batch_size, window_size, device):
    df = pd.read_csv('../../../Projects/data/SWaT_Dataset_Normal_v1.csv')
    df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
    df = df.astype('float64')
    columns = df.columns
    mm = StandardScaler()
    Normalized = pd.DataFrame(mm.fit_transform(df))
    train_set = Normalized[: int(0.8 * Normalized.shape[0])]
    validation_set = Normalized[int(0.8 * Normalized.shape[0]):]
    train_dataset = SWat_dataset(train_set, train_set, window_size, device)
    validation_dataset = SWat_dataset(validation_set, validation_set, window_size, device)

    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, validation_loader, columns, mm

def train_USAD(AE1, AE2, optimizer1, optimizer2, train_loader, validation_loader, epochs):
    AE1_val_history = []
    AE2_val_history = []
    for i in range(epochs):
        running_loss_AE1 = []
        running_loss_AE2 = []
        val_loss_AE1 = []
        val_loss_AE2 = []
        for _ , features in enumerate(train_loader):
            features = features.view(features.shape[0], -1)
            w1 = AE1(features)
            w2 = AE2(features)
            w3 = AE2(w1)
            lossAE1 = (1 / (i + 1)) * torch.mean((features - w1) ** 2) + (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
            lossAE2 = (1 / (i + 1)) * torch.mean((features - w2) ** 2) - (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
            
            running_loss_AE1.append(lossAE1)
            lossAE1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            w1 = AE1(features)
            w2 = AE2(features)
            w3 = AE2(w1)
            lossAE1 = (1 / (i + 1)) * torch.mean((features - w1) ** 2) + (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
            lossAE2 = (1 / (i + 1)) * torch.mean((features - w2) ** 2) - (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
            
            running_loss_AE2.append(lossAE2)
            lossAE2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
        
        for _ , features in enumerate(validation_loader):
            with torch.no_grad():
                features = features.view(features.shape[0], -1)
            
                w1 = AE1(features)
                w2 = AE2(features)
                w3 = AE2(w1)
                lossAE1 = (1 / (i + 1)) * torch.mean((features - w1) ** 2) + (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
                lossAE2 = (1 / (i + 1)) * torch.mean((features - w2) ** 2) - (1 - (1 / (i + 1))) * torch.mean((features - w3) ** 2)
                val_loss_AE1.append(lossAE1)
                val_loss_AE2.append(lossAE2)
        AE1_val_history.append(torch.stack(val_loss_AE1).mean().item())
        AE2_val_history.append(torch.stack(val_loss_AE2).mean().item())
        print(f'Epoch: {i} ---> Val loss: AE1 {AE1_val_history[-1]:.4f}, AE2: {AE2_val_history[-1]:.4f}')
        print(f'Train loss: AE1 {torch.stack(running_loss_AE1).mean().item():.4f}, AE2 {torch.stack(running_loss_AE2).mean().item():.4f}')
    return AE1, AE2, AE1_val_history, AE2_val_history

def get_test_loader(columns, batch_size, window_size, device, mm):
    df2 = pd.read_csv('../../../Projects/data/SWaT_Dataset_Attack_v0.csv')
    labels = df2['Normal/Attack']
    df2 = df2.drop(columns=[' Timestamp', 'Normal/Attack'])
    df2 = df2.astype('float64')
    df2.columns = columns
    test_normalized = pd.DataFrame(mm.transform(df2))
    test_dataset = SWat_dataset(test_normalized, test_normalized, window_size, device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels = labels.values
    labels = [0 if (lab == 'Normal') else 1 for lab in labels]
    windows_labels=[]
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int32(labels[i:i+window_size])))
    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    return test_loader, y_test

def testing(AE1, AE2, test_loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.view(batch.shape[0], -1)
            w1=AE1(batch)
            w2=AE2(w1)
            results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results

def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.fc3 = nn.Linear(input_size // 4, hidden_size)
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoder):        
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.input_size = input_size
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(hidden_size, input_size // 4)
        self.fc2 = nn.Linear(input_size // 4, input_size // 2)
        self.fc3 = nn.Linear(input_size // 2, input_size)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out
