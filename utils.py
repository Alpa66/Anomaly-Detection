from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch


def get_train_loaders(File_Path, device):
    df = pd.read_csv('../../Projects/data/SWaT_Dataset_Normal_v1.csv')
    df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
    df = df.astype('float64')
    mm = StandardScaler()
    Normalized = pd.DataFrame(mm.fit_transform(df))
    train_set = Normalized[: int(0.8 * Normalized.shape[0])]
    validation_set = Normalized[int(0.8 * Normalized.shape[0]):]


def test(model, test_loader):
    scores=[]
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            w1=model(batch)
            scores.append(torch.mean(torch.mean((batch-w1)**2, axis=1), axis=1))
    labels = labels.values
    labels = [0 if (lab == 'Normal') else 1 for lab in labels]
    windows_labels=[]
    for i in range(len(labels)-window_size):
        windows_labels.append(list(np.int32(labels[i:i+window_size])))

def testing(loader, alpha=.5, beta=.5):
    results=[]
    with torch.no_grad():
        for batch in loader:
            w1=AE1(batch)
            results.append(torch.mean((batch-w1)**2, axis=1))
    return results