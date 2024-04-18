from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_train_loaders(File_Path, device):
    df = pd.read_csv('../../Projects/data/SWaT_Dataset_Normal_v1.csv')
    df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
    df = df.astype('float64')
    mm = StandardScaler()
    Normalized = pd.DataFrame(mm.fit_transform(df))
    train_set = Normalized[: int(0.8 * Normalized.shape[0])]
    validation_set = Normalized[int(0.8 * Normalized.shape[0]):]