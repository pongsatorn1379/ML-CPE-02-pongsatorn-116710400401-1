import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import to_features


def split_dataset(y, seed=42, test_size=0.2, val_size=0.2):
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    train, val = train_test_split(train, test_size=val_size / (1 - test_size),
                                 random_state=seed, stratify=y[train])
    return train, val, test


def prepare_data(data_path, img_size=224, seed=42):
    X, y, classes, _, _, _ = load_data(data_path, img_size)
    X = to_features(X)
    train, val, test = split_dataset(y, seed)
    print(f"Train: {len(train)}, validation: {len(val)}, test: {len(test)}")
    return X[train], X[val], X[test], y[train], y[val], y[test], classes
