import numpy as np
import pandas as pd
import torch.utils.data


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


class MovieLens1MDataset(MovieLens20MDataset):
    """
    MovieLens 1M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path, sep='::', engine='python', header=None)
