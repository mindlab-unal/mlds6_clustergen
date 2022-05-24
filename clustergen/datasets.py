from numpy import ndarray as Array
from clustergen.types import DatasetEnum
from typing import Callable
from sklearn.datasets import make_blobs, make_moons, make_circles

datasets_table = {
        DatasetEnum.BLOBS: lambda noise, n_samples: make_blobs(
            n_samples=n_samples, centers=2, cluster_std=noise
            ),
        DatasetEnum.CIRCLES: lambda noise, n_samples: make_circles(
            noise=noise, n_samples=n_samples
            ),
        DatasetEnum.MOONS: lambda noise, n_samples: make_moons(
            noise=noise, n_samples=n_samples
            )
        }

class Dataset:

    def __init__(
            self, func: Callable, noise: float,
            n_samples: int, name: DatasetEnum
            ):
        self.name = name
        self.func = func
        self.noise = noise
        self.n_samples = n_samples

    def extract(self) -> Array:
        X, _ = self.func(self.noise, self.n_samples)
        return X

    def __repr__(self) -> str:
        return f"Dataset(dataset={self.name})"

    def __str__(self) -> str:
        return self.__repr__()

class DatasetProxy:

    def __init__(
            self,
            dataset: DatasetEnum,
            noise: float,
            n_samples: int
            ):
        self.dataset = dataset
        self.noise = noise
        self.n_samples = n_samples

    def resolve(self) -> Dataset:
        ds_func = datasets_table[self.dataset]
        dataset = Dataset(ds_func, self.noise, self.n_samples, name=self.dataset)
        return dataset
