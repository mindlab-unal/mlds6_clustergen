from numpy import ndarray as Array
from enum import Enum, auto
from dataclasses import dataclass
from typing import Protocol

class StrategyEnum(Enum):
    """
    Enum to define possible clustering algorithms.
    """
    KMEANS = auto()
    GMM = auto()
    SPECTRAL = auto()

class DatasetEnum(Enum):
    """
    Enum to define possible datasets.
    """
    BLOBS = auto()
    MOONS = auto()
    CIRCLES = auto()

@dataclass
class CliParams:
    strategy: StrategyEnum
    n_clusters: int
    dataset: DatasetEnum

class ClusteringModel(Protocol):
    labels_: Array

    def fit(self, X: Array, y: Array):
        ...

"""
class CliParams:

    def __init__(self, strategy: StrategyEnum, n_clusters: int, dataset: DatasetEnum):
        self.strategy = strategy
        self.n_clusters = n_clusters
        self.dataset = dataset

    def __repr__(self) -> str:
        return f"CliParams(strategy={self.strategy}, n_clusters={self.n_clusters}, dataset={self.dataset})"

    def __str__(self) -> str:
        return self.__repr__()
"""
