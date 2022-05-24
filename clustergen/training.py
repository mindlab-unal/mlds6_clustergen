from dataclasses import dataclass
from numpy import ndarray as Array
from clustergen.types import ClusteringModel
from clustergen.datasets import Dataset

@dataclass
class TrainingResults:
    data_cache: Array
    labels_cache: Array

class Trainer:
    def __init__(self, model: ClusteringModel, dataset: Dataset):
        self.model = model
        self.dataset = dataset
        self.data_cache: Array
        self.labels_cache: Array

    def train(self) -> "Trainer":
        X = self.dataset.extract()
        self.model.fit(X)

        self.data_cache = X
        self.labels_cache = self.model.labels_
        return self

    def get_cache(self) -> TrainingResults:
        return TrainingResults(
                data_cache=self.data_cache, labels_cache=self.labels_cache
                )

    @property
    def results(self):
        return self.model.labels_

