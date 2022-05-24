from abc import abstractmethod
from clustergen.types import DatasetEnum

class ModelProxy:

    def __init__(self, strategy: StrategyEnum, n_clusters: int):
        self.strategy = strategy
        self.n_clusters = n_clusters

    def resolve(self) -> ClusteringModel:
        model_class, param = models_table[self.strategy]
        model = model_class(**{param: self.n_clusters})
        return model

