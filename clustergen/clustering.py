from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from clustergen.types import StrategyEnum, ClusteringModel

models_table = {
        StrategyEnum.KMEANS: [KMeans, "n_clusters"],
        StrategyEnum.SPECTRAL: [SpectralClustering, "n_clusters"],
        StrategyEnum.GMM: [GaussianMixture, "n_components"]
        }

class ModelProxy:

    def __init__(self, strategy: StrategyEnum, n_clusters: int):
        self.strategy = strategy
        self.n_clusters = n_clusters

    def resolve(self) -> ClusteringModel:
        model_class, param = models_table[self.strategy]
        model = model_class(**{param: self.n_clusters})
        return model

