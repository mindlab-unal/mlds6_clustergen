import matplotlib.pyplot as plt
from clustergen.training import Trainer

class ClusteringVisualization:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

    def draw(self) -> "ClusteringVisualization":
        train_results = self.trainer.get_cache()
        self.ax.scatter(
                train_results.data_cache[:, 0],
                train_results.data_cache[:, 1],
                c=train_results.labels_cache
                )
        return self

    def save(self, path: str):
        self.fig.savefig(path)
