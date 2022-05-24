from argparse import ArgumentParser, Namespace
from clustergen.types import StrategyEnum, DatasetEnum, CliParams
from clustergen.clustering import ModelProxy
from clustergen.datasets import DatasetProxy
from clustergen.training import Trainer
from clustergen.visualization import ClusteringVisualization

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Terminal utility to compare different clustering strategies.")
    parser.add_argument(
            "--strategy", type=str, required=True,
            help="Clustering strategy to use. {'kmeans', 'gmm', 'spectral'}"
            )
    parser.add_argument(
            "--n_clusters", type=int, default=2,
            help="Number of clusters to find."
            )
    parser.add_argument(
            "--n_samples", type=int, default=1000,
            help="Number of samples to generate."
            )
    parser.add_argument(
            "--noise", type=float, default=0.1,
            help="Noise level."
            )
    parser.add_argument(
            "--dataset", type=str, default="blobs",
            help="Type of generated dataset. {'blobs', 'moons', 'circles'}"
            )
    parser.add_argument(
            "--save_path", type=str, default="im.png",
            help="Save path."
            )
    return parser

def validate_args(args: Namespace) -> CliParams:
    try:
        strategy = StrategyEnum[args.strategy.upper()]
    except:
        raise Exception(f"<{args.strategy}> is not a valid StrategyEnum.")

    try:
        dataset = DatasetEnum[args.dataset.upper()]
    except:
        raise Exception(f"<{args.dataset}> is not a valid DatasetEnum.")

    return CliParams(
            strategy=strategy, dataset=dataset,
            n_clusters=args.n_clusters, n_samples=args.n_samples,
            noise=args.noise, save_path=args.save_path
            )

def main():
    parser = make_parser()
    cliparams = validate_args(parser.parse_args())

    model_proxy = ModelProxy(cliparams.strategy, cliparams.n_clusters)
    dataset_proxy = DatasetProxy(cliparams.dataset, cliparams.noise, cliparams.n_samples)

    model = model_proxy.resolve()
    dataset = dataset_proxy.resolve()

    trainer = Trainer(model, dataset).train()

    visualization = ClusteringVisualization(trainer).draw()
    visualization.save(cliparams.save_path)
