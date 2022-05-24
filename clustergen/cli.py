from argparse import ArgumentParser, Namespace
from clustergen.types import StrategyEnum, DatasetEnum, CliParams
from clustergen.clustering import ModelProxy

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Terminal utility to compare different clustering strategies.")
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="blobs")
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
            strategy=strategy, dataset=dataset, n_clusters=args.n_clusters
            )

def main():
    parser = make_parser()
    cliparams = validate_args(parser.parse_args())

    model_proxy = ModelProxy(cliparams.strategy, cliparams.n_clusters)

    model = model_proxy.resolve()
    print(model)
