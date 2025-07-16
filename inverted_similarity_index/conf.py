# Copyright James Clampffer - 2025

import argparse

def add_conf_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--results_limit",
        type=int,
        default=15,
        help="Maxinum number of results to return",
    )
    parser.add_argument(
        "--local_overscan_factor",
        type=int,
        default=10,
        help="Multiply local fetch count by this much prior to shard search",
    )
    parser.add_argument(
        "--idx_tree_count",
        type=int,
        default=25,
        choices=[i for i in range(5, 50, 5)],
        help="number of trees to use in an annoy index",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="nomic-embed-text",
        choices=[
            "nomic-embed-text",
            "granite-embedding:30m",
            "snowflake-arctic-embed:137m",
        ],
        help="model to use to generate text emebddings",
    )
    parser.add_argument(
        "--merge_shards",
        type=bool,
        default=False,
        help="test-only option to merge shards prior to final ranking",
    )
    return parser


class SimpleConfig:
    """Bare minimum config for now"""

    __slots__ = [
        "_qry_oversample_ratio",  # Local oversampling
        "_qry_max_results",  # Max to report after global ranking
        "_idx_tree_count",  # Specific to Annoy
        "_idx_merge_shards",  # [test only]: run the MergeShard after building individual shards
        "_embedding_model_name",  # nomic, snowflake etc
    ]

    _qry_oversample_ratio: int
    _qry_max_results: int
    _idx_tree_count: int
    _idx_merge_shards: bool
    _embedding_model_name: str

    def __init__(self, args):
        self._qry_oversample_ratio = args.local_overscan_factor
        self._qry_max_results = args.results_limit
        self._idx_tree_count = args.idx_tree_count
        self._idx_merge_shards = args.merge_shards
        self._embedding_model_name = args.embed_model

    @property
    def qry_oversample_ratio(self) -> int:
        return self._qry_oversample_ratio

    @property
    def qry_max_results(self) -> int:
        return self._qry_max_results

    @property
    def idx_tree_count(self) -> int:
        return self._idx_tree_count

    @property
    def idx_merge_shards(self) -> bool:
        return self._idx_merge_shards

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name


_inst = None


def init_config_with_args(args):
    global _inst
    _inst = SimpleConfig(args)


def get_config():
    global _inst
    return _inst
