# Copyright Jim Clampffer - 2025
"""
Proof of concept for a self-hosted semantic similarity search engine.

This defines a text similarity search interface decoupled from the
storage index type or mechanism. It's responsible for broadcasting
a search query to an arbitrary number of independent indices referred
to as shards. Upon receiving shard-level results it unions and reranks
the results in a (hopefully) intelligent manner.

I'm interested in the platform that manages scalability, concurrency,
and redundancy and generally making it easy to run an on-prem RAG or
document search. Indexing high dimensional vectors and the embedding
models that make them are someone else's problem. To that end, this
leans heavily on Annoy, ollama, numpy, torch, etc.
"""

import conf
import token_sources
import util
import shard_merger
from annoy_shard import LatentSpaceIndexShard
from embedding_generator import make_embedding

import argparse
import logging
import typing

# search processing
search_logger = logging.getLogger("ann_search")

search_log_warn = lambda s: search_logger.warning(s)
search_log = lambda s: search_logger.info(s)
search_log_debug = lambda s: search_logger.debug(s)


class FederatedIndexSearch:
    """
    Proof of concept for a self-hosted semantic similarity search engine.

    Defines a text similarity interface decoupled from the underlying index
    type or storage mechanism. It broadcasts queries to an arbitrary number
    of independent index shards, collects and unions their results, then
    runs a global ranking pass.

    The focus here is on platform-level concerns: scalability, concurrency,
    redundancy, and ease of running on-prem RAG or document search setups.
    """

    # Ratio of extra items to return from each shard. Data load can introduce
    # skew where a subset of indicies contain all the relevant results. This
    # could be due to a natural cluster in the latent space or a result of
    # data loaded sequentially ending up in the same shard.
    OverscanFactor: int = 10

    # Refcount will keep shards alive even if they are no longer available
    # to new searches due to background consolidation.
    __slots__ = "_shards", "_global_k"
    _shards: list[LatentSpaceIndexShard]
    _global_k: int | float

    def __init__(self, global_top_k: int = 40):
        self._shards = []
        self._global_k = global_top_k

    def add_shard(self, shard: LatentSpaceIndexShard) -> None:
        self._shards.append(shard)

    def run_federated_search(self, query_embedding) -> list[typing.Any]:
        """
        Run the federated search and combine the results.
        """
        N = len(self._shards)
        local_k = int(self._global_k * N * conf.get_config().qry_oversample_ratio)

        if N == 0:
            # Avoid crashing until bumper rails are implemented
            search_log("No shards available for search. Go Away.")
            return None

        # Temporary shard to merge top-N candidates from each shard.
        # Direct sort on cosine would likely be better, and feasible
        # here due to bounded number of embedding vectors.
        ephemeral_shard = LatentSpaceIndexShard()

        for shard in self._shards:
            # Could run concurrently, but it's fast enough for now.
            if shard == None:
                search_log_warn("Got None for a shard!")
                continue
            search_log("Searching shard: {}".format(shard._name))
            top = shard.search_shard_direct(query_embedding, k=local_k)
            if top == None:
                search_log_warn("Got empty results from shard!")
                break
            for emb, text in top:
                ephemeral_shard.add_embedding(emb, text)

        # Build the index - this is fast, it appears you pay up front with
        # annoy indicies. That's sort of ideal for the access pattern here,
        # where an index is held open right until a search runs.
        ephemeral_shard.build_index()

        # Run a search on the pooled top-ks
        # todo: Make a new shard type to do direct cosine similarity
        # sort. Here there's a bounded number of embeddings and more
        # accurate ranking helps with skew.
        global_top = ephemeral_shard.search_shard(query_embedding, k=self._global_k)
        return global_top


def make_demo_shards() -> list[LatentSpaceIndexShard]:
    """
    Load some public domain books and create a shard for each.

    The 1:1 source->shard relation is arbitrary and won't hold true
    once background index merges are implemented.
    """
    # See token_sources.py
    books = token_sources.get_chosen_texts()
    shardlist: list[LatentSpaceIndexShard] = []

    for book_uri in books:
        # note: eventually parallelize bulk loads
        sh = LatentSpaceIndexShard(book_uri)
        bookchunker = token_sources.GutenbergTextReader(book_uri)

        cnt = 0
        for chunk in bookchunker.read_chunks():
            cnt += 1
            emb = make_embedding(chunk)
            if emb is not None:
                sh.add_embedding(emb, chunk)
            else:
                # todo: log rejects.
                print("Failed to create embedding for chunk: {}".format(chunk[:50]))
        search_log("Embedded {} shards".format(cnt))

        sh.build_index()
        shardlist.append(sh)
    return shardlist


def load_shard_example() -> None:
    """
    @brief Proof of concept interactive interface. Take a user prompt
           and find the most relevant data.
    """
    fed = FederatedIndexSearch(global_top_k=40)
    minishards = make_demo_shards()

    RUN_MERGE = conf.get_config().idx_merge_shards
    if RUN_MERGE:
        consolidator = shard_merger.ShardMerger(minishards)
        consolidated = consolidator.run_merge()
        for sh in consolidated:
            fed.add_shard(sh)
    else:
        for sh in minishards:
            fed.add_shard(sh)

    # Search prompt loop
    while True:
        try:
            qry = input("Enter text or phrase to lookup: ")
            search_log_debug("user entered '{}'".format(qry))
            emb = make_embedding(qry)
            search_log("searching for similar indexed content")
            vals = fed.run_federated_search(emb)
            search_log("got {} search results back".format(len(vals)))
            # Make configurable. Expect some low quality results in set due
            # to skew. Skim the top matches.
            result_limit = conf.get_config().qry_max_results
            for idx, val in enumerate(vals):
                if idx > result_limit:
                    break
                # list top-k matches
                print("\n match {}".format(idx))
                print(val)
                print("-" * 20 + "\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    # always set up first
    util.init_logging()

    parser = argparse.ArgumentParser()

    # Let the config push args directly to the parser
    conf.add_conf_args(parser)
    args = parser.parse_args()
    conf.init_config_with_args(args)

    search_log("Running inverted similarity search")
    load_shard_example()
