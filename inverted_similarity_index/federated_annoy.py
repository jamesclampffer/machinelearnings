# Copyright Jim Clampffer 2025
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
import token_sources
from annoy_shard import LatentSpaceIndexShard
from embedding_generator import make_embedding

import typing


class FederatedIndexSearch:
    """
    Proof of concept for a self-hosted semantic similarity search engine.

    Defines a text similarity interface decoupled from the underlying index
    type or storage mechanism. It broadcasts queries to an arbitrary number
    of independent index shards, merges their results, and re-ranks them
    with (hopefully) reasonable logic.

    The focus here is on platform-level concerns: scalability, concurrency,
    redundancy, and ease of running on-prem RAG or document search setups.
    Embedding models and vector indexing are out of scope—they're delegated
    to external tools like Annoy, Ollama, NumPy, and Torch.
    """

    # Ratio of extra items to return from each shard. Data load can introduce
    # skew where a subset of indicies contain all the relevant results. This
    # could be due to a natural cluster in the latent space or a result of
    # data loaded sequentially ending up in the same shard.
    OverscanFactor: int = 5

    # Refcount will keep shards alive even if they are no longer available
    # to new searches due to background consolidation.
    __slots__ = "shards", "global_k"
    shards: list[LatentSpaceIndexShard]
    global_k: int | float

    def __init__(self, global_top_k: int = 40):
        self.shards = []
        self.global_k = global_top_k

    def add_shard(self, shard: LatentSpaceIndexShard) -> None:
        self.shards.append(shard)

    def run_federated_search(self, query_embedding) -> list[typing.Any]:
        """
        Run the federated search and combine the results.
        """
        N = len(self.shards)
        local_k = int(self.global_k * FederatedIndexSearch.OverscanFactor)

        if N == 0:
            # Avoid crashing until bumper rails are implemented
            print("No shards available for search. Go Away.")
            return None

        # Temporary shard to merge top-N candidates from each shard.
        # Direct sort on cosine would likely be better, and feasible
        # here due to bounded number of embedding vectors.
        ephemeral_shard = LatentSpaceIndexShard()

        for shard in self.shards:
            # Could run concurrently, but it's fast enough for now.
            print("Searching shard: {}".format(shard.name))
            top = shard.search_shard_direct(query_embedding, k=local_k)
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
        global_top = ephemeral_shard.search_shard(query_embedding)
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
    for sh in minishards:
        fed.add_shard(sh)

    # Search prompt loop
    while True:
        try:
            emb = make_embedding(input("Enter text or phrase to lookup: "))
            print("searching for similar indexed content")
            vals = fed.run_federated_search(emb)
            for val in vals:
                print(val[1])
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    load_shard_example()
