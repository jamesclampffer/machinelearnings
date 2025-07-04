"""
James Clampffer 2025
"""

import annoy
import embedding_generator
import time
import uuid


class LatentSpaceIndexShard:
    """
    @brief Embedding index coupled with an inverted index to fetch
           source text.
    @todo  Add metadata to text chunks to map back to the correct
           portion of the original doc rather simple matches.
    """

    # Length of the embedding vector.
    DIM: int | None = None

    __slots__ = (
        "name",  # Shard name / ID
        "annoy_index",  # Annoy index for similarity search
        "inverted_idx",  # Inverted embedding ID->original text index
        "built",  # Once an Annoy index is built it's immutable
        "embedding_num",  # Counter to generate shard-unique IDs for chunks
    )

    def __init__(self, name: str = uuid.uuid4().hex):
        """
        @brief Set up the annoy index
        @note federator and shard design is intended to decouple logic
              from the similarity index implementation.
        @param name - name for this shard. For now something easy to
                      debug is all that's needed.
        """
        # monotonic counter for instance-unique embedding ID.
        self.embedding_num = 0

        # More relevant later when ancestry of merged or purged shards in a
        # multinode deployment get implemented.
        self.name = name

        # Map a embedding ID back to the source text with a traditional
        # inverted idx.
        #
        # note: While Annoy doesn't support deletes, the inverted index can
        # replace the text with a tombstone marker to indicate it shall not be
        # included in search results and should be omitted on the next shard
        # merge or rebuild.
        self.inverted_idx = dict()

        # todo: Look into faiss and/or or write a self indexing column-store
        # Use a dummy embedding to figure out dimensionality of latent space
        # this does assume all shards in the same process use the same embedding
        # model. For now that's a reasonable assumption.
        if LatentSpaceIndexShard.DIM == None:
            tmp = "hello world"
            LatentSpaceIndexShard.DIM = len(embedding_generator.make_embedding(tmp))

        self.built = False
        self.annoy_index = annoy.AnnoyIndex(LatentSpaceIndexShard.DIM, "angular")

    def add_embedding(self, vec: list[float], text: str) -> None:
        """
        @brief For each block of text in vec generate an embedding
               vector. Add that to the Annoy search idx, and add the
               original text to the inverted idx dict.
        @todo Use token_sources.TextChunk when adding to the inverted
              index.
        """

        self.inverted_idx[self.embedding_num] = text
        self.annoy_index.add_item(self.embedding_num, vec)

        # Consider optional text block compression, or just maintain a pointer
        # back to the source document with an offset in memmory.
        self.inverted_idx[self.embedding_num] = text
        self.embedding_num += 1

    def build_index(self) -> bool:
        """
        @brief Tell Annoy that it can finalize the index structure.
        @note  With Annoy it seems like almost all the overhead comes
               with the initial insert. Building is fairly snappy which
               lends itself well deferring building the most recent idx.
        """
        if self.built == True:
            # idempotent
            return True
        try:
            t1 = time.time()
            self.annoy_index.build(10)
            t2 = time.time()
            print(
                "Annoy({}) indexed {} embeddings in {} seconds".format(
                    self.name, self.embedding_num, t2 - t1
                )
            )
        except:
            return False
        self.built = True

    def save_index(self, path: str) -> bool:
        """
        @brief: persist to disk (it'll be rebuilt on start for now)
        @todo:  load from disk. Take care of this along with uri doc
                caching.
        """
        self.annoy_index.save("test_shard_{}".format(self.name) + ".annoy")

    def search_shard_direct(self, query_embedding, k=10) -> list[list, str]:
        """
        @brief Take the embedding of a search phrase and run a top-k
               similarity search of indexed contents.
        """
        if query_embedding == None:
            return None

        # This shouldn't be possible when the fedorator is working correctly, however
        # it is nice for catching nonsense in unit tests.
        assert self.built

        # Resolve the top-k IDs to the source text
        embids = self.annoy_index.get_nns_by_vector(
            query_embedding, k, search_k=k * 100 * 16
        )
        vals = []
        for embid in embids:
            text = self.inverted_idx[embid]
            embed = self.annoy_index.get_item_vector(embid)
            vals.append((embed, text))
        return vals

    def search_shard(self, query_embedding, k=10):
        """
        @brief Run a search and return by index ID rather than
               including embedding
        """
        if self.built == False:
            raise ValueError("Annoy index not built yet.")

        embids = self.annoy_index.get_nns_by_vector(
            query_embedding, k, search_k=k * 100 * 16
        )
        vals = []
        for embid in embids:
            vals.append((embid, self.inverted_idx[embid]))
        return vals

    def get_kv_pairs(self) -> list[tuple[list[float], str]]:
        """
        @brief For use when consolidating shards. Return the full
               embeddings and source data
        #todo  Generator interface.
        """
        buf = []
        for emb, src in self.inverted_idx.items():
            buf.append((emb, src))
        return buf
