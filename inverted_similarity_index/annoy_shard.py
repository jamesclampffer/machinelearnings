# Copyright James Clampffer - 2025

import annoy
import conf
import embedding_generator
import logging
import time
import uuid

search_logger = logging.getLogger("ann_search")
system_logger = logging.getLogger("platform")

ann_log = lambda s: search_logger.info(s)
sys_log = lambda s: system_logger.info(s)


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
        "_name",  # Shard name / ID
        "_annoy_index",  # Annoy index for similarity search
        "_inverted_idx",  # Inverted embedding ID->original text index
        "_built",  # Once an Annoy index is built it's immutable
        "_embedding_num",  # Counter to generate shard-unique IDs for chunks
        "_insert_time_s",  # time each addition of embedding and sum them
    )

    _name: str
    _annoy_index: annoy.AnnoyIndex
    _built: bool
    _embedding_num: int

    def __init__(self, name: str = uuid.uuid4().hex):
        """
        @brief Set up the annoy index
        @note federator and shard design is intended to decouple logic
              from the similarity index implementation.
        @param name - name for this shard. For now something easy to
                      debug is all that's needed.
        """
        # monotonic counter for instance-unique embedding ID.
        self._embedding_num = 0

        # More relevant later when ancestry of merged or purged shards in a
        # multinode deployment get implemented.
        self._name = name

        # Map a embedding ID back to the source text with a traditional
        # inverted idx.
        #
        # Annoy doesn't support deletes, the inverted index can replace the text with a
        # tombstone marker to indicate it shall not be included in search results and
        # will be omitted if/when the search goes through a merge.
        self._inverted_idx = dict()

        # For now, assumes all shards use the same embedding model.
        if LatentSpaceIndexShard.DIM == None:
            tmp = "hello world"
            LatentSpaceIndexShard.DIM = len(embedding_generator.make_embedding(tmp))

        self._built = False
        self._annoy_index = annoy.AnnoyIndex(LatentSpaceIndexShard.DIM, "angular")

        # Add elapsed time from each add_embedding call
        self._insert_time_s = 0.0

    def add_embedding(self, vec: list[float], text: str) -> None:
        """
        @brief For each block of text in vec generate an embedding vector. Add that to
               the Annoy search idx, and add the original text to the inverted idx dict.
        @todo Encapsulate source document and position associated with the embedding.
        """
        emb_start = time.time()
        self._annoy_index.add_item(self._embedding_num, vec)

        # Consider optional text block compression, or just maintain a pointer
        # back to the source document with an offset in memmory.
        self._inverted_idx[self._embedding_num] = text
        self._embedding_num += 1
        self._insert_time_s += time.time() - emb_start

    @property
    def _embedding_elapsed_time_s(self):
        return self._insert_time_s

    def build_index(self) -> bool:
        """
        @brief Finalize the index structure.
        @post  Index becomes readable, no longer writable.
        @note  With Annoy it seems like almost all the overhead comes
               with the initial insert. Building is fairly snappy which
               lends itself well deferring building the most recent idx.
        """
        if self._built == True:
            # idempotent
            return True
        try:
            t1 = time.time()
            self._annoy_index.build(
                conf.get_config().idx_tree_count
            )  # make configurable
            t2 = time.time()
            sys_log(
                "Total time calculating embedding vectors: {}".format(
                    self._embedding_elapsed_time_s
                )
            )
            sys_log(
                "Annoy({}) indexed {} embeddings in {} seconds".format(
                    self._name, self._embedding_num, t2 - t1
                )
            )
        except:
            return False
        self._built = True

    def save_index(self, path: str) -> bool:
        """
        @brief: persist to disk (it'll be rebuilt on start for now)
        @todo:  load from disk. Take care of this along with uri doc
                caching.
        """
        self._annoy_index.save("test_shard_{}".format(self._name) + ".annoy")

    def search_shard_direct(self, query_embedding, k=10) -> list[list, str]:
        """
        @brief Take the embedding of a search phrase and run a top-k
               similarity search of indexed contents.
        """
        if query_embedding == None:
            return None

        # This shouldn't be possible when the FederatedSearch is working correctly,
        # however it is nice for catching nonsense in unit tests.
        assert self._built

        # Resolve the top-k IDs to the source text
        embids = self._annoy_index.get_nns_by_vector(
            query_embedding,
            k,
        )
        vals = []
        for embid in embids:
            text = self._inverted_idx[embid]
            embed = self._annoy_index.get_item_vector(embid)
            vals.append((embed, text))
        return vals

    def search_shard(self, query_embedding, k=10):
        """
        @brief Run a search and return by index ID rather than
               including embedding
        """
        if self._built == False:
            raise ValueError("Annoy index not built yet.")

        embids = self._annoy_index.get_nns_by_vector(query_embedding, k * 5)
        vals = []
        for embid in embids:
            vals.append((embid, self._inverted_idx[embid]))
        return vals

    def get_kv_pairs(self) -> list[tuple[list[float], str]]:
        """
        @brief For use when consolidating shards. Return the full
               embeddings and source data
        #todo  Generator interface.
        """
        buf = []

        for idx, src in self._inverted_idx.items():
            emb: list[float] = self._annoy_index.get_item_vector(idx)
            buf.append((emb, src))
        return buf
