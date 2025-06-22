"""
Jim Clampffer 2025

@brief Proof of concept for a self-hosted approximate index for
"""

import ollama
import time
import annoy
import uuid

import token_sources


def make_embedding(text: str):
    if 0 == len(text):
        return None
    try:
        return ollama.embeddings(model="nomic-embed-text", prompt=str(text))[
            "embedding"
        ]
    except Exception as e:
        return None


class LatentSpaceIndexShard:
    """
    @brief Embedding index coupled with an inverted index to fetch the
           source text.
    @todo  Add metadata to text chunks to map back to the correct
           portion of the original doc rather than matches.
    """

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
        self.annoy_index = annoy.AnnoyIndex(
            len(make_embedding("hello world")), "angular"
        )
        self.built = False

    def add_embedding(self, vec: list[float], text: str) -> None:
        """
        @brief For each block of text in vec generate an embedding
               vector. Add that to the Annoy search idx, and add the
               original text to the inverted idx dict.
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
        todo: Save the annoy and inverted index as a unit.
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
        @copydoc search_shared_direct
        @todo get rid of this, leftover from when the federated search
              could also push the search phrase.
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
        For use when consolidating shards. Return the full embeddings,
        as well as the content and metadata used to create them.
        """
        buf = []
        for emb, src in self.inverted_idx.items():
            buf.append((emb, src))
        return buf


class FederatedIndexSearch:
    """
    Run top-k on child indicies, group that, then run a global top-k
    on the results. This hides the fact there are multiple index
    structures Participating in answering searchs.
    """

    # Ratio of extra items to return from each shard. Data load can
    # introduce skew where a subset of indicies contain all the
    # relevant results. Pull an overscan amount from each shart as a
    # workaround: tok-k at this level turns top-(k*overscan) for each
    # shard.
    OverscanFactor = 5

    # Refcount will keep shards alive even if they are no longer available
    # to new searches due to background consolidation.
    __slots__ = "shards", "global_k"

    def __init__(self, global_top_k=40):
        self.shards = []
        self.global_k = global_top_k

    def add_shard(self, shard: LatentSpaceIndexShard) -> None:
        self.shards.append(shard)

    def run_federated_search(self, query_embedding):
        """
        @brief Run a similarity search of the embedding in each shard.
               Union those results and make a single top-k similarity
               ordwered set.
        """

        N = len(self.shards)
        local_k = int(self.global_k * FederatedIndexSearch.OverscanFactor)

        if N == 0:
            # Avoid crashing until bumper rails are implemented
            return None

        # Temporary shard to handle the top-k of the top-n lists from each
        # consider cosine or dot prod similarity rather than a new index.
        ephemeral_shard = LatentSpaceIndexShard()

        candidate_chunk_count = 0
        serial_search_start = time.time()
        for shard in self.shards:
            print("Searching shard: {}".format(shard.name))
            top = shard.search_shard_direct(query_embedding, k=local_k)
            for emb, text in top:
                ephemeral_shard.add_embedding(emb, text)
        serial_search_end = time.time()

        if False:
            print(
                "Federated search took: {} seconds for {} sub-searches".format(
                    serial_search_end - serial_search_start, len(self.shards)
                )
            )
            print(
                "Searching for top-{} out of {} consolidated".format(
                    self.global_k, candidate_chunk_count
                )
            )

        index_pre = time.time()
        ephemeral_shard.build_index()
        index_post = time.time()

        print("Building index took: {} seconds".format(index_post - index_pre))

        # Run a search on the pooled top-ks
        search_pre = time.time()
        global_top = ephemeral_shard.search_shard(query_embedding)
        search_post = time.time()
        print("Global search took: {} seconds".format(search_post - search_pre))
        return global_top


def load_shared_example():
    """
    @brief Proof of concept interactive interface. Take a user prompt
           and find the most relevant data.
    @note  Data is a hard-coded list from project Gutenberg.
    """

    # See token_sources.py
    books = token_sources.get_chosen_texts()
    fed = FederatedIndexSearch(20)

    for book_uri in books:
        # note: eventually parallelize bulk loads
        sh = LatentSpaceIndexShard(book_uri)
        bookchunker = token_sources.GutenbergTextReader(book_uri)

        t1 = time.time()
        cnt = 0
        for chunk in bookchunker.read_chunks():
            cnt += 1
            emb = make_embedding(chunk)
            if emb is not None:
                sh.add_embedding(emb, chunk)
            else:
                print("Failed to create embedding for chunk: {}".format(chunk[:50]))
        t2 = time.time()
        print(
            "Chunk {} took: {} seconds or {}/s".format(
                book_uri, t2 - t1, cnt / float(t2 - t1)
            )
        )

        sh.build_index()
        fed.add_shard(sh)

    # now pull up context based on prompt
    while True:
        try:
            emb = make_embedding(input("Enter text or phrase to lookup: "))
            print("searching for sinilar indexed content")
            vals = fed.run_federated_search(emb)
            for val in vals:
                print(val[1])
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    load_shared_example()
