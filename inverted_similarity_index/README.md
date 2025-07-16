## Federated Approximate Nearest Neighbor Search
Infrastructure for a similarity search engine.

### What is this?
This project lets you semantically search text collections using high-dimensional embeddings (latent vectors), not just matching keywords or tokens.

### Why?
Find documents by meaning, not literal phrasing—results always trace back to the actual document chunk. 

Unlike hosted services, everything can run locally or on-prem for full data control. Hybrid deloyment is also possible e.g. a high-end instance to run the embedding model for bulk ingest.

## Quickstart

```
python federated_annoy.py
```

- The prompt loops until you interrupt (Ctrl+C).
- Enter a search phrase; the system returns the best-matching indexed content based on vector similarity.

#### Example use.

I'll just say something about a cold ship. It points to shackleton as expected.
```
Enter text or phrase to lookup: The ocean froze over
searching for similar indexed content
Searching shard: https://www.gutenberg.org/cache/epub/5200/pg5200.txt
Searching shard: https://www.gutenberg.org/cache/epub/25344/pg25344.txt
Searching shard: https://www.gutenberg.org/cache/epub/205/pg205.txt
Searching shard: https://www.gutenberg.org/cache/epub/66944/pg66944.txt
Searching shard: https://www.gutenberg.org/cache/epub/48874/pg48874.txt
Searching shard: https://www.gutenberg.org/cache/epub/5199/pg5199.txt
Total time calculating embedding vectors: 0.06741046905517578
Annoy(2704d2564890442a9776984fec1de2f6) indexed 4623 embeddings in 0.041797637939453125 seconds
got 200 search results back

 match 0
(3532, '“_April_ 10, 1.30 p.m.—Ice breaking from shore under influence of\n\nsouth-east wind. Two starboard quarter wires parted; all bights of\n\nstern wires frozen in ice; chain taking weight. 2 p.m.—Ice opened,\n\nleaving ice in bay in line from Cape to landward of glacier. 8\n\np.m.—Fresh wind; ship holding ice in bay; ice in Sound wind-driven to\n\nnorth-west.')

 match 1
(3533, '“Since the ship had been moored the bay had frequently frozen over, and\n\nthe ice had as frequently gone out on account of blizzards. The ice\n\ndoes not always go out before the wind has passed its maximum. It\n\ndepends on the state of tides and currents; for the sea-ice has been\n\nseen more than once to go out bodily when a blizzard had almost\n\ncompletely calmed down.')
```

### Limitations to sort out in an upcoming commit
Planned
- Original source propagation
- Metadata catalog and index persistence
- Data Management infrastructure
- Multinode scaleout over TCP.

Longer term
- User authentication and permissions
- Fault tolerence and load balancing when running on multiple nodes.

## System Overview

### Main Components

#### LatentSpaceIndexShard (annoy_shard.py)

  Encapsulates an Annoy approximate nearest neighbor (ANN) index. Manages storage of embedding vectors and an inverted mapping to original text chunks.

#### Embedding Generator (embedding_generator.py)

  Provides a simple API for generating text embeddings using local Ollama models. Embedding model selection is configurable via environment variable.

#### Federated Search Orchestrator (FederatedIndexSearch in federated_annoy.py)

  Coordinates queries across all loaded shards, merges and reranks results, and returns the most relevant text segments matching the user’s input.

#### Text Chunking and Ingest (token_sources.py)

  Downloads remote text resources (e.g., Project Gutenberg), splits them into paragraph-like chunks, and serves these for embedding and indexing.

#### Local Document Cache (local_document_cache.py)

Simple file-based persistent cache to avoid redundant network downloads of the same resources during data ingestion.

#### Shard Merger (shard_merger.py)

Supports dynamic recombination of shards into a single index, useful for index consolidation and future platform scaling.

### Data & Control Flow

#### Data Ingestion:
- Documents are fetched and chunked (token_sources.py).
- Chunks are embedded (embedding_generator.py).
- Embedding–chunk pairs are added to new or existing shards (LatentSpaceIndexShard).
#### Caching:
- All file downloads go through local_document_cache.py to avoid spamming 3rd party hosts during repeated ingestion.
#### Search:
- User provides a text query via CLI.
- Federated search broadcasts the embedded query to each loaded shard, retrieving top matches.
- Candidate results are globally reranked to improve relevance before being presented.
#### Merging:
- Optional: multiple shard indices can be consolidated via the ShardMerger, producing a unified search index.


### Component Interactions

#### token_sources ↔ local_document_cache:
- Leverages the cache for efficient and resilient resource downloads (method calls, file IO).
#### token_sources ↔ embedding_generator:
- Transforms each chunk of text into an embedding vector (method call).
#### federated_annoy ↔ annoy_shard:
- Handles adding data to shards, querying for matches, and building indices (method calls).
#### federated_annoy ↔ shard_merger:
- Supports merging of indices for more advanced workflows (method call).
#### System ↔ User:
- The primary interface is command-line interaction.

### Extension Points

Modular structure allows for swapping in alternative chunking strategies, embedding backends, vector search engines (e.g., FAISS, ScaNN), or more sophisticated federated query and sharding logic with minimal codebase disruption.
