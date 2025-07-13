## Federated Approximate Nearest Neighbor Search

Build an on-premise semantic similarity search using embeddings and approximate nearest neighbor (ANN) indexing.

 ### What is this?
This system performs similarity searches using text embeddings—high-dimensional vectors (often 500+ dimensions) encoding semantic features rather than exact matches.

 ### Why make this?
Quickly find documents by meaning, not exact keywords. Unlike generative LLMs, results remain traceable to original documents. Future plans include OCR integration for scanned documents.

Unlike hosted services, this project emphasizes local/hybrid deployment and infrastructure control.

## Quickstart

```
python federated_annoy.py
```

- The prompt loops until a keyboard interrupt (Ctrl+C) is detected.
- Enter a search phrase; matching indexed content is returned.

#### Example use.

Here I'll just say something about a cold ship. It points to shackleton as expected.
```
Enter text or phrase to lookup: The ship got stuck.

searching for similar indexed content
Searching shard: https://www.gutenberg.org/cache/epub/11/pg11.txt
... snip ...
Annoy(f18bc84e7d594ddd8ccebe2655683f21) indexed 1100 embeddings in 0.002552509307861328 seconds
Building index took: 0.002637147903442383 seconds
Global search took: 0.0008308887481689453 seconds
“_July_ 22.—Ship in bad position in newly frozen lane, with bow and
stern jammed against heavy floes; heavy strain with much creaking and
groaning. 8 a.m.—Called all hands to stations for sledges, and made
final preparations for abandoning ship. Allotted special duties to
several hands to facilitate quickness in getting clear should ship be
crushed. Am afraid the ship’s back will be broken if the pressure
continues, but cannot relieve her.
<more results cut out>
```

### Limitations to sort out in an upcoming commit
Planned
- Original source propagation

- Static resource caching and index persistence.

- Shard management infrastructure

- Parallel data ingest

- Multinode scaleout over TCP.

Longer term

- User authentication and permissions

- Fault tolerence and load balancing when running on multiple nodes.

## System Overview

### Index Shards (`LatentSpaceIndexShard`)
- Built using Annoy for fast similarity search.
- Inverted text index to match original text when returning an embedding.
- Background process will handle merging shards and purging deleted embeddings

### Federated Search (`FederatedIndexSearch`)
- Runs top-K across all shards with oversampling (the overscan factor)
  - Overscan factor: ratio of extra items to return from the local top-K from each shard.
- Re-ranks globally using an ephemeral Annoy index built from per-shard candidates

### Token Source

Abstract chunking methods for flexible text preprocessing. The current example produces paragraph level chunks.