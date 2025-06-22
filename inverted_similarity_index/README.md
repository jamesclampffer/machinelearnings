## Federated approx nearest neighbor search

This is intended to look at how hard it'd be to run an on-prem embedding index using existing algorithms. The only notable thing going on is being build to handle concurrent loads and queries efficiently.

### To run:
```
python federated_annoy.py
```
- The prompt will loop until a keyboard interrupt is detected.

#### Example use.

Here I'll just say something about a cold ship. It points to shackleton as epected.
```
Enter text or phrase to lookup: The ship got stuck.

searching for sinilar indexed content
Searching shard: https://www.gutenberg.org/cache/epub/11/pg11.txt
Searching shard: https://www.gutenberg.org/cache/epub/5200/pg5200.txt
Searching shard: https://www.gutenberg.org/cache/epub/7849/pg7849.txt
Searching shard: https://www.gutenberg.org/cache/epub/25344/pg25344.txt
Searching shard: https://www.gutenberg.org/cache/epub/205/pg205.txt
Searching shard: https://www.gutenberg.org/cache/epub/16643/pg16643.txt
Searching shard: https://www.gutenberg.org/cache/epub/29433/pg29433.txt
Searching shard: https://www.gutenberg.org/cache/epub/1232/pg1232.txt
Searching shard: https://www.gutenberg.org/cache/epub/66944/pg66944.txt
Searching shard: https://www.gutenberg.org/cache/epub/48874/pg48874.txt
Searching shard: https://www.gutenberg.org/cache/epub/5199/pg5199.txt
Annoy(f18bc84e7d594ddd8ccebe2655683f21) indexed 1100 embeddings in 0.002552509307861328 seconds
Building index took: 0.002637147903442383 seconds
Global search took: 0.0008308887481689453 seconds
“_July_ 22.—Ship in bad position in newly frozen lane, with bow and
stern jammed against heavy floes; heavy strain with much creaking and
groaning. 8 a.m.—Called all hands to stations for sledges, and made
final preparations for abandoning ship. Allotted special duties to
several hands to facilitate quickness in getting clear should ship be
crushed. Am afraid the ship’s back will be broken if the pressure
continues, but cannot relieve her. 2 p.m.—Ship lying easier. Poured
Sulphuric acid on the ice astern in hopes of rotting crack and
relieving pressure on stern-post, but unsuccessfully. Very heavy
pressure on and around ship (taking strain fore and aft and on
starboard quarter). Ship, jumping and straining and listing badly. 10
p.m.—Ship has crushed her way into new ice on starboard side and slewed
aslant lane with stern-post clear of land-ice. 12 p.m.—Ship is in safer
position; lanes opening in every direction.

<more results cut out>

```



### Limitations to sort out an an upcoming commit
- No shard management infrastructure yet.
- Data ingest is trivially parallelizable, so do that.
- No network API
- No auth or notion of users. There are cases where documents should be masked from the result set of some users.

### Why make this?
Imagine one has a ton of documents and often forgets the exact titles but does remember specific context. This provides a search that can look for documents based on a similar phrase, so one doesn't need to know exactly which file contained some info.

There are plenty of hosted solutions for this. I want something that is under my control running on-prem to manage sensitive information. Embeddings happen to be one of the more interesting ML topics I've run into so far. The underlying data structures for indicies are also interesting. That wasn't the focus here.

### What is this?
An embedding model will take some input, often a short chunk of text (depends on model and use). Given that input it'll run through a pretrained model and return an embedding vector.

The embedding vectors are generally >500 dimensions and are use the weights in that vector to encode features of the text in a high dimensional space. In toy models, or vector generators that use a CNN frontend for image classification the weights map reasonably well to understandable concepts. In this sort of embedding model the vector represents much more abstract traits to encode more semantic meaning.

## System Overview

- Index Shards (`LatentSpaceIndexShard`)
  - Built using [Annoy](https://github.com/spotify/annoy) because it was very easy to set up and works fairly well.
  - Inverted text index to match original text when returning an embedding.
  - Background process will eventually handle merging shards and purging deleted embeddings.
- Federated Search (`FederatedIndexSearch`)
  - Runs top-K across all shards with oversampling (overscan factor)
  - Re-ranks globally using an ephemeral Annoy index built from per-shard candidates