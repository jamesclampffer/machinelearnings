# Copyright James Clampffer - 2025
"""
Operator to consolidate indicies.
- Run in background, avoid having lots of tiny files.
"""
import logging
import typing

import annoy_shard

dm_log = logging.getLogger("data_management")
platform_logger = logging.getLogger("platform")

dm_log = lambda s: dm_log.info(s)
plarform_log = lambda s: platform_logger.info(s)


class ShardMerger:
    """Combine a set of shards"""

    # Sort of like a compacting garbage collector. Logic related to
    # inout selection and atomic swap-in for new federated queries gets
    # complicated. That'll be handled elsewhere.

    shard_dict_t = typing.Dict[str, annoy_shard.LatentSpaceIndexShard]
    shard_t = annoy_shard.LatentSpaceIndexShard

    __slots__ = ("_input_set", "_output_shard", "_built")
    _input_set: list[shard_t]
    _output_shard: list[shard_t] | None
    _built: bool

    def __init__(self, shards_to_merge: list[shard_t]):
        self._input_set = shards_to_merge
        self._output_shard = annoy_shard.LatentSpaceIndexShard()
        self._built = False

    def cancel(self) -> None:
        """Best effort cancel, trash index under construction"""
        # Intended to be used for load shedding. Make this a background
        # task. Fire off a bunch when lightly loaded etc.
        # Might not be able to do anything.
        pass

    def _build_index(self) -> bool:
        if self._output_shard != None:
            built = self._output_shard.build_index()
        if built:
            dm_log("Built index")
            self._built = True

    def run_merge(self) -> list[shard_t]:
        """Iterate through input shards, produce a shard that covers the input set"""
        # Lots of knobs could be added here depending on use case
        # upper bound on time, upper bound on index size,
        dm_log("constructing annoy index")
        for shard in self._input_set:
            # Assume plenty of memory.
            contents: list[tuple[list[float], str]] = shard.get_kv_pairs()
            for pair in contents:
                embedding_vec: list[float] = pair[0]
                original_txt: str = pair[1]

                # In the future optionally rerun through model? Gets rid
                # of upgrade path complications.
                self._output_shard.add_embedding(embedding_vec, original_txt)
        dm_log("done constructing index - remains unbuilt")

        self._build_index()

        # todo: not really storing state. Turn fn or __call__. The
        # latter would help with resource management and reporting.
        return self._output_shard

    if __name__ == "__main__":
        pass
