# Copyright James Clampffer - 2025
"""
Expose a easy to use interface for generating embeddings.

Plug in some text and get an embedding vector in return. Caching is not
implemented - input is assumed to be high mix. Dedupe can happen in the
background as part of shard consolidation.
"""

import logging
import ollama
import os
import time
import traceback

import conf

# things related to embedding model
system_logger = logging.getLogger("platform")
# things related to text->embedding vector
ingress_logger = logging.getLogger("data_ingress")

sys_log = lambda s: system_logger.info(s)
ing_log = lambda s: ingress_logger.info(s)

# Some <1GiB embedding models to produce vectors of similar dimensionality.
# Nomic works well. Evaluating the others so the default may change.
OLLAMA_EMB_MODELS = {
    "nomic-embed-text",
    "granite-embedding:30m",
    "snowflake-arctic-embed:137m",
}


# Indirect in case it's changed in a running process. Other issues to address
# in that case as well.
def get_emb_model_name(params: str | None = None) -> str:
    """
    Look for a model name in an env var. Sane default if not found.
    """
    default = conf.get_config().embedding_model_name
    modelname = os.getenv("TEXT_EMB_MODEL", default)
    if modelname not in OLLAMA_EMB_MODELS:
        modelname = "nomic-embed-text"
    return modelname


emb_call_cnt = 0  # time to put this in a class
time_spent_embedding = 0.0


def make_embedding(text: str, normalize=True) -> list[float] | None:
    """
    Make an embedding vector from input text.

    The input is generally a few words to a few paragraphs long. Exact
    chunking approach may be input dependent: paragraphs, sections of a
    BOM. Optionlly normalize the input to lower case and single whitespaces.

    TGiven that the ollama client interface is over http it'll be easy to distribute
    work over many independeny nodes, or ollama process-per-gpu multi-gpu setups.
    """
    global emb_call_cnt
    global time_spent_embedding

    t1 = time.time()  # monotonic timer later

    if normalize:
        text = text.lower().strip()

    # Turn into empty string? Filter format specific noise during chunking.
    if 0 == len(text):
        return None

    try:
        # todo: numpy array nemory savings vs. conversion cost eval
        emb = ollama.embeddings(model=get_emb_model_name(), prompt=str(text))[
            "embedding"
        ]
        elapsed = time.time() - t1
        time_spent_embedding += elapsed
        emb_call_cnt += 1
        if emb_call_cnt // 10:
            avg_emb_s = time_spent_embedding / float(emb_call_cnt)
            sys_log(
                "model {} embedded {} avg time {}s".format(
                    get_emb_model_name(), emb_call_cnt, avg_emb_s
                )
            )

        return emb
    except Exception as e:
        sys_log("{}\ncaught:\n{}\n".format("make embedding", traceback.format_exc()))
        return None


if __name__ == "__main__":
    pass
