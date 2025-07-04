# Copyright James Clampffer 2025
"""
Expose a easy to use interface for generating embeddings.

Plug in some text and get an embedding vector in return. Caching is not
implemented - input is assumed to be high mix. Dedupe can happen in the
background as part of shard consolidation.
"""

import ollama
import os
import traceback


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
    modelname = os.getenv("TEXT_EMB_MODEL", "nomic-embed-text")
    if modelname not in OLLAMA_EMB_MODELS:
        print("Model {} not found. Using default".format(modelname))
        modelname = "nomic-embed-text"

    return modelname


def make_embedding(text: str, normalize=True) -> list[float] | None:
    """
    Make an embedding vector from input text.

    The input is generally a few words to a few paragraphs long. Exact
    chunking approach may be input dependent: paragraphs, sections of a
    BOM. Optionlly normalize the input to lower case and single whitespaces.

    Some ollama tickets from the last year or so indicate potential perf
    issues with batched embeddings. Not going to prioritize that - the easy
    to use library is worth it.
    """
    if normalize:
        text = text.lower().strip()

    # Turn into empty string? Filter format specific noise during chunking.
    if 0 == len(text):
        return None

    try:
        # todo: numpy array nemory savings vs. conversion cost eval
        return ollama.embeddings(model=get_emb_model_name(), prompt=str(text))[
            "embedding"
        ]
    except Exception as e:
        print("{}\ncaught:\n{}\n".format("make embedding", traceback.format_exc()))
        return None
