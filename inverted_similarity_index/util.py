# Copyright James Clampffer - 2025
"""
Common bits like logging and config. Bifurcate file once there's enough
implementation for that to make sense.
"""

import logging


def init_logging() -> None:
    """
    Set up loggers:
        ann_search: query processing
        data_ingress: sourcing and chunking input
        data_management: data in system being merged, deleted, moved etc.
        system: file IO, embedding model runner

    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Avoid logging every IPC over http to ollama
    logging.getLogger("httpx").setLevel(logging.WARNING)


if __name__ == "__main__":
    pass
