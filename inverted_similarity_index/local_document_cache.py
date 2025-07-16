# Copyright James Clampffer - 2025
"""Avoid pulling from the same static URL"""

import logging
import os
from hashlib import sha256 as stable_hash

system_logger = logging.getLogger("platform")
sys_logger = lambda s: system_logger.info(s)


class DocumentCache:
    """Persistant Key-Value Store. Implementation defined medium"""

    # This doesn't handle data management. Files may get deleted out from
    # under the cache. Mostly to keep a copy of a resource handy for testing
    # and multimode chunk generation.

    __slots__ = "_datadir"
    _datadir: str

    def __init__(self, datadir=""):
        self._datadir = datadir

    def try_lookup(self, uri: str) -> str | None:
        path = self._uri_to_path(uri)
        buf = ""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as fin:
                buf = fin.read()
        except Exception as e:
            sys_logger("tried reading {} :{}".format(uri, str(e)))
            return None
        return buf

    def save_document(self, uri: str, blob: str) -> bool:  # true if stored
        """Persist the blob, future lookup with URL"""
        path = self._uri_to_path(uri)
        if os.path.exists(path):
            # Not dealing with collisions for now
            sys_logger("file name already exists for {}".format(uri))
            return False
        try:
            with open(path, "w") as fout:
                fout.write(blob)
                sys_logger("saved {} to {}".format(uri, path))
            return True
        except Exception as e:
            sys_logger(str(e))
            return False

    def _uri_to_path(self, uri: str) -> str:
        """Convert URI into a file name"""
        # NOTE: does not strip URI query portion
        uri = uri.lower()
        fname = (
            str(stable_hash(uri.encode("utf-8")).hexdigest()).replace("-", "") + ".dat"
        )
        path = os.path.join(self.datadir, fname)
        sys_logger("converted '{}' to '{}'".format(uri, path))
        return path

    @property
    def datadir(self) -> str:
        return self._datadir


def minitest():
    """Smoke test when module is run directly"""
    cache = DocumentCache()
    p1 = ("http://www.foo.bar/path/book.txt", "all work and no play")
    p2 = ("http://website.com/a/b/c.txt?foo=bar?1=2", "makes jim a dull boy")

    # Miss case requires tracking files for deletion, not critical now.
    # Populate
    uri, blob = p1
    cache.save_document(uri, blob)

    # Hit
    val = cache.try_lookup(uri)
    assert val == blob

    # make sure >1 item works
    uri, blob = p2
    cache.save_document(uri, blob)
    val = cache.try_lookup(uri)
    assert val == blob


if __name__ == "__main__":
    minitest()
