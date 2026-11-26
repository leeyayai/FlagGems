# Temporary execution probe used to verify which FlagGems operator paths are hit.
# Enable with FLAGOS_PROBE=1; limit repeats per tag with FLAGOS_PROBE_LIMIT.
import os
import sys

_ENABLED = os.environ.get("FLAGOS_PROBE", "0") == "1"
_LIMIT = int(os.environ.get("FLAGOS_PROBE_LIMIT", "3"))
_counts = {}


def probe(tag, msg):
    if not _ENABLED:
        return
    n = _counts.get(tag, 0)
    if n >= _LIMIT:
        return
    _counts[tag] = n + 1
    print(f"[PROBE][{tag}] pid={os.getpid()} n={n + 1} {msg}", file=sys.stdout, flush=True)
