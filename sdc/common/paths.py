"""
sdc.common.paths
----------------
Where the data lives, resolved ONCE from the repository root.

Every script used to write `HERE = Path(__file__).resolve().parent` and then `HERE / "runs"`.
That is correct only while the script sits at the repo root, and it fails SILENTLY when it does
not: a module at `sdc/compare/` resolves `HERE / "figures"` to `sdc/compare/figures`, which
matplotlib will happily create, so figures land somewhere nobody looks instead of raising.

So the anchor is computed from THIS file's known depth in the package and exported. Modules
import the directory they need and never do their own `__file__` arithmetic, which means moving
a module between `sdc/` subpackages costs nothing.
"""
from pathlib import Path

# sdc/common/paths.py -> sdc/common -> sdc -> repo root
ROOT = Path(__file__).resolve().parents[2]

RUNS = ROOT / "runs"                 # real-data detections, one npz per recording
SIM_RUNS = ROOT / "sim_runs"         # synthetic detections, one npz per (SNR, operating point)
SIM_DATA = ROOT / "sim_data"         # synthetic EDFs + .truth.npz sidecars (gitignored)
FIGURES = ROOT / "figures"           # figures/<real|sim>/<run>/...
ARCHIVE = ROOT / "archive"           # superseded runs kept deliberately
CACHE = ROOT / ".delphos_cache"      # memoised Delphos CLI output


def figdir(*parts):
    """`FIGURES/<parts...>`, created if missing. Returns the directory."""
    d = FIGURES.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
