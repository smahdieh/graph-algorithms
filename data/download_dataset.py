"""Download the BHOSLIB / FRB benchmark graphs (MIS format) into data/frb/.

Source: https://rb-bench.github.io/benchmarks/hidden-opt-graph.html

Each archive frbN-K-mis.tar.gz holds 5 instances frbN-K-1.mis .. -5.mis.
An frbN-K graph has N*K vertices and a hidden independent set of size N,
so its minimum vertex cover has size N*K - N.

Usage:
    python data/download_dataset.py            # all 8 families
    python data/download_dataset.py 30-15 40-19   # only some families
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://rb-bench.github.io/assets/bench"
FAMILIES = ["30-15", "35-17", "40-19", "45-21", "50-23", "53-24", "56-25", "59-26"]
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "frb"


def download_family(family: str) -> list[Path]:
    """Download and unpack one FRB family.

    Input:  family -- e.g. "30-15".
    Output: paths of the .mis files now in data/frb/.
    Files that are already there are not downloaded again.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = [OUT_DIR / f"frb{family}-{i}.mis" for i in range(1, 6)]
    if all(p.exists() for p in expected):
        print(f"frb{family}: already present")
        return expected

    archive = DATA_DIR / f"frb{family}-mis.tar.gz"
    url = f"{BASE_URL}/frb{family}-mis.tar.gz"
    print(f"frb{family}: downloading {url}")
    urllib.request.urlretrieve(url, archive)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            # Keep only the regular .mis files, flattened into data/frb/.
            if member.isfile() and member.name.endswith(".mis"):
                member.name = Path(member.name).name
                tar.extract(member, OUT_DIR, filter="data")
    archive.unlink()
    return sorted(OUT_DIR.glob(f"frb{family}-*.mis"))


def optimal_cover_size(family: str) -> int:
    """Return the known minimum vertex cover size N*K - N of family "N-K"."""
    n, k = (int(x) for x in family.split("-"))
    return n * k - n


def main(families: list[str]) -> None:
    """Download the given families (all of them if the list is empty)."""
    for family in families or FAMILIES:
        if family not in FAMILIES:
            raise SystemExit(f"unknown family {family!r}; choose from {FAMILIES}")
        files = download_family(family)
        print(f"  {len(files)} files, optimal cover size {optimal_cover_size(family)}")


if __name__ == "__main__":
    main(sys.argv[1:])
